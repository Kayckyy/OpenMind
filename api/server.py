#!/usr/bin/env python3
"""
OpenMind HTTP Server (sem WebSocket, sem navegador obrigatório)
================================================================
Substitui o WebSocket por dois modelos:

  1. /process  POST  — envia WAV, recebe WAV processado (batch)
  2. /play     POST  — toca diretamente no dispositivo do servidor
                       (ideal para uso local no Termux)
  3. /stream   GET   — SSE de status em tempo real
  4. /health   GET   — health check

Uso:
  python server.py                    # sobe na porta 8765
  python server.py --port 9000

Cliente de exemplo (qualquer máquina na rede local):
  curl -X POST http://192.168.x.x:8765/play \
       -H "Content-Type: application/json" \
       -d '{"source": "lofi hip hop", "az": 0, "el": 0}'

  curl -X POST http://192.168.x.x:8765/process \
       -F "file=@musica.wav" \
       -F "az=45" -F "el=0" \
       --output resultado.wav
"""

import argparse
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import soundfile as sf

# api/server.py roda de dentro de api/ — sobe um nível pra raiz do projeto
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from engine.convolution_engine import ConvolutionEngine
from engine.ir_loader import IrLoader

HRTF_DIR  = os.path.join(_ROOT, 'hrtf', 'sadie')
TARGET_SR = 44100
BLOCK     = 2048
CACHE_DIR = os.path.join(_ROOT, 'cache')
TMP_WAV   = os.path.join(CACHE_DIR, 'yt_cache.wav')
FFMPEG    = '/data/data/com.termux/files/usr/bin/ffmpeg'

# estado global simples
_status_listeners: list = []
_status_lock = threading.Lock()
_playback_thread: threading.Thread | None = None


# ---------------------------------------------------------------------------
def emit_status(msg: dict):
    """Envia status para todos os listeners SSE conectados."""
    data = f"data: {json.dumps(msg)}\n\n".encode()
    with _status_lock:
        dead = []
        for q in _status_listeners:
            try:
                q.put_nowait(data)
            except Exception:
                dead.append(q)
        for d in dead:
            _status_listeners.remove(d)


# ---------------------------------------------------------------------------
def build_engine(az: float, el: float) -> ConvolutionEngine:
    loader = IrLoader(HRTF_DIR)

    def az_str(a): return f"{a:.1f}".replace('.', ',')
    def el_str(e): return f"{e:.1f}".replace('.', ',')

    az_r = (360.0 - az) % 360.0
    try:
        _, ll, lr = loader.load(f"azi_{az_str(az)}_ele_{el_str(el)}.wav")
        _, rl, rr = loader.load(f"azi_{az_str(az_r)}_ele_{el_str(el)}.wav")
    except FileNotFoundError:
        _, ll, lr = loader.load("azi_0,0_ele_0,0.wav")
        rl, rr = ll.copy(), lr.copy()

    return ConvolutionEngine(ir_ll=ll, ir_lr=lr, ir_rl=rl, ir_rr=rr)


# ---------------------------------------------------------------------------
def download_yt(query: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(TMP_WAV):
        os.remove(TMP_WAV)

    source = query if query.startswith('http') else f'ytsearch1:{query}'

    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError('yt-dlp não instalado')

    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'outtmpl': os.path.join(CACHE_DIR, 'yt_cache.%(ext)s'),
        'ffmpeg_location': FFMPEG,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '0',
        }],
    }

    emit_status({'status': 'downloading', 'query': query})
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([source])

    return TMP_WAV


# ---------------------------------------------------------------------------
def resample_if_needed(data: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return data
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(TARGET_SR, sr)
    return resample_poly(data, TARGET_SR // g, sr // g).astype(np.float32)


# ---------------------------------------------------------------------------
def process_audio(data: np.ndarray, engine: ConvolutionEngine) -> np.ndarray:
    """Processa array (N,2) em blocos. Retorna array (N,2) processado."""
    in_l = data[:, 0]
    in_r = data[:, 1] if data.shape[1] > 1 else data[:, 0]
    out_chunks = []

    for i in range(0, len(in_l), BLOCK):
        bl = in_l[i:i + BLOCK]
        br = in_r[i:i + BLOCK]
        ol, or_ = engine.process(bl, br)
        length = min(len(ol), len(or_))
        out_chunks.append(np.stack([ol[:length], or_[:length]], axis=1))

    return np.concatenate(out_chunks, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
def play_background(source: str, az: float, el: float):
    """Toca no dispositivo de áudio do servidor (Termux)."""
    global _playback_thread

    try:
        import sounddevice as sd
        import queue as _queue

        is_yt = source.startswith('http') or not os.path.exists(source)
        wav_path = download_yt(source) if is_yt else source

        emit_status({'status': 'loading'})
        raw, sr = sf.read(wav_path, dtype='float32', always_2d=True)
        raw = resample_if_needed(raw, sr)

        engine = build_engine(az, el)
        in_l = raw[:, 0]
        in_r = raw[:, 1] if raw.shape[1] > 1 else raw[:, 0]

        buf_q: _queue.Queue = _queue.Queue(maxsize=8)
        remainder = np.zeros((0, 2), dtype=np.float32)

        def producer():
            for i in range(0, len(in_l), BLOCK):
                bl = in_l[i:i + BLOCK]
                br = in_r[i:i + BLOCK]
                ol, or_ = engine.process(bl, br)
                length = min(len(ol), len(or_))
                buf_q.put(np.stack([ol[:length], or_[:length]], axis=1))
            buf_q.put(None)

        threading.Thread(target=producer, daemon=True).start()

        def callback(outdata, frames, time_info, status):
            nonlocal remainder
            out = np.zeros((frames, 2), dtype=np.float32)
            pos = 0
            needed = frames

            if len(remainder) > 0:
                take = min(len(remainder), needed)
                out[pos:pos + take] = remainder[:take]
                remainder = remainder[take:]
                pos += take
                needed -= take

            while needed > 0:
                try:
                    chunk = buf_q.get_nowait()
                except _queue.Empty:
                    break
                if chunk is None:
                    raise sd.CallbackStop()
                take = min(len(chunk), needed)
                out[pos:pos + take] = chunk[:take]
                if take < len(chunk):
                    remainder = chunk[take:]
                pos += take
                needed -= take

            outdata[:] = out

        emit_status({'status': 'playing'})
        with sd.OutputStream(samplerate=TARGET_SR, channels=2,
                              dtype='float32', blocksize=BLOCK,
                              callback=callback):
            while not buf_q.empty() or buf_q.qsize() > 0:
                time.sleep(0.1)
            time.sleep(0.3)

        emit_status({'status': 'done'})

    except Exception as e:
        emit_status({'status': 'error', 'message': str(e)})


# ---------------------------------------------------------------------------
def parse_multipart(content_type: str, body: bytes):
    """Parser mínimo de multipart/form-data."""
    import email
    msg = email.message_from_bytes(
        f"Content-Type: {content_type}\r\n\r\n".encode() + body
    )
    result = {}
    for part in msg.walk():
        cd = part.get('Content-Disposition', '')
        if 'name=' not in cd:
            continue
        name = cd.split('name=')[1].strip('"').split(';')[0].strip('"')
        payload = part.get_payload(decode=True)
        if payload is not None:
            result[name] = payload
    return result


# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f'[http] {self.address_string()} {fmt % args}')

    def _send_json(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_wav(self, audio: np.ndarray):
        buf = io.BytesIO()
        sf.write(buf, audio, TARGET_SR, format='WAV', subtype='FLOAT')
        body = buf.getvalue()
        self.send_response(200)
        self.send_header('Content-Type', 'audio/wav')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Content-Disposition', 'attachment; filename=output.wav')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path

        if path == '/health':
            self._send_json(200, {'status': 'ok'})

        elif path == '/stream':
            # SSE
            import queue as _queue
            q: _queue.Queue = _queue.Queue()
            with _status_lock:
                _status_listeners.append(q)

            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            try:
                while True:
                    try:
                        data = q.get(timeout=15)
                        self.wfile.write(data)
                        self.wfile.flush()
                    except _queue.Empty:
                        self.wfile.write(b': ping\n\n')
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with _status_lock:
                    if q in _status_listeners:
                        _status_listeners.remove(q)
        else:
            self._send_json(404, {'error': 'not found'})

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        ct = self.headers.get('Content-Type', '')

        if path == '/play':
            # JSON body: {"source": "...", "az": 0, "el": 0}
            try:
                params = json.loads(body)
            except Exception:
                self._send_json(400, {'error': 'JSON inválido'})
                return

            source = params.get('source', '')
            az = float(params.get('az', 0.0))
            el = float(params.get('el', 0.0))

            if not source:
                self._send_json(400, {'error': 'source obrigatório'})
                return

            global _playback_thread
            if _playback_thread and _playback_thread.is_alive():
                self._send_json(409, {'error': 'já tocando'})
                return

            _playback_thread = threading.Thread(
                target=play_background, args=(source, az, el), daemon=True
            )
            _playback_thread.start()
            self._send_json(200, {'status': 'started'})

        elif path == '/process':
            # multipart/form-data: file + az + el
            try:
                parts = parse_multipart(ct, body)
                wav_bytes = parts.get(b'file') or parts.get('file')
                az = float((parts.get(b'az') or parts.get('az', b'0')).decode())
                el = float((parts.get(b'el') or parts.get('el', b'0')).decode())
            except Exception as e:
                self._send_json(400, {'error': f'parse error: {e}'})
                return

            try:
                raw, sr = sf.read(io.BytesIO(wav_bytes), dtype='float32',
                                  always_2d=True)
                raw = resample_if_needed(raw, sr)
                engine = build_engine(az, el)
                output = process_audio(raw, engine)
                self._send_wav(output)
            except Exception as e:
                self._send_json(500, {'error': str(e)})

        else:
            self._send_json(404, {'error': 'not found'})

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='OpenMind HTTP Server')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f'[server] OpenMind rodando em http://{args.host}:{args.port}')
    print(f'[server] curl -X POST http://localhost:{args.port}/play \\')
    print(f'         -H "Content-Type: application/json" \\')
    print(f"         -d '{{\"source\": \"nome da musica\", \"az\": 0, \"el\": 0}}'")

    server = HTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n[server] encerrado')


if __name__ == '__main__':
    main()
          

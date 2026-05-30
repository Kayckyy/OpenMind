#!/usr/bin/env python3
"""
OpenMind HTTP Server
=====================
Endpoints:
  POST /stream   — envia arquivo, recebe áudio f32le chunked em tempo real
                   uso: curl ... | ffplay -f f32le -ar 44100 -ac 2 -
  POST /process  — envia arquivo, recebe WAV completo processado
  POST /play     — toca no dispositivo do tablet diretamente
  GET  /status   — SSE de status (downloading/processing/done/error)
  GET  /health   — health check

Uso rápido no celular:
  curl -s -X POST http://IP:8765/stream \
    -F "file=@/storage/emulated/0/Music/musica.mp3" \
    -F "az=0" -F "el=0" | ffplay -nodisp -f f32le -ar 44100 -ac 2 -
"""

import argparse
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

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

_status_listeners: list = []
_status_lock = threading.Lock()
_playback_thread: threading.Thread | None = None


# ---------------------------------------------------------------------------
def emit_status(msg: dict):
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
def resample_if_needed(data: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return data
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(TARGET_SR, sr)
    return resample_poly(data, TARGET_SR // g, sr // g).astype(np.float32)


# ---------------------------------------------------------------------------
def parse_multipart(content_type: str, body: bytes) -> dict:
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
def load_audio(file_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """
    Lê arquivo de qualquer formato suportado pelo soundfile/ffmpeg.
    Retorna (in_l, in_r) como float32 em TARGET_SR.
    """
    # tenta soundfile direto (WAV, FLAC, OGG, AIFF...)
    try:
        raw, sr = sf.read(io.BytesIO(file_bytes), dtype='float32', always_2d=True)
    except Exception:
        # fallback: decodifica via ffmpeg pra WAV temporário
        import subprocess
        tmp = os.path.join(CACHE_DIR, 'upload_tmp.wav')
        proc = subprocess.run(
            [FFMPEG, '-y', '-i', 'pipe:0',
             '-f', 'wav', '-ar', str(TARGET_SR), '-ac', '2', tmp],
            input=file_bytes, capture_output=True
        )
        if proc.returncode != 0:
            raise RuntimeError('ffmpeg falhou ao decodificar arquivo')
        raw, sr = sf.read(tmp, dtype='float32', always_2d=True)

    raw = resample_if_needed(raw, sr)
    in_l = raw[:, 0]
    in_r = raw[:, 1] if raw.shape[1] > 1 else raw[:, 0]
    return in_l, in_r


# ---------------------------------------------------------------------------
def stream_generator(in_l, in_r, engine):
    """Gera chunks f32le interleaved processados bloco a bloco."""
    for i in range(0, len(in_l), BLOCK):
        bl = in_l[i:i + BLOCK]
        br = in_r[i:i + BLOCK]
        ol, or_ = engine.process(bl, br)
        length = min(len(ol), len(or_))
        # interleave: L R L R L R ...
        out = np.empty(length * 2, dtype=np.float32)
        out[0::2] = ol[:length]
        out[1::2] = or_[:length]
        yield out.tobytes()


# ---------------------------------------------------------------------------
def play_background(source: str, az: float, el: float):
    global _playback_thread
    try:
        import sounddevice as sd
        import queue as _queue

        is_remote = source.startswith('http') or not os.path.exists(source)
        if is_remote:
            raise RuntimeError('play só funciona com arquivos locais do tablet')

        emit_status({'status': 'loading'})
        with open(source, 'rb') as f:
            file_bytes = f.read()
        in_l, in_r = load_audio(file_bytes)

        engine = build_engine(az, el)
        buf_q: _queue.Queue = _queue.Queue(maxsize=8)
        remainder = np.zeros((0, 2), dtype=np.float32)

        def producer():
            for chunk_bytes in stream_generator(in_l, in_r, engine):
                samples = np.frombuffer(chunk_bytes, dtype=np.float32)
                stereo = samples.reshape(-1, 2)
                buf_q.put(stereo)
            buf_q.put(None)

        threading.Thread(target=producer, daemon=True).start()

        def callback(outdata, frames, time_info, status):
            nonlocal remainder
            out = np.zeros((frames, 2), dtype=np.float32)
            pos, needed = 0, frames
            if len(remainder):
                take = min(len(remainder), needed)
                out[pos:pos+take] = remainder[:take]
                remainder = remainder[take:]
                pos += take; needed -= take
            while needed > 0:
                try:
                    chunk = buf_q.get_nowait()
                except _queue.Empty:
                    break
                if chunk is None:
                    raise sd.CallbackStop()
                take = min(len(chunk), needed)
                out[pos:pos+take] = chunk[:take]
                if take < len(chunk):
                    remainder = chunk[take:]
                pos += take; needed -= take
            outdata[:] = out

        emit_status({'status': 'playing'})
        with sd.OutputStream(samplerate=TARGET_SR, channels=2,
                              dtype='float32', blocksize=BLOCK,
                              callback=callback):
            while buf_q.qsize() > 0 or (
                _playback_thread and _playback_thread.is_alive()
            ):
                time.sleep(0.05)
            time.sleep(0.3)

        emit_status({'status': 'done'})
    except Exception as e:
        emit_status({'status': 'error', 'message': str(e)})


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

    # -----------------------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path

        if path == '/health':
            self._send_json(200, {'status': 'ok'})

        elif path == '/status':
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

    # -----------------------------------------------------------------------
    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        ct = self.headers.get('Content-Type', '')

        # --- /stream: chunked f32le de volta pro cliente ---
        if path == '/stream':
            try:
                parts = parse_multipart(ct, body)
                file_bytes = parts.get(b'file') or parts.get('file')
                az = float((parts.get(b'az') or parts.get('az', b'0')).decode())
                el = float((parts.get(b'el') or parts.get('el', b'0')).decode())
            except Exception as e:
                self._send_json(400, {'error': f'parse error: {e}'})
                return

            try:
                in_l, in_r = load_audio(file_bytes)
                engine = build_engine(az, el)
            except Exception as e:
                self._send_json(500, {'error': str(e)})
                return

            self.send_response(200)
            self.send_header('Content-Type', 'audio/x-raw-f32le')
            self.send_header('X-Sample-Rate', str(TARGET_SR))
            self.send_header('X-Channels', '2')
            self.send_header('Transfer-Encoding', 'chunked')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            try:
                for chunk_bytes in stream_generator(in_l, in_r, engine):
                    # formato chunked HTTP: tamanho em hex + CRLF + dados + CRLF
                    size_line = f'{len(chunk_bytes):X}\r\n'.encode()
                    self.wfile.write(size_line)
                    self.wfile.write(chunk_bytes)
                    self.wfile.write(b'\r\n')
                    self.wfile.flush()
                # chunk final de tamanho 0 = fim do stream
                self.wfile.write(b'0\r\n\r\n')
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # cliente desconectou antes do fim — normal

        # --- /process: WAV completo ---
        elif path == '/process':
            try:
                parts = parse_multipart(ct, body)
                file_bytes = parts.get(b'file') or parts.get('file')
                az = float((parts.get(b'az') or parts.get('az', b'0')).decode())
                el = float((parts.get(b'el') or parts.get('el', b'0')).decode())
            except Exception as e:
                self._send_json(400, {'error': f'parse error: {e}'})
                return

            try:
                in_l, in_r = load_audio(file_bytes)
                engine = build_engine(az, el)
                chunks = list(stream_generator(in_l, in_r, engine))
                raw = np.frombuffer(b''.join(chunks), dtype=np.float32)
                stereo = raw.reshape(-1, 2)
                self._send_wav(stereo)
            except Exception as e:
                self._send_json(500, {'error': str(e)})

        # --- /play: toca no tablet ---
        elif path == '/play':
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

    # descobre IP local pra facilitar
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = 'localhost'

    print(f'[server] OpenMind rodando em http://{ip}:{args.port}')
    print()
    print(f'  # streaming (celular → tablet → celular):')
    print(f'  curl -s -X POST http://{ip}:{args.port}/stream \\')
    print(f'    -F "file=@musica.mp3" -F "az=0" -F "el=0" \\')
    print(f'    | ffplay -nodisp -f f32le -ar {TARGET_SR} -ac 2 -')
    print()
    print(f'  # download WAV processado:')
    print(f'  curl -X POST http://{ip}:{args.port}/process \\')
    print(f'    -F "file=@musica.mp3" -F "az=0" -F "el=0" -o saida.wav')

    server = HTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n[server] encerrado')


if __name__ == '__main__':
    main()

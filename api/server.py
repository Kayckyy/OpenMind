#!/usr/bin/env python3
"""
OpenMind HTTP Server
  GET  /stream?file=musica.opus&az=90&el=0  — streaming WAV binaural
  GET  /process?file=musica.opus&az=90&el=0 — WAV completo para download
  GET  /status                              — SSE de status
  GET  /health                              — health check
  POST /play  {"file":"...","az":90,"el":0} — toca no tablet

Pasta de músicas: ~/Music  (configurável em MUSIC_DIR)
"""

import argparse, io, json, os, struct, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs, unquote

import numpy as np
import soundfile as sf

_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from engine.convolution_engine import ConvolutionEngine
from engine.ir_loader import IrLoader

HRTF_DIR   = os.path.join(_ROOT, 'hrtf', 'sadie')
MUSIC_DIR  = os.path.expanduser('~/Music')   # onde o servidor procura os arquivos
TARGET_SR  = 44100
BLOCK      = 2048

import shutil as _sh
FFMPEG = _sh.which('ffmpeg') or '/data/data/com.termux/files/usr/bin/ffmpeg'

_status_listeners, _status_lock = [], threading.Lock()
_playback_thread = None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

_IR_MAP = {
    0:   "azi_0,0_ele_0,0.wav",
    90:  "azi_90,0_ele_0,0.wav",
    180: "azi_180,0_ele_0,0.wav",
    270: "azi_270,0_ele_0,0.wav",
}

def _nearest_ir(az):
    az = az % 360.0
    return _IR_MAP[min(_IR_MAP, key=lambda k: min(abs(k-az), 360-abs(k-az)))]

def build_engine(az, el):
    loader = IrLoader(HRTF_DIR)
    _, ll, _  = loader.load(_nearest_ir(az))
    _, __, lr = loader.load("azi_0,0_ele_0,0.wav")
    _, rl, _  = loader.load("azi_0,0_ele_0,0.wav")
    _, __, rr = loader.load(_nearest_ir((360.0 - az) % 360.0))
    return ConvolutionEngine(ir_ll=ll, ir_lr=lr*0.25, ir_rl=rl*0.25, ir_rr=rr)


# ---------------------------------------------------------------------------
# Audio pipeline
# ---------------------------------------------------------------------------

def resolve_path(filename):
    """Aceita caminho absoluto ou nome relativo à MUSIC_DIR."""
    if os.path.isabs(filename):
        path = filename
    else:
        path = os.path.join(MUSIC_DIR, unquote(filename))
    if not os.path.exists(path):
        raise FileNotFoundError(f'arquivo não encontrado: {path}')
    return path

def wav_stream_header(sr, channels):
    byte_rate = sr * channels * 4
    return struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 0xFFFFFFFF, b'WAVE',
        b'fmt ', 16, 3, channels, sr,
        byte_rate, channels*4, 32,
        b'data', 0xFFFFFFFF)

def stream_audio(path, engine):
    """Decodifica arquivo via ffmpeg e processa em blocos. Yields bytes f32le."""
    FRAME  = 2 * 4
    RBYTES = BLOCK * FRAME

    proc = subprocess.Popen(
        [FFMPEG, '-y', '-i', path,
         '-f', 'f32le', '-ar', str(TARGET_SR), '-ac', '2', 'pipe:1'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    remainder = b''
    try:
        while True:
            chunk = proc.stdout.read(RBYTES)
            if not chunk:
                proc.stdout.close(); proc.wait(timeout=2); break
            chunk     = remainder + chunk
            remainder = b''
            usable    = (len(chunk) // FRAME) * FRAME
            if usable == 0:    remainder = chunk; continue
            if usable < len(chunk): remainder = chunk[usable:]; chunk = chunk[:usable]
            s = np.frombuffer(chunk, dtype=np.float32)
            ol, or_ = engine.process(s[0::2], s[1::2])
            n = min(len(ol), len(or_))
            out = np.empty(n * 2, dtype=np.float32)
            out[0::2] = ol[:n]; out[1::2] = or_[:n]
            yield out.tobytes()
    except Exception: pass
    finally:
        try: proc.stdout.close()
        except: pass
        try: proc.kill(); proc.wait(timeout=2)
        except: pass


# ---------------------------------------------------------------------------
# SSE status
# ---------------------------------------------------------------------------

def emit_status(msg):
    data = f"data: {json.dumps(msg)}\n\n".encode()
    with _status_lock:
        dead = []
        for q in _status_listeners:
            try: q.put_nowait(data)
            except: dead.append(q)
        for d in dead: _status_listeners.remove(d)


# ---------------------------------------------------------------------------
# /play (toca no tablet via sounddevice)
# ---------------------------------------------------------------------------

def play_background(path, az, el):
    try:
        import sounddevice as sd, queue as Q
        emit_status({'status': 'playing', 'file': os.path.basename(path)})
        engine = build_engine(az, el)
        buf_q  = Q.Queue(maxsize=8)
        rem    = np.zeros((0,2), dtype=np.float32)

        def producer():
            for b in stream_audio(path, engine):
                buf_q.put(np.frombuffer(b, dtype=np.float32).reshape(-1,2))
            buf_q.put(None)

        threading.Thread(target=producer, daemon=True).start()

        def callback(outdata, frames, t, s):
            nonlocal rem
            out = np.zeros((frames,2), dtype=np.float32)
            pos, need = 0, frames
            if len(rem):
                take = min(len(rem), need)
                out[pos:pos+take] = rem[:take]; rem = rem[take:]; pos+=take; need-=take
            while need > 0:
                try: chunk = buf_q.get_nowait()
                except Q.Empty: break
                if chunk is None: raise sd.CallbackStop()
                take = min(len(chunk), need)
                out[pos:pos+take] = chunk[:take]
                if take < len(chunk): rem = chunk[take:]
                pos+=take; need-=take
            outdata[:] = out

        with sd.OutputStream(samplerate=TARGET_SR, channels=2,
                              dtype='float32', blocksize=BLOCK, callback=callback):
            while buf_q.qsize() > 0: time.sleep(0.05)
            time.sleep(0.5)
        emit_status({'status': 'done'})
    except Exception as e:
        emit_status({'status': 'error', 'message': str(e)})


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f'[http] {self.address_string()} {fmt % args}')

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def _params(self):
        """Extrai query params: file, az, el."""
        qs   = parse_qs(urlparse(self.path).query)
        file = unquote(qs.get('file', [''])[0])
        az   = float(qs.get('az', [0])[0])
        el   = float(qs.get('el', [0])[0])
        return file, az, el

    def do_GET(self):
        path = urlparse(self.path).path

        if path == '/health':
            self._json(200, {'status': 'ok', 'music_dir': MUSIC_DIR})

        elif path == '/status':
            import queue as Q
            q = Q.Queue()
            with _status_lock: _status_listeners.append(q)
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()
            try:
                while True:
                    try: self.wfile.write(q.get(timeout=15)); self.wfile.flush()
                    except Q.Empty: self.wfile.write(b': ping\n\n'); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError): pass
            finally:
                with _status_lock:
                    if q in _status_listeners: _status_listeners.remove(q)

        elif path == '/stream':
            try:
                filename, az, el = self._params()
                fpath  = resolve_path(filename)
                engine = build_engine(az, el)
            except Exception as e:
                self._json(400, {'error': str(e)}); return

            self.send_response(200)
            self.send_header('Content-Type', 'audio/wav')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            try:
                self.wfile.write(wav_stream_header(TARGET_SR, 2))
                for chunk in stream_audio(fpath, engine):
                    self.wfile.write(chunk); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError): pass

        elif path == '/process':
            try:
                filename, az, el = self._params()
                fpath  = resolve_path(filename)
                engine = build_engine(az, el)
                chunks = list(stream_audio(fpath, engine))
                raw    = np.frombuffer(b''.join(chunks), dtype=np.float32).reshape(-1,2)
                buf    = io.BytesIO()
                sf.write(buf, raw, TARGET_SR, format='WAV', subtype='FLOAT')
                body   = buf.getvalue()
                self.send_response(200)
                self.send_header('Content-Type', 'audio/wav')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Content-Disposition', 'attachment; filename=output.wav')
                self.end_headers(); self.wfile.write(body)
            except Exception as e:
                self._json(500, {'error': str(e)})

        else:
            self._json(404, {'error': 'not found'})

    def do_POST(self):
        if urlparse(self.path).path != '/play':
            self._json(404, {'error': 'not found'}); return
        try:
            params = json.loads(self.rfile.read(int(self.headers.get('Content-Length',0))))
            fpath  = resolve_path(params.get('file',''))
        except Exception as e:
            self._json(400, {'error': str(e)}); return
        global _playback_thread
        if _playback_thread and _playback_thread.is_alive():
            self._json(409, {'error': 'já tocando'}); return
        _playback_thread = threading.Thread(
            target=play_background,
            args=(fpath, float(params.get('az',90)), float(params.get('el',0))),
            daemon=True)
        _playback_thread.start()
        self._json(200, {'status': 'started', 'file': os.path.basename(fpath)})

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--music-dir', default=MUSIC_DIR)
    args = parser.parse_args()

    global MUSIC_DIR
    MUSIC_DIR = os.path.expanduser(args.music_dir)
    os.makedirs(MUSIC_DIR, exist_ok=True)

    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80)); ip = s.getsockname()[0]; s.close()
    except: ip = 'localhost'

    print(f'[server] OpenMind em http://{ip}:{args.port}')
    print(f'[server] música em {MUSIC_DIR}')
    print(f'\n  ffplay -nodisp "http://{ip}:{args.port}/stream?file=musica.opus&az=90"')

    try: HTTPServer((args.host, args.port), Handler).serve_forever()
    except KeyboardInterrupt: print('\n[server] encerrado')

if __name__ == '__main__':
    main()
          

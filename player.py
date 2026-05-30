#!/usr/bin/env python3
"""
OpenMind CLI Player
====================
Uso:
  python player.py "nome da música ou URL"
  python player.py "URL" --az 0 --el 0
  python player.py arquivo.wav --az 45 --el 15
  python player.py --list-devices

Dependências:
  pip install numpy scipy soundfile sounddevice yt-dlp
  (sounddevice no Termux: pkg install portaudio && pip install sounddevice)
"""

import argparse
import os
import sys
import queue
import threading
import time

import numpy as np
import soundfile as sf
import sounddevice as sd

# importa do mesmo diretório
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.convolution_engine import ConvolutionEngine
from engine.ir_loader import IrLoader

# ---------------------------------------------------------------------------
HRTF_DIR  = os.path.join(os.path.dirname(__file__), '..', 'hrtf', 'sadie')
TARGET_SR = 44100
BLOCK     = 2048          # bloco menor = latência menor (vs 4096 anterior)
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'cache')
TMP_WAV   = os.path.join(CACHE_DIR, 'yt_cache.wav')
FFMPEG    = '/data/data/com.termux/files/usr/bin/ffmpeg'


# ---------------------------------------------------------------------------
def build_engine(az: float, el: float) -> ConvolutionEngine:
    """
    Carrega IR de azimute/elevação e monta True-Stereo engine.
    SADIE II usa arquivos stereo: canal L = HRTF ipsilateral, R = contralateral.
    True-Stereo:
      ir_ll = IR do lado esquerdo, ouvido esquerdo
      ir_lr = IR do lado esquerdo, ouvido direito   (crossfeed)
      ir_rl = IR do lado direito,  ouvido esquerdo  (crossfeed)
      ir_rr = IR do lado direito,  ouvido direito
    """
    loader = IrLoader(HRTF_DIR)

    def az_str(a): return f"{a:.1f}".replace('.', ',')
    def el_str(e): return f"{e:.1f}".replace('.', ',')

    # IR principal (posição solicitada)
    fname_l = f"azi_{az_str(az)}_ele_{el_str(el)}.wav"
    # IR espelhada para o canal direito (azimute espelhado)
    az_r = (360.0 - az) % 360.0
    fname_r = f"azi_{az_str(az_r)}_ele_{el_str(el)}.wav"

    try:
        _, ll, lr = loader.load(fname_l)  # L → out_L, L → out_R
        _, rl, rr = loader.load(fname_r)  # R → out_L (cross), R → out_R
    except FileNotFoundError as e:
        print(f"[erro] IR não encontrada: {e}")
        print(f"       Tentando fallback em azi_0,0_ele_0,0.wav")
        _, ll, lr = loader.load("azi_0,0_ele_0,0.wav")
        _, rl, rr = loader.load("azi_0,0_ele_0,0.wav")

    return ConvolutionEngine(ir_ll=ll, ir_lr=lr, ir_rl=rl, ir_rr=rr,
                             sample_rate=TARGET_SR)


# ---------------------------------------------------------------------------
def download_yt(query: str) -> str:
    """Baixa do YouTube e retorna caminho do WAV."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(TMP_WAV):
        os.remove(TMP_WAV)

    source = query if query.startswith('http') else f'ytsearch1:{query}'

    try:
        import yt_dlp
    except ImportError:
        print('[erro] yt-dlp não instalado. pip install yt-dlp')
        sys.exit(1)

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

    print(f'[yt]  baixando: {query}')
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([source])

    if not os.path.exists(TMP_WAV):
        print('[erro] download falhou')
        sys.exit(1)

    return TMP_WAV


# ---------------------------------------------------------------------------
def resample_if_needed(data: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return data
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(TARGET_SR, sr)
        return resample_poly(data, TARGET_SR // g, sr // g).astype(np.float32)
    except ImportError:
        print('[aviso] scipy não disponível, sem resampling')
        return data


# ---------------------------------------------------------------------------
def play(source: str, az: float, el: float, device: int | None):
    # Resolve fonte
    is_yt = source.startswith('http') or (
        not os.path.exists(source) and not source.endswith('.wav')
    )

    if is_yt:
        wav_path = download_yt(source)
    else:
        wav_path = source

    print(f'[load] lendo {wav_path}')
    data, sr = sf.read(wav_path, dtype='float32', always_2d=True)
    data = resample_if_needed(data, sr)

    in_l = data[:, 0]
    in_r = data[:, 1] if data.shape[1] > 1 else data[:, 0]

    print(f'[hrtf] az={az}° el={el}°  |  {len(in_l)/TARGET_SR:.1f}s  |  bloco={BLOCK}')
    engine = build_engine(az, el)

    # fila de blocos já processados
    # backpressure real: produtor para se fila estiver cheia
    buf_q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=8)

    # --- thread produtora: processa em blocos e enfileira ---
    def producer():
        n = len(in_l)
        for i in range(0, n, BLOCK):
            bl = in_l[i:i + BLOCK]
            br = in_r[i:i + BLOCK]
            out_l, out_r = engine.process(bl, br)
            length = min(len(out_l), len(out_r))
            stereo = np.stack([out_l[:length], out_r[:length]], axis=1)
            buf_q.put(stereo)   # bloqueia se fila cheia (backpressure)
        buf_q.put(None)         # sentinel

    prod_thread = threading.Thread(target=producer, daemon=True)
    prod_thread.start()

    # --- callback do sounddevice (thread de áudio) ---
    # recebe blocos da fila e preenche o buffer de saída
    remainder = np.zeros((0, 2), dtype=np.float32)

    def callback(outdata: np.ndarray, frames: int,
                 time_info, status):
        nonlocal remainder

        needed = frames
        out = np.zeros((frames, 2), dtype=np.float32)
        pos = 0

        # drena remainder primeiro
        if len(remainder) > 0:
            take = min(len(remainder), needed)
            out[pos:pos + take] = remainder[:take]
            remainder = remainder[take:]
            pos += take
            needed -= take

        # busca blocos da fila enquanto precisar
        while needed > 0:
            try:
                chunk = buf_q.get_nowait()
            except queue.Empty:
                break  # underrun silencioso — melhor que travar

            if chunk is None:
                raise sd.CallbackStop()

            take = min(len(chunk), needed)
            out[pos:pos + take] = chunk[:take]
            remainder = np.vstack([remainder, chunk[take:]]) if take < len(chunk) else remainder
            pos += take
            needed -= take

        outdata[:] = out

    print('[play] reproduzindo...')
    try:
        with sd.OutputStream(
            samplerate=TARGET_SR,
            channels=2,
            dtype='float32',
            blocksize=BLOCK,
            device=device,
            callback=callback,
        ):
            prod_thread.join()
            # espera até a fila esvaziar + buffer de saída drenar
            while not buf_q.empty():
                time.sleep(0.05)
            time.sleep(0.3)  # margem pra último bloco tocar

    except KeyboardInterrupt:
        print('\n[stop]')


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='OpenMind CLI — player binaural True-Stereo'
    )
    parser.add_argument('source', nargs='?',
                        help='Arquivo WAV, URL do YouTube ou nome de música')
    parser.add_argument('--az', type=float, default=0.0,
                        help='Azimute em graus (0=frente, 90=esquerda, 270=direita)')
    parser.add_argument('--el', type=float, default=0.0,
                        help='Elevação em graus')
    parser.add_argument('--device', type=int, default=None,
                        help='Índice do dispositivo de áudio (ver --list-devices)')
    parser.add_argument('--list-devices', action='store_true',
                        help='Lista dispositivos de áudio disponíveis')

    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if not args.source:
        parser.print_help()
        sys.exit(1)

    play(args.source, args.az, args.el, args.device)


if __name__ == '__main__':
    main()
      

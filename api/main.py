import io
import os
import asyncio
import numpy as np
import soundfile as sf
import yt_dlp
from fastapi import FastAPI, UploadFile, File, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from engine.ir_loader import IrLoader
from engine.resampler import Resampler
from engine.convolution_engine import ConvolutionEngine

app = FastAPI(title="OpenMind Audio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HRTF_DIR = "hrtf/sadie"
TARGET_SR = 44100
CHUNK = 4096
TMP_PATH = "/data/data/com.termux/files/home/OpenMind/cache/yt_cache.wav"
loader = IrLoader(HRTF_DIR)
resampler = Resampler()
audio_cache: dict = {}


def ir_filename(az: float, el: float) -> str:
    az_str = f"{az:.1f}".replace(".", ",")
    el_str = f"{el:.1f}".replace(".", ",")
    return f"azi_{az_str}_ele_{el_str}.wav"


def load_engine(az_l: float, el_l: float,
                az_r: float, el_r: float) -> ConvolutionEngine:
    _, ll_l, ll_r = loader.load(ir_filename(az_l, el_l))
    _, lr_l, lr_r = loader.load(ir_filename(az_l, el_l))
    _, rl_l, rl_r = loader.load(ir_filename(az_r, el_r))
    _, rr_l, rr_r = loader.load(ir_filename(az_r, el_r))

    return ConvolutionEngine(
        ir_ll=ll_l, ir_lr=ll_r,
        ir_rl=rl_l, ir_rr=rl_r
    )


def resolve_query(query: str) -> str:
    """Se for URL do YouTube, retorna ela. Senão, faz busca pelo nome."""
    if query.startswith("http"):
        return query
    return f"ytsearch1:{query}"


def download_audio(query: str) -> np.ndarray:
    source = resolve_query(query)

    if source in audio_cache:
        return audio_cache[source]

    # Remove cache anterior
    if os.path.exists(TMP_PATH):
        os.remove(TMP_PATH)

    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'outtmpl': '/data/data/com.termux/files/home/OpenMind/cache/yt_cache.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '0',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([source])

    data, sr = sf.read(TMP_PATH)
    data = data.astype(np.float32)

    if sr != TARGET_SR:
        data = resampler.resample(data, sr)

    audio_cache[source] = data
    return data


@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    az_l: float = Query(90.0),
    el_l: float = Query(0.0),
    az_r: float = Query(270.0),
    el_r: float = Query(0.0),
):
    data, sr = sf.read(io.BytesIO(await file.read()))
    data = data.astype(np.float32)

    if sr != TARGET_SR:
        data = resampler.resample(data, sr)

    if data.ndim == 1:
        input_l = input_r = data
    else:
        input_l = data[:, 0]
        input_r = data[:, 1]

    engine = load_engine(az_l, el_l, az_r, el_r)
    out_l, out_r = engine.process(input_l, input_r)

    length = min(len(out_l), len(out_r))
    output = np.stack([out_l[:length], out_r[:length]], axis=1)

    buf = io.BytesIO()
    sf.write(buf, output, TARGET_SR, format="WAV", subtype="FLOAT")
    buf.seek(0)

    return StreamingResponse(buf, media_type="audio/wav",
                             headers={"Content-Disposition":
                                      "attachment; filename=output.wav"})


@app.websocket("/stream")
async def stream_audio(websocket: WebSocket):
    await websocket.accept()

    try:
        params = await websocket.receive_json()
        az_l = float(params.get("az_l", 90.0))
        el_l = float(params.get("el_l", 0.0))
        az_r = float(params.get("az_r", 270.0))
        el_r = float(params.get("el_r", 0.0))

        engine = load_engine(az_l, el_l, az_r, el_r)
        engine.reset()

        while True:
            data = await websocket.receive_bytes()

            if data == b"END":
                break

            samples = np.frombuffer(data, dtype=np.float32)

            if len(samples) % 2 == 0:
                input_l = samples[0::2]
                input_r = samples[1::2]
            else:
                input_l = input_r = samples

            out_l, out_r = engine.process(input_l, input_r)

            output = np.empty(len(out_l) + len(out_r), dtype=np.float32)
            output[0::2] = out_l
            output[1::2] = out_r

            await websocket.send_bytes(output.tobytes())

    except WebSocketDisconnect:
        pass


@app.websocket("/stream/youtube")
async def stream_youtube(websocket: WebSocket):
    await websocket.accept()

    try:
        params = await websocket.receive_json()
        query = params.get("query", "")
        az_l = float(params.get("az_l", 90.0))
        el_l = float(params.get("el_l", 0.0))
        az_r = float(params.get("az_r", 270.0))
        el_r = float(params.get("el_r", 0.0))

        if not query:
            await websocket.close(code=1008)
            return

        await websocket.send_json({"status": "downloading"})

        audio = await asyncio.get_event_loop().run_in_executor(
            None, download_audio, query
        )

        await websocket.send_json({"status": "processing"})

        engine = load_engine(az_l, el_l, az_r, el_r)
        engine.reset()

        if audio.ndim == 1:
            input_l = input_r = audio
        else:
            input_l = audio[:, 0]
            input_r = audio[:, 1]

        for i in range(0, len(input_l), CHUNK):
            block_l = input_l[i:i + CHUNK]
            block_r = input_r[i:i + CHUNK]

            out_l, out_r = engine.process(block_l, block_r)

            output = np.empty(len(out_l) + len(out_r), dtype=np.float32)
            output[0::2] = out_l
            output[1::2] = out_r

            await websocket.send_bytes(output.tobytes())
            await asyncio.sleep(0)

        await websocket.send_json({"status": "done"})

    except WebSocketDisconnect:
        pass


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

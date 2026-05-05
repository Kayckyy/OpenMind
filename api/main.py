import io
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Query, WebSocket, WebSocketDisconnect
import struct
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from engine.ir_loader import IrLoader
from engine.resampler import Resampler
from engine.convolution_engine import ConvolutionEngine
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="OpenMind Audio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HRTF_DIR = "hrtf/sadie"
TARGET_SR = 44100

loader = IrLoader(HRTF_DIR)
resampler = Resampler()


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


@app.get("/health")
def health():
    return {"status": "ok"}

@app.websocket("/stream")
async def stream_audio(websocket: WebSocket):
    await websocket.accept()

    try:
        # Recebe parâmetros iniciais como JSON
        params = await websocket.receive_json()
        az_l = float(params.get("az_l", 90.0))
        el_l = float(params.get("el_l", 0.0))
        az_r = float(params.get("az_r", 270.0))
        el_r = float(params.get("el_r", 0.0))

        engine = load_engine(az_l, el_l, az_r, el_r)
        engine.reset()

        # Processa chunks
        while True:
            data = await websocket.receive_bytes()

            # Sinal de fim
            if data == b"END":
                break

            # Converte bytes pra float32
            samples = np.frombuffer(data, dtype=np.float32)

            # Separa L e R (intercalado: L0, R0, L1, R1...)
            if len(samples) % 2 == 0:
                input_l = samples[0::2]
                input_r = samples[1::2]
            else:
                input_l = input_r = samples

            # Processa
            out_l, out_r = engine.process(input_l, input_r)

            # Intercala L e R de volta
            output = np.empty(len(out_l) + len(out_r), dtype=np.float32)
            output[0::2] = out_l
            output[1::2] = out_r

            await websocket.send_bytes(output.tobytes())

    except WebSocketDisconnect:
        pass

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

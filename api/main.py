import io
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from engine import IrLoader, Resampler, ConvolutionEngine

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


def load_engine(az_l: float, el_l: float,
                az_r: float, el_r: float) -> ConvolutionEngine:
    def ir_filename(az: float, el: float) -> str:
        # Adapte conforme a nomenclatura dos seus arquivos SADIE II
        return f"az{int(az)}_el{int(el)}.wav"

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
    # Lê o arquivo de entrada
    data, sr = sf.read(io.BytesIO(await file.read()))
    data = data.astype(np.float32)

    # Resample se necessário
    if sr != TARGET_SR:
        data = resampler.resample(data, sr)

    # Separa canais
    if data.ndim == 1:
        input_l = input_r = data
    else:
        input_l = data[:, 0]
        input_r = data[:, 1]

    # Processa
    engine = load_engine(az_l, el_l, az_r, el_r)
    out_l, out_r = engine.process(input_l, input_r)

    # Monta stereo de saída
    output = np.stack([out_l[:min(len(out_l), len(out_r))],
                       out_r[:min(len(out_l), len(out_r))]], axis=1)

    # Exporta como WAV em memória
    buf = io.BytesIO()
    sf.write(buf, output, TARGET_SR, format="WAV", subtype="FLOAT")
    buf.seek(0)

    return StreamingResponse(buf, media_type="audio/wav",
                             headers={"Content-Disposition":
                                      "attachment; filename=output.wav"})


@app.get("/health")
def health():
    return {"status": "ok"}

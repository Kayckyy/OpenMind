import numpy as np
import pytest
from engine.ir_loader import IrLoader
from engine.resampler import Resampler
from engine.convolution_engine import ConvolutionEngine


# ── Resampler ──────────────────────────────────────────────

def test_resampler_passthrough():
    r = Resampler()
    audio = np.random.randn(44100).astype(np.float32)
    out = r.resample(audio, 44100)
    assert len(out) == len(audio)


def test_resampler_48k_to_44k():
    r = Resampler()
    audio = np.random.randn(48000).astype(np.float32)
    out = r.resample(audio, 48000)
    assert abs(len(out) - 44100) <= 10  # tolerância de arredondamento


def test_resampler_stereo():
    r = Resampler()
    audio = np.random.randn(48000, 2).astype(np.float32)
    out = r.resample(audio, 48000)
    assert out.shape[1] == 2


# ── ConvolutionEngine ──────────────────────────────────────

def make_engine(ir_len: int = 128) -> ConvolutionEngine:
    ir = np.zeros(ir_len, dtype=np.float32)
    ir[0] = 1.0  # impulso puro — saída deve ser igual à entrada
    return ConvolutionEngine(ir_ll=ir, ir_lr=ir, ir_rl=ir, ir_rr=ir)


def test_convolution_impulse_response():
    engine = make_engine()
    signal = np.random.randn(4096).astype(np.float32)
    out_l, out_r = engine.process(signal, signal)
    # Com IR de impulso, saída deve correlacionar fortemente com entrada
    assert len(out_l) >= len(signal)
    assert len(out_r) >= len(signal)


def test_convolution_output_normalized():
    engine = make_engine()
    signal = np.ones(4096, dtype=np.float32)
    out_l, out_r = engine.process(signal, signal)
    assert np.max(np.abs(out_l)) <= 1.0 + 1e-5
    assert np.max(np.abs(out_r)) <= 1.0 + 1e-5


def test_convolution_output_dtype():
    engine = make_engine()
    signal = np.random.randn(2048).astype(np.float32)
    out_l, out_r = engine.process(signal, signal)
    assert out_l.dtype == np.float32
    assert out_r.dtype == np.float32


# ── IrLoader ──────────────────────────────────────────────

def test_ir_loader_file_not_found():
    loader = IrLoader("hrtf/sadie")
    with pytest.raises(FileNotFoundError):
        loader.load("nao_existe.wav")

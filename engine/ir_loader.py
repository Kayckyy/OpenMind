import numpy as np
from scipy.io import wavfile
from pathlib import Path


class IrLoader:
    def __init__(self, hrtf_dir: str):
        self.hrtf_dir = Path(hrtf_dir)

    def load(self, filename: str) -> tuple[int, np.ndarray, np.ndarray]:
        path = self.hrtf_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"IR not found: {path}")

        sr, data = wavfile.read(path)

        # Converte pra float32 normalizado
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max

        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f"Expected stereo IR, got shape {data.shape}")

        ir_l = data[:, 0]
        ir_r = data[:, 1]

        # Normaliza pelo pico absoluto
        peak = max(np.max(np.abs(ir_l)), np.max(np.abs(ir_r)))
        if peak > 0:
            ir_l = ir_l / peak
            ir_r = ir_r / peak

        return sr, ir_l, ir_r

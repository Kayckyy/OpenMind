import numpy as np
from scipy.signal import resample_poly
from math import gcd


class Resampler:
    TARGET_SR = 44100

    def resample(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr == self.TARGET_SR:
            return audio

        common = gcd(sr, self.TARGET_SR)
        up = self.TARGET_SR // common
        down = sr // common

        if audio.ndim == 1:
            return resample_poly(audio, up, down).astype(np.float32)

        # Stereo: processa cada canal
        l = resample_poly(audio[:, 0], up, down)
        r = resample_poly(audio[:, 1], up, down)
        return np.stack([l, r], axis=1).astype(np.float32)

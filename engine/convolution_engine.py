"""
True-Stereo Convolution Engine
================================
4 IRs: LL, LR, RL, RR
  out_L = conv(in_L, ir_LL) + conv(in_R, ir_RL)
  out_R = conv(in_L, ir_LR) + conv(in_R, ir_RR)
"""

import numpy as np


class ConvolutionEngine:
    def __init__(
        self,
        ir_ll: np.ndarray,
        ir_lr: np.ndarray,
        ir_rl: np.ndarray,
        ir_rr: np.ndarray,
        sample_rate: int = 44100,
    ):
        self.sample_rate = sample_rate
        self._irs = {
            'll': ir_ll.astype(np.float32),
            'lr': ir_lr.astype(np.float32),
            'rl': ir_rl.astype(np.float32),
            'rr': ir_rr.astype(np.float32),
        }
        self._ir_len = max(len(ir) for ir in self._irs.values())
        self._fft_cache: dict[int, dict[str, np.ndarray]] = {}
        self.reset()

    def reset(self):
        ov_len = self._ir_len - 1
        self._ov = {k: np.zeros(ov_len, dtype=np.float32) for k in self._irs}

    def _get_ir_ffts(self, fft_size: int) -> dict[str, np.ndarray]:
        if fft_size not in self._fft_cache:
            self._fft_cache[fft_size] = {
                k: np.fft.rfft(ir, n=fft_size)
                for k, ir in self._irs.items()
            }
        return self._fft_cache[fft_size]

    def _convolve_block(self, signal, ir_key, fft_size, ir_ffts):
        sig_len = len(signal)
        ir_len  = len(self._irs[ir_key])

        sig_fft = np.fft.rfft(signal, n=fft_size)
        conv    = np.fft.irfft(sig_fft * ir_ffts[ir_key], n=fft_size)
        conv    = conv[:sig_len + ir_len - 1].astype(np.float32)

        ov_len = len(self._ov[ir_key])
        conv[:ov_len] += self._ov[ir_key]
        self._ov[ir_key][:] = conv[sig_len:sig_len + ov_len]

        return conv[:sig_len]

    def process(self, in_l: np.ndarray, in_r: np.ndarray):
        n = len(in_l)
        fft_size = 1
        while fft_size < n + self._ir_len - 1:
            fft_size <<= 1

        ir_ffts = self._get_ir_ffts(fft_size)

        ll = self._convolve_block(in_l, 'll', fft_size, ir_ffts)
        lr = self._convolve_block(in_l, 'lr', fft_size, ir_ffts)
        rl = self._convolve_block(in_r, 'rl', fft_size, ir_ffts)
        rr = self._convolve_block(in_r, 'rr', fft_size, ir_ffts)

        out_l = ll + rl
        out_r = rr + lr

        # sem normalização por bloco — só clip hard pra proteger o DAC
        np.clip(out_l, -1.0, 1.0, out=out_l)
        np.clip(out_r, -1.0, 1.0, out=out_r)

        return out_l.astype(np.float32), out_r.astype(np.float32)
      

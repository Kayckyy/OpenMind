"""
True-Stereo Convolution Engine
================================
4 IRs: LL, LR, RL, RR
  out_L = conv(in_L, ir_LL) + conv(in_R, ir_RL)
  out_R = conv(in_L, ir_LR) + conv(in_R, ir_RR)

Otimizações vs versão anterior:
  - FFT das IRs cacheada por fft_size (não recalcula a cada bloco)
  - Sem BandRouter — pipeline direto, sem múltiplos mix stages
  - reset() limpa overlaps sem recriar arrays
"""

import numpy as np
from typing import Optional


class ConvolutionEngine:
    def __init__(
        self,
        ir_ll: np.ndarray,  # L → out_L
        ir_lr: np.ndarray,  # L → out_R  (crossfeed)
        ir_rl: np.ndarray,  # R → out_L  (crossfeed)
        ir_rr: np.ndarray,  # R → out_R
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

        # overlap buffers: tamanho fixo = ir_len - 1
        self._ov: dict[str, np.ndarray] = {}
        self.reset()

        # cache: fft_size → {key: ir_fft}
        # evita rfft(ir) a cada bloco
        self._fft_cache: dict[int, dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------------
    def reset(self):
        """Zera os buffers de overlap (use ao iniciar nova faixa)."""
        ov_len = self._ir_len - 1
        self._ov = {k: np.zeros(ov_len, dtype=np.float32) for k in self._irs}

    # ------------------------------------------------------------------
    def _get_ir_ffts(self, fft_size: int) -> dict[str, np.ndarray]:
        if fft_size not in self._fft_cache:
            self._fft_cache[fft_size] = {
                k: np.fft.rfft(ir, n=fft_size)
                for k, ir in self._irs.items()
            }
        return self._fft_cache[fft_size]

    # ------------------------------------------------------------------
    def _convolve_block(
        self,
        signal: np.ndarray,
        ir_key: str,
        fft_size: int,
        ir_ffts: dict,
    ) -> np.ndarray:
        """
        Overlap-add de um bloco.
        Retorna exatamente len(signal) amostras.
        """
        sig_len = len(signal)
        ir_len  = len(self._irs[ir_key])

        sig_fft = np.fft.rfft(signal, n=fft_size)
        conv    = np.fft.irfft(sig_fft * ir_ffts[ir_key], n=fft_size)
        conv    = conv[:sig_len + ir_len - 1].astype(np.float32)

        ov_len = len(self._ov[ir_key])
        conv[:ov_len] += self._ov[ir_key]
        # salva tail para o próximo bloco
        tail_start = sig_len
        tail_end   = tail_start + ov_len
        self._ov[ir_key][:] = conv[tail_start:tail_end]

        return conv[:sig_len]

    # ------------------------------------------------------------------
    def process(
        self,
        in_l: np.ndarray,
        in_r: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Processa um bloco stereo.
        Retorna (out_l, out_r) com o mesmo comprimento da entrada.
        """
        n = len(in_l)
        # fft_size = próxima potência de 2 >= n + ir_len - 1
        fft_size = 1
        while fft_size < n + self._ir_len - 1:
            fft_size <<= 1

        ir_ffts = self._get_ir_ffts(fft_size)

        # True-Stereo: 4 convoluções
        ll = self._convolve_block(in_l, 'll', fft_size, ir_ffts)
        lr = self._convolve_block(in_l, 'lr', fft_size, ir_ffts)
        rl = self._convolve_block(in_r, 'rl', fft_size, ir_ffts)
        rr = self._convolve_block(in_r, 'rr', fft_size, ir_ffts)

        out_l = ll + rl  # L direto + crossfeed de R
        out_r = rr + lr  # R direto + crossfeed de L

        # normalização suave para evitar clip
        peak = max(np.max(np.abs(out_l)), np.max(np.abs(out_r)), 1e-9)
        if peak > 0.99:
            out_l /= peak
            out_r /= peak

        return out_l.astype(np.float32), out_r.astype(np.float32)
      

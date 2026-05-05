import numpy as np


class ConvolutionEngine:
    def __init__(self, ir_ll, ir_lr, ir_rl, ir_rr):
        self.ir_ll = ir_ll
        self.ir_lr = ir_lr
        self.ir_rl = ir_rl
        self.ir_rr = ir_rr

        ir_len = max(len(ir_ll), len(ir_lr), len(ir_rl), len(ir_rr))
        self._overlap_l = np.zeros(ir_len - 1, dtype=np.float32)
        self._overlap_r = np.zeros(ir_len - 1, dtype=np.float32)

    def reset(self):
        ir_len = max(len(self.ir_ll), len(self.ir_lr),
                     len(self.ir_rl), len(self.ir_rr))
        self._overlap_l = np.zeros(ir_len - 1, dtype=np.float32)
        self._overlap_r = np.zeros(ir_len - 1, dtype=np.float32)

    def _overlap_add(self, signal, ir, overlap):
        ir_len = len(ir)
        fft_size = 1
        while fft_size < len(signal) + ir_len - 1:
            fft_size <<= 1

        ir_fft = np.fft.rfft(ir, n=fft_size)
        block_fft = np.fft.rfft(signal, n=fft_size)
        conv = np.fft.irfft(block_fft * ir_fft, n=fft_size)
        conv = conv[:len(signal) + ir_len - 1].astype(np.float32)

        # Aplica overlap
        ov_len = len(overlap)
        conv[:ov_len] += overlap

        # Salva novo overlap
        overlap[:] = conv[len(signal):len(signal) + ov_len]

        return conv[:len(signal)]

    def process(self, input_l, input_r):
        ll = self._overlap_add(input_l, self.ir_ll, self._overlap_l)
        lr = self._overlap_add(input_l, self.ir_lr, self._overlap_r)
        rl = self._overlap_add(input_r, self.ir_rl, self._overlap_l)
        rr = self._overlap_add(input_r, self.ir_rr, self._overlap_r)

        out_l = ll + rl
        out_r = lr + rr

        peak = max(np.max(np.abs(out_l)), np.max(np.abs(out_r)))
        if peak > 0:
            out_l = out_l / peak
            out_r = out_r / peak

        return out_l.astype(np.float32), out_r.astype(np.float32)

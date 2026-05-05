import numpy as np


class ConvolutionEngine:
    def __init__(self, ir_ll, ir_lr, ir_rl, ir_rr):
        self.ir_ll = ir_ll
        self.ir_lr = ir_lr
        self.ir_rl = ir_rl
        self.ir_rr = ir_rr
        self._init_overlaps()

    def _init_overlaps(self):
        ir_len = max(len(self.ir_ll), len(self.ir_lr),
                     len(self.ir_rl), len(self.ir_rr))
        self._ov_ll = np.zeros(ir_len - 1, dtype=np.float32)
        self._ov_lr = np.zeros(ir_len - 1, dtype=np.float32)
        self._ov_rl = np.zeros(ir_len - 1, dtype=np.float32)
        self._ov_rr = np.zeros(ir_len - 1, dtype=np.float32)

    def reset(self):
        self._init_overlaps()

    def _overlap_add(self, signal, ir, overlap):
        ir_len = len(ir)
        fft_size = 1
        while fft_size < len(signal) + ir_len - 1:
            fft_size <<= 1

        ir_fft = np.fft.rfft(ir, n=fft_size)
        block_fft = np.fft.rfft(signal, n=fft_size)
        conv = np.fft.irfft(block_fft * ir_fft, n=fft_size)
        conv = conv[:len(signal) + ir_len - 1].astype(np.float32)

        ov_len = len(overlap)
        conv[:ov_len] += overlap
        overlap[:] = conv[len(signal):len(signal) + ov_len]

        return conv[:len(signal)]
        
    def process(self, input_l, input_r):
        ll = self._overlap_add(input_l, self.ir_ll, self._ov_ll)
        lr = self._overlap_add(input_l, self.ir_lr, self._ov_lr)
        rl = self._overlap_add(input_r, self.ir_rl, self._ov_rl)
        rr = self._overlap_add(input_r, self.ir_rr, self._ov_rr)
        out_l = (ll + rl) * 0.5
        out_r = (lr + rr) * 0.5

    return out_l.astype(np.float32), out_r.astype(np.float32)
        out_l = ll + rl
        out_r = lr + rr

        # Remove normalização por chunk — deixa o nível natural
        return out_l.astype(np.float32), out_r.astype(np.float32)

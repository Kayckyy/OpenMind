import numpy as np


class ConvolutionEngine:
    def __init__(self, ir_ll: np.ndarray, ir_lr: np.ndarray,
                 ir_rl: np.ndarray, ir_rr: np.ndarray):
        """
        ir_ll: IR para canal L -> ouvido L (ipsilateral)
        ir_lr: IR para canal L -> ouvido R (contralateral)
        ir_rl: IR para canal R -> ouvido L (contralateral)
        ir_rr: IR para canal R -> ouvido R (ipsilateral)
        """
        self.ir_ll = ir_ll
        self.ir_lr = ir_lr
        self.ir_rl = ir_rl
        self.ir_rr = ir_rr

    def _overlap_add(self, signal: np.ndarray, ir: np.ndarray,
                     block_size: int = 2048) -> np.ndarray:
        ir_len = len(ir)
        fft_size = 1
        while fft_size < block_size + ir_len - 1:
            fft_size <<= 1

        ir_fft = np.fft.rfft(ir, n=fft_size)
        output = np.zeros(len(signal) + ir_len - 1, dtype=np.float32)

        for i in range(0, len(signal), block_size):
            block = signal[i:i + block_size]
            block_fft = np.fft.rfft(block, n=fft_size)
            conv = np.fft.irfft(block_fft * ir_fft, n=fft_size)
            output[i:i + len(conv)] += conv[:len(output) - i]

        return output

    def process(self, input_l: np.ndarray,
                input_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # True Stereo: 4 convoluções
        ll = self._overlap_add(input_l, self.ir_ll)
        lr = self._overlap_add(input_l, self.ir_lr)
        rl = self._overlap_add(input_r, self.ir_rl)
        rr = self._overlap_add(input_r, self.ir_rr)

        # Mix nos canais de saída
        out_l = ll + rl
        out_r = lr + rr

        # Normaliza pra evitar clipping
        peak = max(np.max(np.abs(out_l)), np.max(np.abs(out_r)))
        if peak > 0:
            out_l = out_l / peak
            out_r = out_r / peak

        return out_l.astype(np.float32), out_r.astype(np.float32)

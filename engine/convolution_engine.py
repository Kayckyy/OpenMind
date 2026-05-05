import numpy as np
from engine.upmix import Upmix


class ConvolutionEngine:
    def __init__(self, ir_front_l, ir_front_r,
                       ir_rear_l,  ir_rear_r,
                       ir_side_ll, ir_side_lr,
                       ir_side_rl, ir_side_rr):
        self.irs = {
            'front_l':  ir_front_l,
            'front_r':  ir_front_r,
            'rear_l':   ir_rear_l,
            'rear_r':   ir_rear_r,
            'side_ll':  ir_side_ll,
            'side_lr':  ir_side_lr,
            'side_rl':  ir_side_rl,
            'side_rr':  ir_side_rr,
        }
        self.upmix = Upmix()
        self._init_overlaps()

    def _init_overlaps(self):
        # Cada IR tem seu próprio tamanho de overlap — sem desperdício de memória
        self._overlaps = {
            k: np.zeros(len(ir) - 1, dtype=np.float32)
            for k, ir in self.irs.items()
        }

    def reset(self):
        self._init_overlaps()

    def _overlap_add(self, signal: np.ndarray,
                     ir: np.ndarray,
                     overlap: np.ndarray) -> np.ndarray:
        ir_len = len(ir)
        fft_size = 1
        while fft_size < len(signal) + ir_len - 1:
            fft_size <<= 1

        ir_fft    = np.fft.rfft(ir, n=fft_size)
        block_fft = np.fft.rfft(signal, n=fft_size)
        conv = np.fft.irfft(block_fft * ir_fft, n=fft_size)
        conv = conv[:len(signal) + ir_len - 1].astype(np.float32)

        ov_len = len(overlap)
        conv[:ov_len] += overlap
        overlap[:] = conv[len(signal):len(signal) + ov_len]

        return conv[:len(signal)]

    def process(self, input_l: np.ndarray,
                      input_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        channels = self.upmix.process(input_l, input_r)

        # Front agora tem L/R separados vindos do upmix
        front_l = self._overlap_add(channels['front_l'], self.irs['front_l'], self._overlaps['front_l'])
        front_r = self._overlap_add(channels['front_r'], self.irs['front_r'], self._overlaps['front_r'])

        rear_l  = self._overlap_add(channels['rear'],    self.irs['rear_l'],  self._overlaps['rear_l'])
        rear_r  = self._overlap_add(channels['rear'],    self.irs['rear_r'],  self._overlaps['rear_r'])

        side_ll = self._overlap_add(channels['side_l'],  self.irs['side_ll'], self._overlaps['side_ll'])
        side_lr = self._overlap_add(channels['side_l'],  self.irs['side_lr'], self._overlaps['side_lr'])
        side_rl = self._overlap_add(channels['side_r'],  self.irs['side_rl'], self._overlaps['side_rl'])
        side_rr = self._overlap_add(channels['side_r'],  self.irs['side_rr'], self._overlaps['side_rr'])

        # Mix binaural corrigido:
        # - Canal direto (ipsilateral): ganho maior
        # - Crossfeed (contralateral): ganho menor — preserva largura estéreo
        # - Rear com ganho reduzido para não vazar Mid no traseiro
        out_l = (
            front_l * 0.60 +
            rear_l  * 0.18 +
            side_ll * 1.00 +   # side esquerdo → ouvido esquerdo (direto)
            side_rl * 0.35     # side direito  → ouvido esquerdo (crossfeed reduzido)
        ) * 0.35

        out_r = (
            front_r * 0.60 +
            rear_r  * 0.18 +
            side_rr * 1.00 +   # side direito  → ouvido direito (direto)
            side_lr * 0.35     # side esquerdo → ouvido direito (crossfeed reduzido)
        ) * 0.35

        return out_l.astype(np.float32), out_r.astype(np.float32)
                          

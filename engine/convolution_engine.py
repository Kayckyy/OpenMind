import numpy as np
from engine.upmix import Upmix


class ConvolutionEngine:
    """
    Mix binaural aditivo:
    - direct_l/r passa pela IR frontal e vai direto pro output com ganho alto
    - sides e rear são convoluídos e somados em cima com ganho menor
    - resultado: 360° sem colapsar a imagem estéreo original
    """

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

        # Front: estéreo direto convoluído com IR frontal (âncora da imagem)
        front_l = self._overlap_add(channels['direct_l'], self.irs['front_l'], self._overlaps['front_l'])
        front_r = self._overlap_add(channels['direct_r'], self.irs['front_r'], self._overlaps['front_r'])

        # Rear: conteúdo difuso convoluído com IR traseira
        rear_l  = self._overlap_add(channels['rear'], self.irs['rear_l'], self._overlaps['rear_l'])
        rear_r  = self._overlap_add(channels['rear'], self.irs['rear_r'], self._overlaps['rear_r'])

        # Sides: conteúdo lateral decorrelado convoluído com IR lateral
        side_ll = self._overlap_add(channels['side_l'], self.irs['side_ll'], self._overlaps['side_ll'])
        side_lr = self._overlap_add(channels['side_l'], self.irs['side_lr'], self._overlaps['side_lr'])
        side_rl = self._overlap_add(channels['side_r'], self.irs['side_rl'], self._overlaps['side_rl'])
        side_rr = self._overlap_add(channels['side_r'], self.irs['side_rr'], self._overlaps['side_rr'])

        # Mix aditivo:
        # Front domina (0.75) — preserva centro e imagem original
        # Sides adicionam largura (0.30 direto, 0.10 crossfeed)
        # Rear adiciona profundidade (0.15) sem puxar a imagem pra trás
        out_l = (
            front_l * 0.75 +
            side_ll * 0.30 +   # lado esquerdo direto
            side_rl * 0.10 +   # crossfeed direito→esquerdo (leve)
            rear_l  * 0.15
        )

        out_r = (
            front_r * 0.75 +
            side_rr * 0.30 +   # lado direito direto
            side_lr * 0.10 +   # crossfeed esquerdo→direito (leve)
            rear_r  * 0.15
        )

        # Normalização suave para evitar clipping sem esmagar dinâmica
        peak = max(np.max(np.abs(out_l)), np.max(np.abs(out_r)), 1e-10)
        if peak > 0.95:
            out_l = out_l * (0.95 / peak)
            out_r = out_r * (0.95 / peak)

        return out_l.astype(np.float32), out_r.astype(np.float32)
                          

import numpy as np
from engine.band_router import BandRouter


class ConvolutionEngine:
    """
    Pipeline:
      input L/R
        → BandRouter (divide por banda, roteia por posição HRTF)
        → convolução por canal virtual
        → mix binaural aditivo (direct domina, virtuais somam em cima)
    """

    def __init__(self, ir_front_l, ir_front_r,
                       ir_rear_l,  ir_rear_r,
                       ir_side_ll, ir_side_lr,
                       ir_side_rl, ir_side_rr,
                       sample_rate: int = 44100):
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
        self.router = BandRouter(sample_rate)
        self._init_overlaps()
        self._ir_ffts = self._precompute_ir_ffts()

    def _init_overlaps(self):
        self._overlaps = {
            k: np.zeros(len(ir) - 1, dtype=np.float32)
            for k, ir in self.irs.items()
        }

    def _precompute_ir_ffts(self) -> dict:
        """
        Pré-computa FFT de todas as IRs — evita recalcular a cada bloco.
        Cada entrada guarda (ir_fft, fft_size, ir_len).
        """
        cache = {}
        for k, ir in self.irs.items():
            ir_len = len(ir)
            # fft_size será ajustado por bloco, mas guardamos a IR no domínio do tempo
            # O rfft da IR é feito no _overlap_add com o fft_size correto
            cache[k] = ir  # mantém referência; rfft cacheado por tamanho abaixo
        return cache

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

        ch = self.router.process(input_l, input_r)

        # Direto (âncora) — convoluído com IR frontal
        direct_l = self._overlap_add(ch['direct_l'], self.irs['front_l'], self._overlaps['front_l'])
        direct_r = self._overlap_add(ch['direct_r'], self.irs['front_r'], self._overlaps['front_r'])

        # Front virtual (bandas sub + médio-baixo + ar leve)
        front_l  = self._overlap_add(ch['front_l'],  self.irs['front_l'], self._overlaps['front_l'])
        front_r  = self._overlap_add(ch['front_r'],  self.irs['front_r'], self._overlaps['front_r'])

        # Rear (médio-alto + presença com coloração traseira)
        rear_l   = self._overlap_add(ch['rear_l'],   self.irs['rear_l'],  self._overlaps['rear_l'])
        rear_r   = self._overlap_add(ch['rear_r'],   self.irs['rear_r'],  self._overlaps['rear_r'])

        # Sides (médio-alto + presença + ar)
        side_ll  = self._overlap_add(ch['side_l'],   self.irs['side_ll'], self._overlaps['side_ll'])
        side_lr  = self._overlap_add(ch['side_l'],   self.irs['side_lr'], self._overlaps['side_lr'])
        side_rl  = self._overlap_add(ch['side_r'],   self.irs['side_rl'], self._overlaps['side_rl'])
        side_rr  = self._overlap_add(ch['side_r'],   self.irs['side_rr'], self._overlaps['side_rr'])

        # Mix binaural aditivo:
        # direct domina → imagem original preservada
        # virtuais somam em cima → espacialização adicional
        out_l = (
            direct_l * 0.80 +   # âncora L
            front_l  * 0.20 +   # coloração frontal virtual
            side_ll  * 0.25 +   # lado esquerdo direto
            side_rl  * 0.08 +   # crossfeed direito→esquerdo
            rear_l   * 0.12     # traseiro esquerdo
        )

        out_r = (
            direct_r * 0.80 +   # âncora R
            front_r  * 0.20 +   # coloração frontal virtual
            side_rr  * 0.25 +   # lado direito direto
            side_lr  * 0.08 +   # crossfeed esquerdo→direito
            rear_r   * 0.12     # traseiro direito
        )

        return out_l.astype(np.float32), out_r.astype(np.float32)
                          

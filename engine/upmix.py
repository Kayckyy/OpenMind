import numpy as np


class Upmix:
    """
    Filosofia: o estéreo original é a âncora da imagem.
    Os canais espaciais (side, rear) são extraídos e retornados separadamente
    para serem misturados de forma aditiva pelo ConvolutionEngine.

    Não há ganho aplicado aqui sobre o sinal original — só extração.
    """

    def __init__(self, block_size: int = 4096, smooth: float = 0.95):
        self.block_size = block_size
        self.smooth = smooth

        self._rear_gain = 0.0
        self._side_gain = 0.5

        # Buffer de delay para decorrelação do side_r (7 samples)
        self._delay_buf = np.zeros(7, dtype=np.float32)

    def _correlation(self, l: np.ndarray, r: np.ndarray) -> float:
        norm = np.sqrt(np.sum(l**2) * np.sum(r**2))
        if norm < 1e-10:
            return 0.0
        return float(np.sum(l * r) / norm)

    def _decorrelate(self, signal: np.ndarray) -> np.ndarray:
        """Delay de 7 samples com buffer entre blocos — sem inversão de fase."""
        delay = len(self._delay_buf)
        padded = np.concatenate([self._delay_buf, signal])
        out = padded[:len(signal)].copy()
        self._delay_buf[:] = signal[-delay:]
        return out.astype(np.float32)

    def process(self, input_l: np.ndarray,
                input_r: np.ndarray) -> dict[str, np.ndarray]:

        mid    = (input_l + input_r) * 0.5
        side_l = (input_l - input_r) * 0.5
        side_r = self._decorrelate((input_r - input_l) * 0.5)

        corr = self._correlation(input_l, input_r)

        # Quanto mais descorrelacionado, mais espaço nos sides e rear
        diffuse = 1.0 - abs(corr)

        target_rear = diffuse * 0.5
        target_side = min(diffuse * 1.0, 0.85)  # nunca passa de 0.85

        self._rear_gain = self.smooth * self._rear_gain + (1 - self.smooth) * target_rear
        self._side_gain = self.smooth * self._side_gain + (1 - self.smooth) * target_side

        return {
            # Estéreo original intacto — âncora da imagem frontal
            'direct_l': input_l,
            'direct_r': input_r,

            # Conteúdo difuso para os sides (decorrelado)
            'side_l':   side_l * self._side_gain,
            'side_r':   side_r * self._side_gain,

            # Rear: componente difusa, não o Mid
            'rear':     (side_l + side_r) * self._rear_gain,
        }
                    

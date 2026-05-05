import numpy as np


class Upmix:
    def __init__(self, block_size: int = 4096, smooth: float = 0.95):
        self.block_size = block_size
        self.smooth = smooth

        self._front_gain = 0.5
        self._rear_gain  = 0.0
        self._side_gain  = 0.5

        # Buffer de delay para decorrelação do side_r (7 samples)
        self._delay_buf = np.zeros(7, dtype=np.float32)

    def _correlation(self, l: np.ndarray, r: np.ndarray) -> float:
        norm = np.sqrt(np.sum(l**2) * np.sum(r**2))
        if norm < 1e-10:
            return 0.0
        return float(np.sum(l * r) / norm)

    def _decorrelate(self, signal: np.ndarray) -> np.ndarray:
        """
        Decorrelação por delay fracionado curto (7 samples ≈ 0.15ms @ 44100Hz).
        Evita inversão de fase sem introduzir coloração perceptível.
        Mantém buffer entre blocos para continuidade.
        """
        delay = len(self._delay_buf)
        padded = np.concatenate([self._delay_buf, signal])
        out = padded[:len(signal)].copy()
        self._delay_buf[:] = signal[-delay:]
        return out.astype(np.float32)

    def process(self, input_l: np.ndarray,
                input_r: np.ndarray) -> dict[str, np.ndarray]:

        mid    = (input_l + input_r) * 0.5
        side_l = (input_l - input_r) * 0.5
        side_r = self._decorrelate((input_r - input_l) * 0.5)  # decorrelado, não invertido

        corr = self._correlation(input_l, input_r)

        target_front = max(0.0, corr)
        target_rear  = (1.0 - abs(corr)) * 0.4
        target_side  = min((1.0 - abs(corr)) * 1.2, 1.0)  # clamp para evitar ganho > 1

        self._front_gain = self.smooth * self._front_gain + (1 - self.smooth) * target_front
        self._rear_gain  = self.smooth * self._rear_gain  + (1 - self.smooth) * target_rear
        self._side_gain  = self.smooth * self._side_gain  + (1 - self.smooth) * target_side

        fg = self._front_gain
        sg = self._side_gain

        return {
            # Front preserva parte da diferença L/R original — evita imagem frontal flat
            'front_l': (mid + side_l * 0.3) * fg,
            'front_r': (mid + side_r * 0.3) * fg,

            # Rear carrega conteúdo difuso (side somado), não o Mid
            'rear':    (side_l + side_r) * self._rear_gain,

            'side_l':  side_l * sg,
            'side_r':  side_r * sg,
        }
                    

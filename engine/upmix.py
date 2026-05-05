import numpy as np


class Upmix:
    def __init__(self, block_size: int = 4096, smooth: float = 0.95):
        self.block_size = block_size
        self.smooth = smooth

        # Ganhos suavizados — persistem entre blocos
        self._front_gain = 0.5
        self._rear_gain  = 0.0
        self._side_gain  = 0.5

    def _correlation(self, l: np.ndarray, r: np.ndarray) -> float:
        norm = np.sqrt(np.sum(l**2) * np.sum(r**2))
        if norm < 1e-10:
            return 0.0
        return float(np.sum(l * r) / norm)

    def process(self, input_l: np.ndarray,
                input_r: np.ndarray) -> dict[str, np.ndarray]:
        mid    = (input_l + input_r) * 0.5
        side_l = (input_l - input_r) * 0.5
        side_r = (input_r - input_l) * 0.5

        corr = self._correlation(input_l, input_r)

        # Ganhos alvo
        target_front = max(0.0, corr)
        target_rear = (1.0 - abs(corr)) * 0.4
        target_side  = (1.0 - abs(corr)) * 1.2

        # Suavização exponencial
        self._front_gain = self.smooth * self._front_gain + (1 - self.smooth) * target_front
        self._rear_gain  = self.smooth * self._rear_gain  + (1 - self.smooth) * target_rear
        self._side_gain  = self.smooth * self._side_gain  + (1 - self.smooth) * target_side

        return {
            'front':  mid    * self._front_gain,
            'rear':   mid    * self._rear_gain,
            'side_l': side_l * self._side_gain,
            'side_r': side_r * self._side_gain,
        }

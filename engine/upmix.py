import numpy as np


class Upmix:
    def __init__(self, block_size: int = 4096):
        self.block_size = block_size

    def _correlation(self, l: np.ndarray, r: np.ndarray) -> float:
        """Correlação normalizada entre L e R (-1 a 1)"""
        norm = np.sqrt(np.sum(l**2) * np.sum(r**2))
        if norm < 1e-10:
            return 0.0
        return float(np.sum(l * r) / norm)

    def process(self, input_l: np.ndarray,
                input_r: np.ndarray) -> dict[str, np.ndarray]:
        mid  = (input_l + input_r) * 0.5
        side_l = (input_l - input_r) * 0.5
        side_r = (input_r - input_l) * 0.5

        corr = self._correlation(input_l, input_r)

        # Ganhos adaptativos baseados na correlação
        # corr ~  1.0 → fonte central (frente forte, trás fraco)
        # corr ~  0.0 → fontes laterais (lados fortes)
        # corr ~ -1.0 → fonte traseira (trás forte)

        front_gain = max(0.0, corr)
        rear_gain  = max(0.0, -corr) * 0.7
        side_gain  = (1.0 - abs(corr)) * 1.2

        return {
            'front':   mid    * front_gain,
            'rear':    mid    * rear_gain,
            'side_l':  side_l * side_gain,
            'side_r':  side_r * side_gain,
        }

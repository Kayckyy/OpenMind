import numpy as np


class BiquadFilter:
    """
    Filtro biquad IIR de 2 polos — Direct Form I.
    Mantém estado entre blocos para continuidade.
    """

    def __init__(self, b0, b1, b2, a1, a2):
        self.b0, self.b1, self.b2 = b0, b1, b2
        self.a1, self.a2 = a1, a2
        self._x1 = self._x2 = 0.0
        self._y1 = self._y2 = 0.0

    def process(self, signal: np.ndarray) -> np.ndarray:
        out = np.empty_like(signal, dtype=np.float32)
        x1, x2, y1, y2 = self._x1, self._x2, self._y1, self._y2
        b0, b1, b2 = self.b0, self.b1, self.b2
        a1, a2 = self.a1, self.a2

        for i, x0 in enumerate(signal):
            y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            out[i] = y0
            x2, x1 = x1, x0
            y2, y1 = y1, y0

        self._x1, self._x2 = x1, x2
        self._y1, self._y2 = y1, y2
        return out


def _lpf(fc: float, sr: float) -> BiquadFilter:
    """Low-pass Butterworth 2ª ordem."""
    w0 = 2 * np.pi * fc / sr
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / np.sqrt(2)  # Q = 0.707

    b0 = (1 - cos_w0) / 2
    b1 =  1 - cos_w0
    b2 = (1 - cos_w0) / 2
    a0 =  1 + alpha
    a1 = -2 * cos_w0
    a2 =  1 - alpha

    return BiquadFilter(b0/a0, b1/a0, b2/a0, a1/a0, a2/a0)


def _hpf(fc: float, sr: float) -> BiquadFilter:
    """High-pass Butterworth 2ª ordem."""
    w0 = 2 * np.pi * fc / sr
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / np.sqrt(2)

    b0 =  (1 + cos_w0) / 2
    b1 = -(1 + cos_w0)
    b2 =  (1 + cos_w0) / 2
    a0 =   1 + alpha
    a1 =  -2 * cos_w0
    a2 =   1 - alpha

    return BiquadFilter(b0/a0, b1/a0, b2/a0, a1/a0, a2/a0)


def _bpf(fc_low: float, fc_high: float, sr: float):
    """Band-pass como HPF + LPF em cascata."""
    class Cascade:
        def __init__(self):
            self.hp = _hpf(fc_low, sr)
            self.lp = _lpf(fc_high, sr)
        def process(self, s):
            return self.lp.process(self.hp.process(s))
    return Cascade()


class BandRouter:
    """
    Divide o sinal M/S em bandas baseadas em HRTF e roteia para os canais virtuais.

    Bandas e destinos (por canal — L e R processados separadamente):
      Sub/baixo   20–300Hz   → front only         (graves não-direcionais)
      Médio-baixo 300–2kHz   → front (100%) + side (20%)
      Médio-alto  2k–8kHz    → side (100%) + rear atenuado (60%)
      Presença    8k–12kHz   → side (100%) + rear boost (80%)
      Ar          12k–24kHz  → side (100%), front leve (20%)

    O sinal direto (L/R original) continua como âncora — não é tocado aqui.
    """

    def __init__(self, sample_rate: int = 44100):
        sr = sample_rate

        # Banco de filtros para canal L
        self._bands_l = self._make_bands(sr)
        # Banco de filtros para canal R
        self._bands_r = self._make_bands(sr)

    def _make_bands(self, sr):
        return {
            'sub':       _lpf(300, sr),
            'mid_low':   _bpf(300, 2000, sr),
            'mid_high':  _bpf(2000, 8000, sr),
            'presence':  _bpf(8000, 12000, sr),
            'air':       _hpf(12000, sr),
        }

    def process(self, signal_l: np.ndarray,
                signal_r: np.ndarray) -> dict[str, np.ndarray]:
        """
        Retorna canais virtuais prontos para convolução.
        'direct_l' e 'direct_r' são passados intactos.
        """

        def route(bands: dict, sig: np.ndarray) -> dict[str, np.ndarray]:
            sub      = bands['sub'].process(sig)
            mid_low  = bands['mid_low'].process(sig)
            mid_high = bands['mid_high'].process(sig)
            presence = bands['presence'].process(sig)
            air      = bands['air'].process(sig)

            # Roteamento por canal virtual — baseado em HRTF
            front = (
                sub      * 1.00 +   # graves: 100% frente
                mid_low  * 1.00 +   # médio-baixo: 100% frente
                air      * 0.20     # ar: leve presença frontal
            )
            rear = (
                mid_high * 0.60 +   # médio-alto: reflexões de sala
                presence * 0.80     # presença: pinna shadow traseiro
            )
            side = (
                mid_low  * 0.20 +   # médio-baixo: leve contribuição lateral
                mid_high * 1.00 +   # médio-alto: localização lateral principal
                presence * 1.00 +   # presença: ITD lateral
                air      * 1.00     # ar: dispersão lateral total
            )

            return front, rear, side

        front_l, rear_l, side_l = route(self._bands_l, signal_l)
        front_r, rear_r, side_r = route(self._bands_r, signal_r)

        return {
            'direct_l': signal_l,   # âncora — intacto
            'direct_r': signal_r,
            'front_l':  front_l,
            'front_r':  front_r,
            'rear_l':   rear_l,
            'rear_r':   rear_r,
            'side_l':   side_l,
            'side_r':   side_r,
    }
                    

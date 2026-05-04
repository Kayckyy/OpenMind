# OpenMind

> Open spatial audio processing for everyone, everywhere.

OpenMind is an open-source spatial audio engine that brings high-quality binaural 3D audio to any platform, any device, and any audience — without walled gardens or proprietary hardware.

## Why OpenMind?

Most spatial audio solutions today are locked behind ecosystems:
- Dolby Atmos requires certified hardware or paid subscriptions
- Sony 360 Reality Audio is tied to their platform
- Apple Spatial Audio only works with AirPods and iPhone

OpenMind changes that. Powered by a Python audio engine and a lightweight web frontend, it runs on anything — from flagship phones to modest tablets.

## Features

- 🎧 True Stereo binaural spatialization (LL, LR, RL, RR convolution)
- 🧠 HRTF-based processing using SADIE II measurements
- ⚡ FFT overlap-add convolution engine (no WebAudio API dependency)
- 🌐 Web frontend + Python API — runs entirely on local hardware
- 🔁 Real-time streaming and file export modes
- 📐 Auto-resampling to 44100 Hz

## Architecture

```
[Web Frontend]  ←→  [FastAPI]  ←→  [Audio Engine]
  HTML / JS          Python         NumPy / SciPy
```

## Getting Started

```bash
git clone https://github.com/Kayckyy/OpenMind
cd OpenMind
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Then open the frontend in your browser.

## Roadmap

- [ ] Core convolution engine (True Stereo)
- [ ] FastAPI backend
- [ ] Web frontend
- [ ] Real-time streaming
- [ ] HRTF personalization via ear photo analysis (AI)
- [ ] Multi-platform SDK

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.

---

*Built from scratch on a phone. Designed for the world.*

# KinoTTS Studio

Offline Android text-to-speech using Supertonic 3.

## Current status

- Supertonic 3 runs locally on Android ARM64.
- Polish speech generation is supported.
- Audio import and local quality analysis are available.
- Reference recordings can be stored locally.
- ONNX Runtime Android 1.32.0 Java/JNI runtime check and XTTS four-session loader are added.
- XTTS-v2 end-to-end voice cloning is NOT implemented yet; BPE, audio features, GPT KV-cache and HiFi-GAN orchestration remain.
- Current inference backend: CPU. NPU acceleration is not integrated.

## Privacy

Do not commit private recordings, downloaded models, signing keys,
credentials, or personal data to this public repository.

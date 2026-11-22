# KinoTTS Studio

Offline Android text-to-speech using Supertonic 3.

## Current status

- Supertonic 3 runs locally on Android ARM64.
- Audio import and local quality analysis are available.
- Reference recordings can be stored locally.
- ONNX Runtime Android 1.28.2 Java/JNI runtime check and XTTS four-session loader are added.
- XTTS-v2 end-to-end voice cloning is implemented and it is being optimized right now.
- Current inference backend: CPU. NPU acceleration is in testing and doesn't always work.

## Privacy

Do not commit private recordings, downloaded models, signing keys,
credentials, or personal data to this public repository.

## Information

This project is created with help from ChatGPT and Claude.

## Requirements

- Android 11+ (API 30), ARM64 phone, a few GB of free storage for models.
- "All files access" permission: models, voice profiles and output live in
  `Documents/KinoTTS/` so they survive uninstalling the app.

## Models

- Supertonic 3 is downloaded in the app (no account needed).
- XTTS-v2 ONNX needs a free Hugging Face token (Read) and accepting the model terms.
  The token is stored encrypted with the Android Keystore.
- XTTS-v2 weights use the Coqui Public Model License (non-commercial). See THIRD_PARTY_NOTICES.md.

## Build

Local build: JDK 17, Android SDK 35, NDK 29, Rust + `cargo-ndk`, then `gradle :app:assembleDebug`.

## Responsible use

Clone only voices you own or have explicit permission to use. Do not use cloned voices to deceive or impersonate people.

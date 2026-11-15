# Third-party software in this archive

| Component | Where | License | Copyright |
|---|---|---|---|
| llama.cpp and ggml (with changes made for this engine) | `lib/libllama.so` | MIT, `llama.cpp-LICENSE` | (c) 2023-2026 The ggml authors |
| stb_image and miniaudio (through llama.cpp's image library) | inside `lib/libllama.so` | MIT or public domain, at the user's choice | (c) Sean Barrett; (c) David Reid |
| cpp-httplib 0.53.1 | inside `bin/` | MIT | (c) 2026 Yuji Hirose |
| nlohmann/json 3.12.0 | inside `bin/` | MIT | (c) 2013-2025 Niels Lohmann |

The MIT license text in `llama.cpp-LICENSE` applies to cpp-httplib and nlohmann/json with their own copyright
lines above.

Not part of this archive, downloaded by `sixi-speech setup` under their own licenses: the sherpa-onnx runtime
(Apache-2.0), NVIDIA Parakeet TDT 0.6b v3 (CC-BY-4.0), Whisper turbo (MIT), and the Piper voices (each voice's
`MODEL_CARD` names its license).

Not part of this archive: the NVIDIA CUDA runtime (from JetPack on the machine), OpenSSL (from the system), and the
model files, which `./sixi pull` downloads from their authors' Hugging Face repositories under Apache-2.0.

# DGX AI Manager

[ภาษาไทย](README.md) · **English**

A web-based LLM model & engine manager for the DGX Spark (GB10) — one page for everything: download models, manage your library, load them into memory, and chat right away.

## Features

### 🎛️ System dashboard
- Six live gauges across the top (CPU · RAM · GPU, etc.), styled to match DGX Spark Monitor

### 🧠 Running models
- See every live instance on every port, with a stop button right on the page
- Detects speculative decoding (FastMTP) and recovers automatically when the draft model fails to load

### 💬 Chat playground
- Talk to a running model from the same page, with real-time streaming responses
- Collapsible "thinking" box · a switch to disable thinking mode for faster replies

### 📚 Model library
- System catalog and user-added models together in one place
- Shows file size and ready status · download or load into memory from a single card

### ➕ Add models from Hugging Face
- Paste any HF link — the app searches and resolves the repo for you
- Pick a quant with its real file size shown before you commit to downloading
- Reads the model architecture from the GGUF header via HTTP Range requests — know whether your engine can run it without downloading the whole file
- Checks engine compatibility against the actually installed engine binary, not a hand-maintained table

### ⬇ Download manager
- Download queue backed by aria2 — per-file progress, speed, and ETA
- Pause / resume / cancel mid-download
- Separate **In progress / Completed** tabs — finished jobs move over automatically, and the library card becomes ready to load into memory right away

### 🔌 Built-in API guide
- A modal with ready-to-copy API examples for connecting external clients

## Supported engines

- **llama.cpp** (llama-server) — GGUF, every quant
- **vLLM** (docker) — NVFP4 with tool calling

## Under the hood

- Backend: Python 3.12 · FastAPI + uvicorn
- Frontend: a single HTML/JS file, no build step
- 253 tests that run fully offline (fixtures captured from real responses)

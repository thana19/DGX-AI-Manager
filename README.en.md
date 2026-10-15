# DGX AI Manager

[ภาษาไทย](README.md) · **English**

A web-based LLM model & engine manager for the DGX Spark (GB10) — one page for everything: download models, manage your library, load them into memory, and chat right away.

## Features

### 🎛️ System dashboard
- Six live gauges across the top (CPU · RAM · GPU, etc.), styled to match DGX Spark Monitor
- Three-level gauge colours: blue ≤60% · yellow >60–80% · red >80% of max

### 🧠 Running models
- See every live instance on every port, with a stop button right on the page
- Detects speculative decoding (FastMTP) and recovers automatically when the draft model fails to load
- Pick the port (8001–8009, shown as in use / free) and context size (8K/32K/64K/128K/max presets, warns above the model's ceiling) when loading a model

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
- 🔍 Re-inspect button on cards with a missing arch/context — reads the local file (or HF if not downloaded yet) and fills it in without touching the catalog

### ⬇ Download manager
- Download queue backed by aria2 — per-file progress, speed, and ETA
- Pause / resume / cancel mid-download
- Separate **In progress / Completed** tabs — finished jobs move over automatically, and the library card becomes ready to load into memory right away
- Clear failed downloads (error/cancelled) button to remove them from the list without deleting files on disk

### 🔌 Built-in API guide
- A modal with ready-to-copy API examples for connecting external clients

## Supported engines

- **llama.cpp** (llama-server) — GGUF, every quant
- **vLLM** (docker) — NVFP4 with tool calling

## Under the hood

- Backend: Python 3.12 · FastAPI + uvicorn
- Frontend: a single HTML/JS file, no build step
- 253 tests that run fully offline (fixtures captured from real responses)

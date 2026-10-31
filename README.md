# Localhost.LM

A local, browser-based chat app for running language models through Ollama. It supports chat history, image and text attachments, and optional tools. Your prompts are sent to the Ollama server running on your computer.

## Hardware requirements

- A 64-bit Windows computer with a modern processor.
- 8 GB RAM | 16 GB or more is recommended
- Ryzen 5000 series or Intel 10th gen
- GPU is optional, CUDA Capable GPU is recommended
- 10gb of Free Storage || 20GB is recommended

Actual memory and storage needs depend on the model, quantization, and context size. Check the model's requirements before downloading it.

## Software requirements

- Windows 10 version 22H2 or newer.
- Python 3.14 or newer. `start.bat` creates a virtual environment and installs the app's Python dependencies on first run.
- [Ollama for Windows](https://ollama.com/download/windows).
- A modern web browser and at least one model downloaded in Ollama.
- An internet connection for the initial Python dependency and model downloads.

## Install

1. Install [Ollama for Windows](https://ollama.com/download/windows). To verify the installation, confirm the `ollama` command works in PowerShell:

   ```powershell
   ollama --version
   ```

   The launcher can also start Ollama's server automatically when it is not already running. It looks for `ollama.exe` on `PATH` and in the standard Windows installation folders.

2. Download a model. For example, a text-and-vision model:

   ```powershell
   ollama pull llama3.2-vision
   ```

   You can also use another model already installed in Ollama. Select an installed model from the model menu in the app.

3. Start the app by double-clicking `start.bat`, or run it from PowerShell:

   ```cmd
   start.bat
   ```

   On its first run, the launcher creates a `.venv` virtual environment and installs the dependencies from `requirements.txt`. Later runs reuse that environment. Python 3.10 or newer must be installed; the launcher prefers the Windows `py` launcher and otherwise uses `python` from `PATH`.

   If Ollama is not already responding on `http://127.0.0.1:11434`, the launcher starts `ollama.exe serve` and waits up to 30 seconds for it to become available. If startup fails, check that Ollama is installed and that the executable is on `PATH` or in its standard Windows install folder.

## Run

The launcher runs the app using the project virtual environment:

```cmd
start.bat
```

The app starts a local server and opens it in your browser. It uses port `8000` when available and tries nearby ports if it is occupied. To stop the app, press `Ctrl+C` in the launcher window.

For development, activate the environment and run Uvicorn instead:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Then open <http://127.0.0.1:8000>. Keep Ollama running while using the app. Open the page through this local server, not by opening `index.html` as a file.

## Reduce Ollama KV-cache memory

Ollama can quantize its key/value (KV) cache to reduce memory use, especially with larger context windows. This is separate from quantizing the model weights: `q4_0` here changes the KV cache format, not the model you downloaded. Ollama documents `q4_0` as using about one quarter of the memory of an `f16` KV cache, with a possible small-to-medium quality tradeoff. Flash Attention must be enabled for KV-cache quantization.

### Set variables on Windows

These variables must be available to the **Ollama server process**. Setting them only in this app's Python process does not configure an already-running Ollama service.

1. In Windows Settings (or Control Panel), search for **environment variables** and open **Edit environment variables for your account**.
2. Add or edit these user variables:

   | Variable | Value | Purpose |
   | --- | --- | --- |
   | `OLLAMA_FLASH_ATTENTION` | `1` | Enables Flash Attention |
   | `OLLAMA_KV_CACHE_TYPE` | `q4_0` | Uses a 4-bit KV cache |

3. Save the changes.
4. Quit Ollama from its notification-area/taskbar icon. Start Ollama again from the Start menu so it inherits the new variables.
5. Restart `python app.py` if it is already running.

To use less aggressive quantization, set `OLLAMA_KV_CACHE_TYPE` to `q8_0`. Use `f16` for the default full-precision cache. The setting is global to the Ollama server, so it affects all models served by that Ollama instance.

## App configuration

These optional variables configure the Python app. Set them in the terminal before starting `python app.py`:

```powershell
$env:MODEL_ID = "llama3.2-vision"
$env:MAX_CONTEXT_TOKENS = "32768"
python app.py
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_ID` | `llama3.1:latest` | Backend fallback model ID |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible API URL |
| `OLLAMA_RAW_URL` | `http://localhost:11434` | Ollama native API URL |
| `OLLAMA_API_KEY` | `ollama` | API key passed to the OpenAI-compatible client |
| `MAX_CONTEXT_TOKENS` | `32768` | App context-window budget |
| `MAX_OUTPUT_TOKENS` | `4096` | Output-token reserve |
| `MAX_CONCURRENT_REQUESTS` | `2` | Concurrent generations allowed by the app |
| `SYSTEM_MESSAGE` | empty | Additional default system message |

The model menu is populated from models available in Ollama; `MODEL_ID` is a backend fallback and does not download a model.

## Data and troubleshooting

- Chat data is stored in the browser's local storage. Server-side conversation history is stored in the `histories/` folder.
- If the model list is empty or requests fail, make sure Ollama is running and that the model has finished downloading (`ollama list`).
- If a configured port is already in use with the Uvicorn command, choose another port, such as `--port 8001`, and open the matching local URL.
- On Windows, Ollama logs are available under `%LOCALAPPDATA%\Ollama`.

## References

- [Ollama Windows documentation](https://docs.ollama.com/windows)
- [Ollama FAQ: server environment variables and KV-cache quantization](https://docs.ollama.com/faq)

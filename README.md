# [![CI](https://github.com/TexasCipher/raspberry-pi-projects/actions/workflows/ci.yml/badge.svg)](https://github.com/TexasCipher/raspberry-pi-projects/actions/workflows/ci.yml)

# brainai — quick start

Prerequisites: Python 3.8+ and a virtual environment.

Recommended venv location
-------------------------

For predictable behaviour, create a per-project virtual environment at `.venv` inside the project root:

```bash
cd ~/linux/projects/brainai
python3 -m venv .venv
source .venv/bin/activate
```

This ensures tools and paths used by scripts (and CI examples in this repo) refer to the same environment.


1) Create and activate a venv

```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

Note: `torch` often requires a platform/CUDA-specific wheel. If you need CPU-only wheels, run:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

3) Run the generator

```bash
python generate.py
```

If you see errors about missing CUDA or incompatible wheels, install the appropriate `torch` wheel for your system following the instructions at https://pytorch.org/get-started/locally/.

Usage examples
---------------

- Run interactively (will prompt):

```bash
python generate.py
```

- Provide a prompt on the command line:

```bash
python generate.py --prompt "Once upon a time"
```

- Pipe a prompt from stdin:

```bash
echo "A short story:" | python generate.py
```

- Read prompt from a file:

```bash
python generate.py --file prompts/story.txt
```

- Pick a different model or run on GPU (if available):

```bash
python generate.py --prompt "Hello" --model gpt2 --device cuda
```

- Quick test without installing heavy libraries:

```bash
echo "Test" | python generate.py --dry-run
```

- Return multiple sequences:

```bash
echo "Idea" | python generate.py --dry-run -n 3
```

Generation tuning flags
-----------------------

You can tune generation parameters when running with an actual model. Examples:

- Control maximum generated tokens and temperature:

```bash
echo "Start" | python generate.py -m gpt2 -d cpu --max-new-tokens 80 --temperature 0.7
```

- Use top-k / top-p sampling and disable sampling for greedy decoding:

```bash
echo "Story" | python generate.py -m gpt2 -d cpu --top-k 50 --top-p 0.9
echo "Deterministic" | python generate.py -m gpt2 -d cpu --no-sample
```

- Adjust repetition penalty and return multiple sequences:

```bash
echo "Prompt" | python generate.py -m gpt2 -d cpu --repetition-penalty 1.2 -n 3
```


REPL and voice
--------------

Start an interactive conversational REPL locally (optional TTS):

```bash
# dry-run REPL (no heavy installs)
python generate.py --repl --dry-run

# REPL with TTS (requires pyttsx3 installed)
python generate.py --repl --tts
```

Conversation history is saved to `conversation.json` by default. You can change it with `--history-file my_history.json`.

Offline speech-to-text (VOSK)
-----------------------------

We support offline STT using VOSK. This requires downloading a VOSK model and installing audio dependencies.

1) Install system deps (Linux example):

```bash
# Debian/Ubuntu
sudo apt-get install -y build-essential libsndfile1
```

2) Install Python deps (in venv):

```bash
pip install -r requirements.txt
```

3) Download a small English VOSK model and place it at `models/vosk-model-small-en-us-0.15`:

```bash
mkdir -p models
cd models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
cd ..
```

4) Run the Flask app and record/send audio:

```bash
python app.py
./scripts/record_and_send.py --duration 5 --url http://localhost:5000
```

The recorded audio will be transcribed locally and the transcript sent to the chat endpoint (dry-run by default).





# brainai — quick start

Prerequisites: Python 3.8+ and a virtual environment.

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




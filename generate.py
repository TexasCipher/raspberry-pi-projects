import argparse
import sys
import logging


def make_mock_pipeline():
    def p(prompt, **kwargs):
        n = int(kwargs.get("num_return_sequences", 1) or 1)
        # include a short note about key generation params in dry-run
        info = []
        if kwargs.get("do_sample") is not None:
            info.append(f"sample={kwargs.get('do_sample')}")
        if kwargs.get("temperature") is not None:
            info.append(f"temp={kwargs.get('temperature')}")
        if kwargs.get("top_k") is not None:
            info.append(f"top_k={kwargs.get('top_k')}")
        if kwargs.get("top_p") is not None:
            info.append(f"top_p={kwargs.get('top_p')}")
        info_str = " " + " ".join(info) if info else ""
        return [{"generated_text": f"{prompt} [DRY RUN]{info_str} (seq {i+1})"} for i in range(n)]

    return p


def init_pipeline(model="distilgpt2", device_opt="cpu", dry_run=False, logger=None):
    """Initialize and return a generation pipeline (callable). Returns (pipeline_callable, device).

    If dry_run is True, returns the mock pipeline.
    """
    if logger is None:
        import logging

        logger = logging.getLogger(__name__)

    if dry_run:
        return make_mock_pipeline(), "dry"

    try:
        from transformers import pipeline, set_seed
    except Exception:
        logger.error("Missing required packages. Install with: pip install -r requirements.txt")
        raise

    try:
        set_seed(42)
    except Exception:
        logger.debug("set_seed not available")

    dev_opt = device_opt.lower()
    if dev_opt == "cpu":
        device = -1
    elif dev_opt.startswith("cuda") or dev_opt == "gpu":
        device = 0
    else:
        try:
            device = int(device_opt)
        except Exception:
            device = -1

    logger.info("Loading model %s on device %s", model, device_opt)
    p = pipeline("text-generation", model=model, device=device)
    return p, device


def generate_text(pipeline_callable, prompt, max_new_tokens=120, do_sample=True, temperature=0.8, top_k=0, top_p=0.92, repetition_penalty=1.15, num_return_sequences=1):
    return pipeline_callable(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        return_full_text=False,
        num_return_sequences=num_return_sequences,
        truncation=True,
        pad_token_id=50256,
    )


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Simple text generator")
    parser.add_argument("--prompt", "-p", help="Prompt text (optional; will read stdin if omitted)")
    parser.add_argument("--file", "-f", help="Read prompt from a file")
    parser.add_argument("--model", "-m", default="distilgpt2", help="Model identifier to pass to transformers pipeline")
    parser.add_argument("--device", "-d", default="cpu", help="Device to run on: 'cpu', 'cuda', or integer GPU id (default: cpu)")
    parser.add_argument("--num-return-sequences", "-n", type=int, default=1, help="Number of returned sequences (default: 1)")
    parser.add_argument("--max-new-tokens", type=int, default=120, help="Maximum number of new tokens to generate (default: 120)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 to disable, default: 0)")
    parser.add_argument("--top-p", type=float, default=0.92, help="Top-p (nucleus) sampling (default: 0.92)")
    parser.add_argument("--repetition-penalty", type=float, default=1.15, help="Repetition penalty (default: 1.15)")
    parser.add_argument("--no-sample", action="store_true", help="Disable sampling (use greedy decoding)")
    parser.add_argument("--dry-run", action="store_true", help="Run without importing heavy libraries")
    parser.add_argument("--repl", action="store_true", help="Enter conversational REPL mode")
    parser.add_argument("--tts", action="store_true", help="Speak responses using pyttsx3 if available")
    parser.add_argument("--history-file", default="conversation.json", help="File to persist conversation history in REPL mode")
    args = parser.parse_args(argv)

    prompt = args.prompt

    # basic logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Starting generation (dry-run=%s)", args.dry_run)
    if not prompt and args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as fh:
                prompt = fh.read().strip()
        except Exception:
            print(f"Failed to read file: {args.file}")
            return 3

    if not prompt:
        # try reading from stdin (non-interactive)
        try:
            prompt = sys.stdin.read().strip()
        except Exception:
            prompt = ""

    if not prompt:
        # interactive fallback
        try:
            prompt = input("Prompt: ").strip()
        except Exception:
            prompt = ""

    if not prompt:
        print("No prompt provided")
        return 1

    # Initialize pipeline (use single init function to avoid duplication)
    try:
        p, device = init_pipeline(model=args.model, device_opt=args.device, dry_run=args.dry_run, logger=logger)
    except Exception:
        logger.error("Failed to initialize generation pipeline. Ensure transformers and torch are installed or use --dry-run.")
        return 2

    # Ensure tokenizer has a pad token to silence warnings when padding is needed
    try:
        tokenizer = getattr(p, "tokenizer", None)
        if tokenizer is not None:
            if getattr(tokenizer, "pad_token", None) is None:
                # fall back to eos token if available
                if getattr(tokenizer, "eos_token", None) is not None:
                    try:
                        tokenizer.pad_token = tokenizer.eos_token
                        if getattr(tokenizer, "eos_token_id", None) is not None:
                            tokenizer.pad_token_id = tokenizer.eos_token_id
                    except Exception:
                        pass
    except Exception:
        pass

    do_sample = not args.no_sample

    # try to import TTS engine if requested
    tts_engine = None
    if args.tts:
        try:
            import pyttsx3

            tts_engine = pyttsx3.init()
        except Exception:
            logger.warning("pyttsx3 not available; TTS disabled")
            tts_engine = None

    def generate_once(prompt_text):
        return p(
            prompt_text,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            return_full_text=False,
            num_return_sequences=args.num_return_sequences,
            truncation=True,
            pad_token_id=50256,
        )

    # REPL mode: interactive loop with optional history persistence and TTS
    if args.repl:
        import json
        history = []
        try:
            if args.history_file:
                with open(args.history_file, "r", encoding="utf-8") as hf:
                    history = json.load(hf)
        except Exception:
            history = []

        print("Entering REPL mode. Type 'exit' or Ctrl-C to quit.")
        try:
            while True:
                try:
                    user = input("You: ").strip()
                except EOFError:
                    break
                if not user:
                    continue
                if user.lower() in ("exit", "quit"):
                    break

                # append to history
                history.append({"role": "user", "text": user})

                # generate
                resp = generate_once(user)
                try:
                    text = resp[0].get("generated_text") if isinstance(resp[0], dict) else str(resp[0])
                except Exception:
                    text = str(resp)

                print("AI:", text.strip())
                history.append({"role": "assistant", "text": text.strip()})

                # speak
                if tts_engine:
                    try:
                        tts_engine.say(text.strip())
                        tts_engine.runAndWait()
                    except Exception:
                        logger.warning("TTS engine failed to speak")

                # persist history
                try:
                    if args.history_file:
                        with open(args.history_file, "w", encoding="utf-8") as hf:
                            json.dump(history[-100:], hf, ensure_ascii=False, indent=2)
                except Exception:
                    logger.debug("Failed to save history")

        except KeyboardInterrupt:
            print("\nExiting REPL")

        return 0

    # single-shot generation
    out = generate_once(prompt)

    print("\n---\n")
    try:
        if args.num_return_sequences and args.num_return_sequences > 1:
            for i, item in enumerate(out):
                text = item.get("generated_text") if isinstance(item, dict) else str(item)
                print(f"[{i+1}] {text.strip()}\n")
        else:
            text = out[0].get("generated_text") if isinstance(out[0], dict) else str(out[0])
            print(text.strip())
    except Exception as e:
        logger.exception("Error while printing output: %s", e)
        print(out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

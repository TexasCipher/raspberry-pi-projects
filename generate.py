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

    # Lazy import transformers so --dry-run works without installing it
    if args.dry_run:
        p = make_mock_pipeline()
    else:
        try:
            from transformers import pipeline, set_seed
        except Exception:
            logger.error("Missing required packages. Install with: pip install -r requirements.txt")
            return 2

        # consistent-ish results
        try:
            set_seed(42)
        except Exception:
            logger.debug("set_seed not available")

        # parse device option
        dev_opt = args.device.lower()
        if dev_opt == "cpu":
            device = -1
        elif dev_opt.startswith("cuda") or dev_opt == "gpu":
            # default to GPU 0 if 'cuda' or 'gpu' provided
            device = 0
        else:
            try:
                device = int(args.device)
            except Exception:
                device = -1

        logger.info("Loading model %s on device %s", args.model, args.device)
        try:
            p = pipeline("text-generation", model=args.model, device=device)
        except Exception as e:
            logger.exception("Failed to initialize model pipeline: %s", e)
            return 4

    do_sample = not args.no_sample

    out = p(
        prompt,
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

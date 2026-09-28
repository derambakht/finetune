"""Command line interface.

Examples::

    ftstudio ui --port 7860
    ftstudio init-config my_run.yaml
    ftstudio validate --config my_run.yaml
    ftstudio train --config my_run.yaml
    ftstudio export --config my_run.yaml --source outputs/run/final_model --methods gguf --gguf q4_k_m q8_0
    ftstudio chat --model outputs/run/final_model
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import EXPORT_METHODS, GGUF_QUANT_METHODS, FinetuneConfig, validate_config


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def _load_cfg(path: str | None) -> FinetuneConfig:
    return FinetuneConfig.load(path) if path else FinetuneConfig()


def cmd_ui(args) -> int:
    from .ui.app import launch

    launch(host=args.host, port=args.port, share=args.share, output_root=args.output_root)
    return 0


def cmd_init(args) -> int:
    path = FinetuneConfig().save(args.path)
    print(f"Wrote default configuration to {path}")
    return 0


def cmd_validate(args) -> int:
    res = validate_config(_load_cfg(args.config))
    for e in res.errors:
        print(f"ERROR: {e}")
    for w in res.warnings:
        print(f"WARNING: {w}")
    print("OK" if res.ok else "INVALID")
    return 0 if res.ok else 1


def cmd_train(args) -> int:
    from .runner import make_run_dir
    from .trainer import run_training

    cfg = _load_cfg(args.config)
    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps
    run_dir = Path(args.run_dir) if args.run_dir else make_run_dir(cfg)
    if not args.run_dir:
        cfg.save(run_dir / "config.yaml")
    logging.getLogger("ftstudio").info("Run directory: %s", run_dir)
    summary = run_training(cfg, run_dir)
    print(f"Final model saved to {summary['final_model']}")
    return 0


def cmd_export(args) -> int:
    from .export import run_export

    cfg = _load_cfg(args.config)
    if args.methods:
        cfg.export.methods = args.methods
    if args.gguf:
        cfg.export.gguf_quantizations = args.gguf
    if args.push:
        cfg.export.push_to_hub = True
        cfg.export.hub_repo_id = args.push
    out = args.out or str(Path(args.source).parent / "export")
    results = run_export(cfg, args.source, out)
    for k, v in results.items():
        print(f"{k}: {v}")
    return 0


def cmd_chat(args) -> int:
    from .inference import ChatEngine

    engine = ChatEngine()
    print(engine.load(args.model, max_seq_length=args.max_seq_length, quantization=args.quantization,
                      chat_template=args.chat_template))
    history: list[dict[str, str]] = []
    if args.system:
        history.append({"role": "system", "content": args.system})
    print("Type your message (/reset to clear, /exit to quit).")
    while True:
        try:
            user = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            history = history[:1] if args.system else []
            continue
        if not user:
            continue
        history.append({"role": "user", "content": user})
        answer, printed = "", 0
        for answer in engine.stream(history, max_new_tokens=args.max_new_tokens, temperature=args.temperature):
            print(answer[printed:], end="", flush=True)
            printed = len(answer)
        print()
        history.append({"role": "assistant", "content": answer})
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ftstudio", description="Unsloth FineTune Studio")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ui", help="Launch the web UI")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=7860)
    s.add_argument("--share", action="store_true", help="Create a public Gradio link (e.g. on Colab)")
    s.add_argument("--output-root", default="outputs")
    s.set_defaults(func=cmd_ui)

    s = sub.add_parser("init-config", help="Write a default YAML config")
    s.add_argument("path", nargs="?", default="config.yaml")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("validate", help="Validate a YAML config")
    s.add_argument("--config", required=True)
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("train", help="Run fine-tuning from a YAML config")
    s.add_argument("--config", required=True)
    s.add_argument("--run-dir", default=None)
    s.add_argument("--max-steps", type=int, default=None, help="Override training.max_steps")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("export", help="Export a trained model")
    s.add_argument("--config", default=None)
    s.add_argument("--source", required=True, help="Directory of the trained model / adapter")
    s.add_argument("--out", default=None)
    s.add_argument("--methods", nargs="+", choices=list(EXPORT_METHODS))
    s.add_argument("--gguf", nargs="+", choices=list(GGUF_QUANT_METHODS))
    s.add_argument("--push", default=None, metavar="REPO_ID", help="Also push to the Hugging Face Hub")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("chat", help="Chat with a model in the terminal")
    s.add_argument("--model", required=True)
    s.add_argument("--max-seq-length", type=int, default=2048)
    s.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", "16bit"])
    s.add_argument("--chat-template", default="auto")
    s.add_argument("--system", default="")
    s.add_argument("--max-new-tokens", type=int, default=512)
    s.add_argument("--temperature", type=float, default=0.7)
    s.set_defaults(func=cmd_chat)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

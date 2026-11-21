"""Command-line entry point; never computes benchmark scores."""
import argparse
import json
from pathlib import Path
from .config import MethodConfig
from .pipeline import load_long_csv, run_walkforward


def main(argv=None):
    parser = argparse.ArgumentParser(description="GLCast method-only historical prediction runner")
    parser.add_argument("action", choices=["validate", "run"])
    parser.add_argument("--input", required=True, help="User-supplied long CSV")
    parser.add_argument("--target", required=True, help="Numeric target column")
    parser.add_argument("--aux", nargs="+", required=True, help="Numeric auxiliary columns, in model order")
    parser.add_argument("--date", default="date", help="Date column (default: date)")
    parser.add_argument("--state", default="state", help="State column (default: state)")
    parser.add_argument("--config", help="JSON method configuration (default: modern weekly)")
    parser.add_argument("--output", help="Output directory, required for run")
    args = parser.parse_args(argv)
    try:
        cfg = MethodConfig.from_json(args.config) if args.config else MethodConfig()
        target, features = load_long_csv(args.input, target=args.target, auxiliaries=args.aux,
                                         config=cfg, date=args.date, state=args.state)
        print(f"Valid input: {len(target)} dates, {len(target.columns)} states, {len(features)} auxiliaries")
        if args.action == "validate":
            return 0
        if not args.output:
            parser.error("--output is required for run")
        result = run_walkforward(target, features, cfg, progress=lambda text: print(text, flush=True))
        if result.predictions.empty:
            raise ValueError("No predictions generated; nothing was saved")
        out = Path(args.output)
        result.save(out)
        (out / "configuration.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
        print(f"Saved {len(result.predictions)} predictions to {out}")
        return 0
    except (ValueError, KeyError, OSError) as e:
        parser.error(str(e))


if __name__ == "__main__":
    raise SystemExit(main())

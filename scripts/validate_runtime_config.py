#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate PPE Guardian runtime configuration.")
    parser.add_argument(
        "--require-valid-ppe-model",
        action="store_true",
        help="Fail if the configured PPE model does not expose the expected PPE classes.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    startup = app.build_startup_checklist()
    integrity = app.run_startup_validation(fail_on_error=False)
    payload = {
        "startup": startup,
        "model_integrity": integrity,
    }
    print(json.dumps(payload, indent=2))

    if not startup["ok"]:
        return 1
    if args.require_valid_ppe_model and not integrity["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

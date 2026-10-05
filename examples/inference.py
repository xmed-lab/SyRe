#!/usr/bin/env python3
"""Minimal example: segment one class in one image with SyRe."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from inference_syre import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

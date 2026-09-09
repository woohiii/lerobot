"""Fail-closed command-line preflight for the pick-and-place pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ConfigError, load_pipeline_config
from .graspnet_adapter import GraspNetUnavailable, validate_graspnet_runtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SO-101 GraspNet pick-and-place preflight")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--graspnet-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run == args.execute:
        parser.error("choose exactly one of --dry-run or --execute")
    try:
        config = load_pipeline_config(args.config)
        if args.execute:
            config.require_execution_ready()
            if args.graspnet_root is None or args.checkpoint is None:
                raise ConfigError("--execute requires --graspnet-root and --checkpoint")
            validate_graspnet_runtime(args.graspnet_root, args.checkpoint)
    except (ConfigError, GraspNetUnavailable, ValueError) as exc:
        parser.error(str(exc))
    mode = "dry-run" if args.dry_run else "execute-preflight"
    print(f"{mode}: configuration accepted; no hardware command was issued")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

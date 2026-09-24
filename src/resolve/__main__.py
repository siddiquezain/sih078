"""CLI entry point: python -m resolve <cmd> --config <path>"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="resolve")
    sub = parser.add_subparsers(dest="cmd", required=True)

    hero_p = sub.add_parser("hero", help="Produce the Montha hero figure")
    hero_p.add_argument("--config", required=True, help="Path to YAML config")
    hero_p.add_argument("--dry-run", action="store_true",
                        help="Load config and print plan without fetching data")

    inspect_p = sub.add_parser(
        "inspect",
        help="Probe catalog schema and compute one 24 h accumulation (no full ingest)",
    )
    inspect_p.add_argument("--config", required=True, help="Path to YAML config")

    args = parser.parse_args()

    if args.cmd == "hero":
        from resolve.hero import run_hero
        run_hero(args.config, dry_run=args.dry_run)
    elif args.cmd == "inspect":
        from resolve.inspect_cmd import run_inspect
        run_inspect(args.config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

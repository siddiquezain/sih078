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

    export_p = sub.add_parser(
        "export-ui",
        help="Write ui/data/<event>.js from pipeline results (resolve-ui/1 schema)",
    )
    export_p.add_argument("--config", required=True, help="Path to YAML config")

    ui_p = sub.add_parser("ui", help="Serve the ui/ directory on :8765")
    ui_p.add_argument("--port", type=int, default=8765, help="Port (default 8765)")

    args = parser.parse_args()

    if args.cmd == "hero":
        from resolve.hero import run_hero
        run_hero(args.config, dry_run=args.dry_run)
    elif args.cmd == "inspect":
        from resolve.inspect_cmd import run_inspect
        run_inspect(args.config)
    elif args.cmd == "export-ui":
        from resolve.export_ui import run_export_ui
        run_export_ui(args.config)
    elif args.cmd == "ui":
        import http.server
        import os
        ui_dir = "ui"
        os.chdir(ui_dir)
        port = args.port
        handler = http.server.SimpleHTTPRequestHandler
        with http.server.HTTPServer(("", port), handler) as httpd:
            print(f"Serving ui/ at http://localhost:{port}/  (Ctrl-C to stop)")
            httpd.serve_forever()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

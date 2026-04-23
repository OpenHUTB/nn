import argparse
from datetime import datetime


def on_config(config, **kwargs):
    year = str(datetime.now().year)
    config.copyright = config.copyright.format(year=year)
def main():
    """
    Command line entry point for hooks.
    """
    parser = argparse.ArgumentParser(
        description="Utility hooks for nn project"
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information"
    )

    args = parser.parse_args()

    if args.version:
        print("nn hooks version 1.0")
        return

    print("Hooks executed successfully.")


if __name__ == "__main__":
    main()
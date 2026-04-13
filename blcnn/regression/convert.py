import argparse
from pathlib import Path

from compare import load_classification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the classification data"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Location of regressive output file"
    )
    args = parser.parse_args()
    df = load_classification(Path(args.input))
    df.to_csv(Path(args.output), index=False)


if __name__ == "__main__":
    main()

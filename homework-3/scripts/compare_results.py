"""Compare every HW3 generated token; report lifecycle evidence as pending."""

import argparse
import json
from pathlib import Path

from run_offloading import compare_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_runs(json.loads(args.reference.read_text()),
                          json.loads(args.candidate.read_text()))
    with args.output.open("x") as file:
        file.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

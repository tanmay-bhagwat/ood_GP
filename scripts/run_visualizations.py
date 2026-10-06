from argparse import ArgumentParser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ood_gp.plots import run_plots


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="Directory containing run_manifest.json and predictions.npz")
    parser.add_argument("--output-dir", type=Path, help="Plot destination (default: RUN_DIRECTORY/plots)")
    parser.add_argument("--normalized", action="store_true", help="Keep predictions in normalized target units")

    arguments = parser.parse_args()
    run_directory = arguments.run_dir
    if not run_directory.is_absolute():
        run_directory = PROJECT_ROOT / run_directory
    run_directory = run_directory.resolve()
    output_directory = (
        arguments.output_dir.expanduser().resolve()
        if arguments.output_dir is not None
        else run_directory / "plots")
    run_plots(run_directory, output_directory, arguments.normalized)


if __name__ == "__main__":
    main()

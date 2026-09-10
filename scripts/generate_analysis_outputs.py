#!/usr/bin/env python3
"""Generate Issue #23 analysis tables/figures from frozen artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

from lagged_facial_graph_forecasting.analysis_pipeline import AnalysisInputs, generate_analysis_outputs
from lagged_facial_graph_forecasting.sensitivity_execution import load_sensitivity_experiment_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--publication-ready", action="store_true")
    parser.add_argument("--include-sensitivity", action="store_true")
    parser.add_argument("--sensitivity-config", type=Path)
    parser.add_argument(
        "--allow-mock-sensitivity",
        action="store_true",
        help="software-verification only; never scientific validation",
    )
    args = parser.parse_args()
    result = generate_analysis_outputs(
        AnalysisInputs.from_directory(args.input_dir, include_sensitivity=args.include_sensitivity),
        repository_root=args.repository_root,
        output_root=args.output_root,
        publication_ready=args.publication_ready,
        include_sensitivity=args.include_sensitivity,
        sensitivity_config=(
            load_sensitivity_experiment_config(
                args.sensitivity_config, repository_root=args.repository_root
            )
            if args.include_sensitivity and args.sensitivity_config else None
        ),
        allow_mock_sensitivity=args.allow_mock_sensitivity,
    )
    print(result.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""`lmb` command-line entrypoint: run / render / convert."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .core.experiment import Experiment, rerender, run
from .systems import get_system


def _cmd_run(args: argparse.Namespace) -> int:
    exp = Experiment(
        system=args.system, params=args.params, dataset=args.dataset, testset=args.testset
    )
    run_dir = run(exp, results_root=args.results)
    print(f"Run complete: {run_dir}")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    """Re-render memory HTML from a frozen state.json (no LLM calls)."""
    system_cls = get_system(args.system)
    state_dir = Path(args.state_dir)
    out_dir = Path(args.out or state_dir)
    system_cls.render_from_state(state_dir, out_dir)
    print(f"Rendered memory viz from {state_dir} -> {out_dir}")
    return 0


def _cmd_rerender(args: argparse.Namespace) -> int:
    """Re-render every run's HTML for an experiment from frozen json (no LLM calls)."""
    exp_dir = rerender(args.dataset, args.testset, results_root=args.results)
    print(f"Re-rendered results under {exp_dir}")
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    from .convert import convert

    ds, ts = convert(args.src, name=args.name)
    print(f"Wrote dataset {ds}\nWrote test-set {ts}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(prog="lmb", description="LLM memory benchmark framework")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run one experiment")
    p_run.add_argument("--system", default="amem")
    p_run.add_argument("--params", default="default")
    p_run.add_argument("--dataset", default="comparison")
    p_run.add_argument("--testset", default="comparison")
    p_run.add_argument("--results", default="results")
    p_run.set_defaults(func=_cmd_run)

    p_render = sub.add_parser("render", help="re-render memory viz from state.json")
    p_render.add_argument("--system", default="amem")
    p_render.add_argument("--state-dir", required=True)
    p_render.add_argument("--out", default=None)
    p_render.set_defaults(func=_cmd_render)

    p_rr = sub.add_parser("rerender", help="re-render all viz for an experiment from frozen json")
    p_rr.add_argument("--dataset", default="comparison")
    p_rr.add_argument("--testset", default="comparison")
    p_rr.add_argument("--results", default="results")
    p_rr.set_defaults(func=_cmd_rerender)

    p_conv = sub.add_parser("convert", help="convert reference dataset.json to YAML")
    p_conv.add_argument("src", help="path to reference dataset.json")
    p_conv.add_argument("--name", default="comparison")
    p_conv.set_defaults(func=_cmd_convert)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

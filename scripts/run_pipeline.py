"""scripts/run_pipeline.py —— 端到端数据管线 CLI（mock 硬件）。

用法（在本仓库根目录）::

    py -3.13 scripts/run_pipeline.py --elevations 15,45,75 --terms Z4,Z8,Z9 \
        --poses 0.05 --out results/pipeline/<date>

    py -3.13 scripts/run_pipeline.py --elevations 15,75 --dofs 2 \
        --out results/pipeline/smoke

它用 :mod:`products.acquisition.mock` 的显式 mock 台架
（label="SYNTHETIC-VALIDATED", mode="mock"）跑完整工作流：俯仰角扫描 +
补偿机构位姿扫描 + CCD 预处理 + PD 反演 + 实测灵敏度 + 有界最小二乘补偿 +
不可补偿项的 FEA/装调请求。输出三个文件：

    pipeline_result.json     完整结果（含 stages/per_elevation/manifest）
    compensation_table.csv   每个俯仰角 x DOF 的补偿量/residual（CSV-ready）
    fea_request.json         有限元分析/装调前处理请求（不是 FEA 结果）

边界：没有真实望远镜/相机/机构 API，也没有工况数据；本脚本不会连接
任何硬件，也不运行 FEA。退出码 0=成功，2=参数/运行错误。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# 允许以 `py -3.13 scripts/run_pipeline.py` 直接运行（脚本目录不在 repo 根）。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _split_floats(values: Sequence[str]) -> List[float]:
    out: List[float] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(float(part))
    return out


def _split_words(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description=("End-to-end wavefront data pipeline on the explicit mock "
                     "rig (no telescope/camera/mechanism SDK; no operating-"
                     "condition data)."))
    parser.add_argument("--elevations", nargs="+", default=["15,45,75"],
                        help="elevation scan in deg, e.g. 15,45,75")
    parser.add_argument("--terms", nargs="+", default=["Z4,Z8,Z9"],
                        help="PD Zernike terms, e.g. Z4,Z8,Z9")
    parser.add_argument("--poses", type=float, default=0.05,
                        help="mechanism pose amplitude (DOF unit, default 0.05)")
    parser.add_argument("--out", default=None,
                        help=("output directory; default "
                              "results/pipeline/<YYYYMMDD-HHMMSS>"))
    parser.add_argument("--dofs", type=int, default=2,
                        help="number of mock mechanism DOFs (1..5, default 2)")
    parser.add_argument("--defocus", nargs="+", default=["-2,-1,1,2"],
                        help="defocus frame list for PD, e.g. -2,-1,1,2")
    parser.add_argument("--frames", type=int, default=3,
                        help="frames per master calibration (default 3)")
    parser.add_argument("--exposure", type=float, default=0.5,
                        help="exposure per frame in s (default 0.5)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="pose-series repeats (default 1)")
    parser.add_argument("--seed", type=int, default=7,
                        help="mock seed (deterministic, default 7)")
    parser.add_argument("--calib-dir", default=None,
                        help="directory with master_frames.npz (else synthetic)")
    parser.add_argument("--rel-threshold", type=float, default=0.2,
                        help="relative observability threshold (default 0.2)")
    parser.add_argument("--maxeval", type=int, default=80,
                        help="PD solver max evaluations (default 80)")
    parser.add_argument("--n-starts", type=int, default=2,
                        help="PD multi-start count (default 2)")
    parser.add_argument("--psf-window", type=int, default=None,
                        help="PSF window override (default: camera metadata)")
    parser.add_argument("--keep-frames", action="store_true",
                        help="keep raw frames in the result (debug; memory heavy)")
    parser.add_argument("--stop-on-error", action="store_true", default=True,
                        help="abort on the first elevation error (default)")
    parser.add_argument("--continue-on-error", dest="stop_on_error",
                        action="store_false",
                        help="skip failed elevations and record warnings")
    return parser


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def _print_summary(result: Dict[str, Any], out: Path, elapsed_s: float) -> None:
    print(f"mode={result['mode']} label={result['label']} "
          f"status={result['status']}")
    print(f"elapsed={elapsed_s:.2f}s")
    stages = result["stages"]
    order = ["calibration", "capture", "preprocess", "invert",
             "sensitivity", "compensation", "observability",
             "elevation_loop", "total"]
    print("stages: " + "  ".join(
        f"{key}={stages.get(key, 0.0):.3f}s" for key in order))
    for rec in result["per_elevation"]:
        targets = rec["target_aberration_waves"]
        cmds = rec["compensation"]["commands"]
        dropped = rec["observability"]["dropped_rows"]
        target_txt = " ".join(f"{t}={v:+.4f}" for t, v in targets.items())
        cmd_txt = " ".join(f"{d}={v:+.5f}" for d, v in cmds.items())
        print(f"elev {rec['elevation_deg']:6.1f}: target[{target_txt}] "
              f"cmd[{cmd_txt}] residual_rms="
              f"{rec['compensation']['residual_rms_waves']:.4f} waves "
              f"unobservable={dropped}")
    fea = result["fea_request"]
    print(f"compensation_table: {len(result['compensation_table'])} rows -> "
          f"{(out / 'compensation_table.csv')}")
    print(f"fea_request: un_compensatable_terms={fea['un_compensatable_terms']} "
          f"-> {(out / 'fea_request.json')}")
    print(f"warnings: {len(result['warnings'])}")
    for warning in result["warnings"]:
        print(f"  - {warning}")
    print(f"result -> {(out / 'pipeline_result.json')}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        elevations = _split_floats(args.elevations)
        terms = _split_words(args.terms)
        defocus = _split_floats(args.defocus)
    except ValueError as exc:
        print(f"argument error: {exc}", file=sys.stderr)
        return 2
    out = (Path(args.out) if args.out
           else ROOT / "results" / "pipeline"
           / datetime.now().strftime("%Y%m%d-%H%M%S"))

    try:
        from products.acquisition.mock import build_mock_rig
        from products.pipeline import PipelineConfig, WavefrontPipeline

        rig = build_mock_rig(n_dofs=int(args.dofs), seed=int(args.seed))
        config = PipelineConfig(
            elevations_deg=elevations,
            defocus=defocus,
            n_frames=int(args.frames),
            exposure_s=float(args.exposure),
            terms=terms,
            pose_amplitude=float(args.poses),
            repeats=int(args.repeats),
            seed=int(args.seed),
            calib_dir=args.calib_dir,
            stop_on_error=bool(args.stop_on_error),
            keep_frames=bool(args.keep_frames),
            rel_threshold=float(args.rel_threshold),
            pd_maxeval=int(args.maxeval),
            pd_n_starts=int(args.n_starts),
            psf_window=args.psf_window,
        )
        pipeline = WavefrontPipeline(rig.telescope, rig.camera, rig.mechanism,
                                     config)
        start = time.perf_counter()
        result = pipeline.run()
        elapsed = time.perf_counter() - start
    except Exception as exc:                        # noqa: BLE001
        print(f"pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    try:
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "pipeline_result.json", result)
        _write_csv(out / "compensation_table.csv",
                   result["compensation_table"])
        _write_json(out / "fea_request.json", result["fea_request"])
    except OSError as exc:
        print(f"failed to write outputs: {exc}", file=sys.stderr)
        return 2
    _print_summary(result, out, elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

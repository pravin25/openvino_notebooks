"""Run OpenVINO benchmark_app across exported YOLO26 pose variants and precisions.

Wraps the OpenVINO `benchmark_app` CLI (latency hint, static input shape per variant) and prints
one combined table (median latency ms, throughput FPS) across nano/small/medium/large x
fp32/fp16/int8 x GPU. Requires the models under `yolo26_export_model` to already exist.

Usage:
    python ov_yolo26_benchmark.py --variant medium
    python ov_yolo26_benchmark.py --variant all
    python ov_yolo26_benchmark.py --variant all --time 10 --devices GPU
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VARIANTS = ("nano", "small", "medium", "large")
PRECISIONS = ("fp32", "fp16", "int8")
MODEL_CONFIGS = {
    "nano": ("yolo26n-pose", 384),
    "small": ("yolo26s-pose", 512),
    "medium": ("yolo26m-pose", 576),
    "large": ("yolo26l-pose", 704),
}

_LATENCY_RE = re.compile(r"Median:\s+([\d.]+)\s*(us|ms)")
_THROUGHPUT_RE = re.compile(r"Throughput:\s+([\d.]+)\s*FPS")


def parse_args() -> argparse.Namespace:
    """Parse benchmark options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    parser.add_argument("--devices", nargs="+", choices=["CPU", "GPU"], default=["GPU"])
    parser.add_argument("--time", type=int, default=15, help="Benchmark duration per model, in seconds")
    return parser.parse_args()


def run_benchmark_app(model_path: Path, device: str, resolution: int, duration: int) -> tuple[float, float]:
    """Run benchmark_app once and return (median_latency_ms, throughput_fps)."""
    cmd = [
        "benchmark_app",
        "-m", str(model_path),
        "-d", device,
        "-shape", f"[1,3,{resolution},{resolution}]",
        "-hint", "latency",
        "-t", str(duration),
    ]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    output = result.stdout + result.stderr
    latency_match = _LATENCY_RE.search(output)
    throughput_match = _THROUGHPUT_RE.search(output)
    if latency_match is None or throughput_match is None:
        raise RuntimeError(f"Could not parse benchmark_app output for {model_path} on {device}:\n{output[-2000:]}")
    latency_ms = float(latency_match.group(1))
    if latency_match.group(2) == "us":
        latency_ms /= 1000
    return latency_ms, float(throughput_match.group(1))


def main() -> None:
    """Benchmark all requested variant/precision/device combinations and print one table."""
    args = parse_args()
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]

    rows: list[tuple[str, str, str, str, float, float]] = []
    for variant in variants:
        model_name, resolution = MODEL_CONFIGS[variant]
        input_shape = f"[1,3,{resolution},{resolution}]"
        for precision in PRECISIONS:
            model_path = ROOT / "yolo26_export_model" / model_name / precision / f"{model_name}.xml"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Missing OpenVINO model: {model_path}. Run export_yolo26_openvino.py first."
                )
            for device in args.devices:
                print(f'Benchmarking {variant}/{precision} on {device} with -shape "{input_shape}"...')
                latency_ms, throughput_fps = run_benchmark_app(model_path, device, resolution, args.time)
                rows.append((variant, precision, device, input_shape, latency_ms, throughput_fps))

    header = f"{'Variant':<8} {'Precision':<9} {'Device':<6} {'Input Shape':<17} {'Latency (ms)':>13} {'Throughput (FPS)':>17}"
    print(f"\n{header}")
    print("-" * len(header))
    previous_variant = None
    for variant, precision, device, input_shape, latency_ms, throughput_fps in rows:
        if previous_variant is not None and variant != previous_variant:
            print()
        print(f"{variant:<8} {precision:<9} {device:<6} {input_shape:<17} {latency_ms:>13.2f} {throughput_fps:>17.2f}")
        previous_variant = variant


if __name__ == "__main__":
    main()

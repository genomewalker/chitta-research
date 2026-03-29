#!/usr/bin/env python3
"""
MCAF compression+speed verifier for chitta-research.

Emits a VerificationResult JSON to stdout.

Usage:
  verify_mcaf.py                  # measure compression ratio (default)
  verify_mcaf.py --mode speed     # measure scan time
  verify_mcaf.py --mcaf PATH      # explicit MCAF file
  verify_mcaf.py --bam  PATH      # explicit BAM file for ratio comparison
"""
import argparse
import json
import os
import sys
import time

MCAF_PATH = "/maps/projects/caeg/scratch/kbd606/bam-filter/viking/LV7008867351.sorted.mcaf"
BAM_PATH  = "/maps/projects/caeg/scratch/kbd606/bam-filter/viking/LV7008867351.sorted.bam"


def file_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0


def measure_ratio(mcaf: str, bam: str) -> dict:
    mcaf_mb = file_size_mb(mcaf)
    bam_mb  = file_size_mb(bam)

    if bam_mb == 0:
        return {
            "status": {"kind": "invalid"},
            "metrics": {},
            "notes": f"BAM not found: {bam}",
        }
    if mcaf_mb == 0:
        return {
            "status": {"kind": "invalid"},
            "metrics": {},
            "notes": f"MCAF not found: {mcaf}",
        }

    ratio = bam_mb / mcaf_mb  # >1 means MCAF is smaller than BAM

    return {
        "status": {"kind": "pass" if ratio > 1.0 else "fail"},
        "metrics": {
            "compression_ratio": round(ratio, 4),
            "mcaf_size_mb": round(mcaf_mb, 2),
            "bam_size_mb":  round(bam_mb, 2),
            "space_saved_mb": round(bam_mb - mcaf_mb, 2),
        },
        "baseline_metrics": {
            "compression_ratio": 1.0,
        },
        "supports": [f"MCAF is {ratio:.2f}x smaller than BAM"] if ratio > 1 else [],
        "refutes":  [f"MCAF is larger than BAM ({mcaf_mb:.1f} MB vs {bam_mb:.1f} MB)"] if ratio <= 1 else [],
        "notes": f"mcaf={mcaf_mb:.1f} MB  bam={bam_mb:.1f} MB  ratio={ratio:.4f}",
    }


def measure_speed(mcaf: str) -> dict:
    if not os.path.exists(mcaf):
        return {
            "status": {"kind": "invalid"},
            "metrics": {},
            "notes": f"MCAF not found: {mcaf}",
        }

    # Measure raw sequential read throughput as a proxy for decode speed.
    # A real mcaf_info scan command can replace this once the CLI supports --json.
    chunk = 4 * 1024 * 1024  # 4 MB chunks
    t0 = time.perf_counter()
    total = 0
    try:
        with open(mcaf, "rb") as f:
            while True:
                buf = f.read(chunk)
                if not buf:
                    break
                total += len(buf)
    except OSError as e:
        return {
            "status": {"kind": "invalid"},
            "metrics": {},
            "notes": str(e),
        }
    elapsed_ms = (time.perf_counter() - t0) * 1000
    throughput_mb_s = (total / (1024 * 1024)) / max(elapsed_ms / 1000, 1e-6)

    return {
        "status": {"kind": "pass"},
        "metrics": {
            "scan_time_ms":      round(elapsed_ms, 1),
            "throughput_mb_s":   round(throughput_mb_s, 1),
            "bytes_read":        total,
        },
        "baseline_metrics": {
            "scan_time_ms": elapsed_ms,  # first run is its own baseline
        },
        "notes": f"sequential read: {elapsed_ms:.0f} ms  {throughput_mb_s:.0f} MB/s",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  default="ratio", choices=["ratio", "speed"])
    parser.add_argument("--mcaf",  default=MCAF_PATH)
    parser.add_argument("--bam",   default=BAM_PATH)
    args = parser.parse_args()

    if args.mode == "speed":
        result = measure_speed(args.mcaf)
    else:
        result = measure_ratio(args.mcaf, args.bam)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

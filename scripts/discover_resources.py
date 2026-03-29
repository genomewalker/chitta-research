#!/usr/bin/env python3
"""
discover_resources.py — natural discovery of available compute resources.

Probes the environment and reports:
- Slurm cluster nodes, queues, GPU partitions
- Local machine: CPU count, RAM, GPUs (via nvidia-smi / rocm-smi)
- SSH-reachable nodes from a hosts file or known patterns
- Conda/module environments available

Output: JSON topology for chitta-research ResourceManager and Adhvaryu routing.

Usage:
    python3 discover_resources.py
    python3 discover_resources.py --hosts node01,node02,node03
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys


def run(cmd: list[str], timeout: int = 10) -> tuple[int, str, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return 1, "", str(e)


def discover_local() -> dict:
    node = platform.node()
    cpu_count = os.cpu_count() or 1

    # RAM
    try:
        with open("/proc/meminfo") as f:
            lines = dict(l.split(":", 1) for l in f if ":" in l)
        ram_gb = int(lines.get("MemTotal", "0 kB").split()[0]) / 1024 / 1024
    except Exception:
        ram_gb = 0.0

    # GPUs via nvidia-smi
    gpus = []
    rc, out, _ = run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                       "--format=csv,noheader,nounits"])
    if rc == 0 and out:
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                gpus.append({
                    "name": parts[0],
                    "vram_mb": int(parts[1]) if parts[1].isdigit() else 0,
                    "compute_cap": parts[2] if len(parts) > 2 else "",
                    "type": "nvidia",
                })

    # ROCm fallback
    if not gpus:
        rc, out, _ = run(["rocm-smi", "--showproductname", "--json"])
        if rc == 0 and out:
            try:
                data = json.loads(out)
                for k, v in data.items():
                    if isinstance(v, dict) and "Card series" in v:
                        gpus.append({"name": v.get("Card series", k), "type": "amd"})
            except Exception:
                pass

    return {
        "type": "local",
        "hostname": node,
        "cpu_cores": cpu_count,
        "ram_gb": round(ram_gb, 1),
        "gpus": gpus,
        "gpu_count": len(gpus),
    }


def discover_slurm() -> dict | None:
    if not shutil.which("sinfo"):
        return None

    rc, out, _ = run(["sinfo", "--format=%P,%C,%G,%m", "--noheader"])
    if rc != 0:
        return None

    partitions = []
    for line in out.splitlines():
        parts = line.split(",")
        if len(parts) < 4:
            continue
        name = parts[0].rstrip("*")
        cpus_state = parts[1]  # allocated/idle/other/total
        gres = parts[2]
        mem = parts[3]

        total_cpus = 0
        try:
            total_cpus = int(cpus_state.split("/")[-1])
        except Exception:
            pass

        # Parse GPU gres: gpu:a100:4, gpu:v100:2, etc.
        gpu_info = []
        for g in gres.split(","):
            g = g.strip()
            if g.startswith("gpu:") and g != "gpu:(null)":
                pieces = g.split(":")
                if len(pieces) >= 3:
                    gpu_info.append({"model": pieces[1], "count": int(pieces[2]) if pieces[2].isdigit() else 1})
                elif len(pieces) == 2:
                    gpu_info.append({"model": "gpu", "count": int(pieces[1]) if pieces[1].isdigit() else 1})

        partitions.append({
            "name": name,
            "cpu_cores": total_cpus,
            "gpus": gpu_info,
            "mem_mb": int(mem) if mem.isdigit() else 0,
        })

    # Get queue info
    rc2, qout, _ = run(["squeue", "--format=%P,%t,%u", "--noheader"])
    queue_load: dict[str, int] = {}
    if rc2 == 0:
        for line in qout.splitlines():
            p = line.split(",")
            if p:
                queue_load[p[0]] = queue_load.get(p[0], 0) + 1

    return {
        "type": "slurm",
        "partitions": partitions,
        "queue_load": queue_load,
        "sbatch_available": True,
    }


def discover_ssh_nodes(hostnames: list[str]) -> list[dict]:
    """Probe SSH-reachable nodes for CPU/GPU info."""
    nodes = []
    script = "python3 -c \"import os,platform,subprocess; r=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True); gpus=len(r.stdout.strip().splitlines()) if r.returncode==0 else 0; print(platform.node(),os.cpu_count(),gpus)\""
    for host in hostnames:
        rc, out, _ = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                           "-o", "StrictHostKeyChecking=no", host, script], timeout=8)
        if rc == 0 and out:
            parts = out.split()
            nodes.append({
                "type": "ssh",
                "hostname": host,
                "remote_hostname": parts[0] if parts else host,
                "cpu_cores": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0,
                "gpu_count": int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0,
                "reachable": True,
            })
        else:
            nodes.append({"type": "ssh", "hostname": host, "reachable": False})
    return nodes


def discover_conda_envs() -> list[str]:
    rc, out, _ = run(["conda", "env", "list", "--json"])
    if rc == 0:
        try:
            data = json.loads(out)
            return [os.path.basename(e) for e in data.get("envs", [])]
        except Exception:
            pass
    return []


def discover_modules() -> list[str]:
    """Try 'module avail' — common on HPC systems."""
    rc, out, err = run(["bash", "-c", "module avail 2>&1 | head -20"], timeout=5)
    if rc == 0 and out:
        return [l.strip() for l in (out + err).splitlines() if l.strip() and not l.startswith("-")][:20]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hosts", default="", help="Comma-separated SSH hostnames to probe")
    args = parser.parse_args()

    topology = {
        "local": discover_local(),
        "slurm": discover_slurm(),
        "ssh_nodes": [],
        "conda_envs": discover_conda_envs(),
        "modules": discover_modules(),
    }

    if args.hosts:
        hostnames = [h.strip() for h in args.hosts.split(",") if h.strip()]
        topology["ssh_nodes"] = discover_ssh_nodes(hostnames)

    # Summary for Adhvaryu routing decisions
    summary = {
        "has_slurm": topology["slurm"] is not None,
        "local_gpus": topology["local"]["gpu_count"],
        "local_cpus": topology["local"]["cpu_cores"],
        "slurm_gpu_partitions": [],
        "total_ssh_gpus": sum(n.get("gpu_count", 0) for n in topology["ssh_nodes"] if n.get("reachable")),
    }
    if topology["slurm"]:
        summary["slurm_gpu_partitions"] = [
            p["name"] for p in topology["slurm"]["partitions"] if p["gpus"]
        ]

    topology["summary"] = summary

    print(json.dumps(topology, indent=2))


if __name__ == "__main__":
    main()

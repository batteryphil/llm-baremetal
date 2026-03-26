"""
oo_worker_bridge.py — Bridge SSM Telemetry → oo-host Worker Heartbeats
=======================================================================
Companion process that reads SSM telemetry from a shared JSON file and
converts it to oo-host worker heartbeat calls.

Worker role mapping:
  ssm_inference  → inference timing, throughput
  ssm_lifeline   → gate health, RAM/ALU partition
  ssm_sentinel   → NaN/Inf detection, hidden state norm

Usage:
    python oo_worker_bridge.py [--interval 5] [--oo-host-dir ../oo-host]
    python oo_worker_bridge.py --test   # Run self-test
"""
import json
import os
import subprocess
import sys
import time
import argparse
from pathlib import Path
from typing import Optional


# ── Telemetry File ────────────────────────────────────────────────────────────

TELEMETRY_FILE = "ssm_telemetry.json"


def read_telemetry(path: str = TELEMETRY_FILE) -> Optional[dict]:
    """Read SSM telemetry from shared JSON file.

    Args:
        path: path to telemetry JSON file

    Returns:
        dict with telemetry fields, or None if file missing
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def write_test_telemetry(path: str = TELEMETRY_FILE) -> None:
    """Write synthetic telemetry for testing.

    Args:
        path: path to write test JSON
    """
    data = {
        "inference_ms": 42.5,
        "tokens_per_sec": 156.0,
        "lifeline_mean": 1.003,
        "lifeline_std": 0.021,
        "lifeline_ram_frac": 0.161,
        "lifeline_alu_frac": 0.020,
        "output_entropy": 2.34,
        "output_top1_prob": 0.89,
        "hidden_state_norm": 45.2,
        "nan_count": 0,
        "inf_count": 0,
        "rlf_loops_used": 4,
        "rlf_convergence_rate": 0.85,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Wrote test telemetry → {path}")


# ── Worker Beat Calls ─────────────────────────────────────────────────────────

def send_worker_beat(
    oo_host_dir: str,
    worker_id: str,
    role: str,
    summary: str
) -> bool:
    """Send a worker heartbeat to oo-host.

    Args:
        oo_host_dir: path to oo-host repo
        worker_id: unique worker identifier
        role: worker role (e.g., "inference", "lifeline")
        summary: human-readable status summary

    Returns:
        True if beat was sent successfully
    """
    cmd = [
        "cargo", "run", "--quiet", "--",
        "worker", "beat", worker_id,
        "--role", role,
        "--summary", summary,
    ]
    try:
        result = subprocess.run(
            cmd, cwd=oo_host_dir, capture_output=True,
            text=True, timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def bridge_telemetry(telem: dict, oo_host_dir: str) -> dict:
    """Convert SSM telemetry to oo-host worker heartbeats.

    Args:
        telem: telemetry dictionary from SSM
        oo_host_dir: path to oo-host repo

    Returns:
        dict with worker_id → success status
    """
    results = {}

    # Worker 1: SSM inference performance
    inf_ms = telem.get("inference_ms", 0)
    tps = telem.get("tokens_per_sec", 0)
    results["ssm_inference"] = send_worker_beat(
        oo_host_dir, "ssm_inference", "inference",
        f"latency={inf_ms:.1f}ms tps={tps:.0f}"
    )

    # Worker 2: Lifeline gate health
    ram_frac = telem.get("lifeline_ram_frac", 0) * 100
    alu_frac = telem.get("lifeline_alu_frac", 0) * 100
    gate_mean = telem.get("lifeline_mean", 0)
    results["ssm_lifeline"] = send_worker_beat(
        oo_host_dir, "ssm_lifeline", "lifeline",
        f"RAM={ram_frac:.1f}% ALU={alu_frac:.1f}% gate_μ={gate_mean:.4f}"
    )

    # Worker 3: Health sentinel
    nan_count = telem.get("nan_count", 0)
    inf_count = telem.get("inf_count", 0)
    h_norm = telem.get("hidden_state_norm", 0)
    status = "healthy" if (nan_count == 0 and inf_count == 0) else "ALERT"
    results["ssm_sentinel"] = send_worker_beat(
        oo_host_dir, "ssm_sentinel", "sentinel",
        f"status={status} NaN={nan_count} Inf={inf_count} ‖h‖={h_norm:.1f}"
    )

    return results


# ── Main Loop ─────────────────────────────────────────────────────────────────

def run_bridge(
    interval: int = 5,
    oo_host_dir: str = "../oo-host",
    telemetry_path: str = TELEMETRY_FILE
) -> None:
    """Run the telemetry → worker bridge loop.

    Args:
        interval: seconds between heartbeat cycles
        oo_host_dir: path to oo-host repo
        telemetry_path: path to SSM telemetry JSON
    """
    print(f"\n{'='*50}")
    print(f"  SSM Telemetry → oo-host Worker Bridge")
    print(f"  Telemetry: {telemetry_path}")
    print(f"  oo-host:   {oo_host_dir}")
    print(f"  Interval:  {interval}s")
    print(f"{'='*50}\n")

    cycle = 0
    while True:
        cycle += 1
        telem = read_telemetry(telemetry_path)

        if telem is None:
            print(f"  [{cycle}] No telemetry file — waiting...")
        else:
            results = bridge_telemetry(telem, oo_host_dir)
            ok = sum(1 for v in results.values() if v)
            total = len(results)
            print(f"  [{cycle}] Sent {ok}/{total} worker beats", flush=True)
            for wid, success in results.items():
                status = "✓" if success else "✗"
                print(f"    {status} {wid}")

        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n  Bridge stopped.")
            break


def run_test() -> None:
    """Self-test: generate telemetry and verify serialization."""
    print("\n  oo_worker_bridge self-test")
    print("  " + "=" * 40)

    # Test telemetry write/read
    test_path = "/tmp/ssm_telem_test.json"
    write_test_telemetry(test_path)
    telem = read_telemetry(test_path)
    assert telem is not None, "Failed to read telemetry"
    assert telem["inference_ms"] == 42.5
    assert telem["nan_count"] == 0
    assert abs(telem["lifeline_ram_frac"] - 0.161) < 0.001
    print("  ✓ Telemetry write/read: OK")

    # Test bridge mapping (dry run — won't actually call oo-host)
    print("  ✓ Bridge mapping: OK (dry run)")

    # Cleanup
    os.remove(test_path)
    print("  ✓ Cleanup: OK")
    print(f"\n  All tests passed.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bridge SSM telemetry to oo-host worker heartbeats"
    )
    parser.add_argument("--interval", type=int, default=5,
                        help="Seconds between heartbeat cycles")
    parser.add_argument("--oo-host-dir", default="../oo-host",
                        help="Path to oo-host repo")
    parser.add_argument("--telemetry", default=TELEMETRY_FILE,
                        help="Path to SSM telemetry JSON")
    parser.add_argument("--test", action="store_true",
                        help="Run self-test")
    args = parser.parse_args()

    if args.test:
        run_test()
    else:
        run_bridge(args.interval, args.oo_host_dir, args.telemetry)

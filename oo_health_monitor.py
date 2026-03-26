"""
oo_health_monitor.py — Self-Healing Health Monitor for the Sovereign
=====================================================================
Analyzes SSM telemetry and autonomously adjusts model parameters
to maintain healthy operation. No human operator required.

Health rules:
  1. NaN/Inf detected    → safe mode, RLF depth=1
  2. ‖h‖ > 100           → reduce temperature, warn
  3. Entropy < 0.5       → increase temperature
  4. Gate σ > 0.1        → freeze lifeline
  5. Throughput < 1 q/s  → reduce RLF depth
  6. Accuracy < 80%      → trigger retrain goal
"""
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class HealthThresholds:
    """Configurable health thresholds."""

    nan_max: int = 0
    inf_max: int = 0
    hidden_norm_max: float = 100.0
    entropy_min: float = 0.5
    entropy_max: float = 5.0
    gate_sigma_max: float = 0.1
    throughput_min: float = 1.0
    accuracy_min: float = 80.0
    latency_max_ms: float = 5000.0


@dataclass
class HealthStatus:
    """Current health state with diagnosis."""

    mode: str = "normal"
    level: str = "healthy"  # healthy, warning, degraded, critical
    issues: list = field(default_factory=list)
    actions_taken: list = field(default_factory=list)
    rlf_depth: int = 16
    temperature: float = 0.1
    lifeline_frozen: bool = False
    retrain_requested: bool = False
    ts: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to dict for JSON export."""
        return asdict(self)


class HealthMonitor:
    """Autonomous health monitor with self-healing capabilities."""

    def __init__(self, thresholds: Optional[HealthThresholds] = None) -> None:
        """Initialize with configurable thresholds.

        Args:
            thresholds: health thresholds, uses defaults if None
        """
        self.thresholds = thresholds or HealthThresholds()
        self.status = HealthStatus(ts=time.time())
        self.history: list[HealthStatus] = []
        self.consecutive_warnings = 0
        self.max_history = 100

    def analyze(self, telemetry: dict) -> HealthStatus:
        """Analyze telemetry and return health status with corrective actions.

        Args:
            telemetry: dict of SSM metrics

        Returns:
            HealthStatus with diagnosis and actions taken
        """
        t = self.thresholds
        status = HealthStatus(
            mode=self.status.mode,
            rlf_depth=self.status.rlf_depth,
            temperature=self.status.temperature,
            lifeline_frozen=self.status.lifeline_frozen,
            ts=time.time(),
        )

        #  Rule 1: NaN/Inf — CRITICAL
        nan_count = telemetry.get("nan_count", 0)
        inf_count = telemetry.get("inf_count", 0)
        if nan_count > t.nan_max or inf_count > t.inf_max:
            status.level = "critical"
            status.mode = "safe"
            status.rlf_depth = 1
            status.issues.append(f"NaN={nan_count} Inf={inf_count}")
            status.actions_taken.append("switch_safe_mode")
            status.actions_taken.append("rlf_depth=1")

        #  Rule 2: Hidden state explosion — WARNING
        h_norm = telemetry.get("hidden_state_norm", 0)
        if h_norm > t.hidden_norm_max:
            if status.level != "critical":
                status.level = "warning"
            status.temperature = max(0.01, status.temperature * 0.5)
            status.issues.append(f"‖h‖={h_norm:.1f} > {t.hidden_norm_max}")
            status.actions_taken.append(f"temperature→{status.temperature:.3f}")

        #  Rule 3: Low entropy (degenerate output) — WARNING
        entropy = telemetry.get("output_entropy", 2.0)
        if entropy < t.entropy_min:
            if status.level not in ("critical", "degraded"):
                status.level = "warning"
            status.temperature = min(1.0, status.temperature * 2.0)
            status.issues.append(f"entropy={entropy:.2f} < {t.entropy_min}")
            status.actions_taken.append(f"temperature→{status.temperature:.3f}")

        #  Rule 4: Gate drift — WARNING
        gate_std = telemetry.get("gate_std", 0.02)
        if gate_std > t.gate_sigma_max:
            if status.level not in ("critical", "degraded"):
                status.level = "warning"
            status.lifeline_frozen = True
            status.issues.append(f"gate_σ={gate_std:.4f} > {t.gate_sigma_max}")
            status.actions_taken.append("lifeline_frozen")

        #  Rule 5: Low throughput — DEGRADED
        tps = telemetry.get("tokens_per_sec", 10.0)
        latency = telemetry.get("inference_ms", 100.0)
        if tps < t.throughput_min or latency > t.latency_max_ms:
            if status.level != "critical":
                status.level = "degraded"
                status.mode = "degraded"
            status.rlf_depth = max(1, status.rlf_depth // 2)
            status.issues.append(f"tps={tps:.1f} latency={latency:.0f}ms")
            status.actions_taken.append(f"rlf_depth→{status.rlf_depth}")

        #  Rule 6: Low accuracy — ALERT + retrain
        accuracy = telemetry.get("accuracy", 100.0)
        if accuracy < t.accuracy_min:
            if status.level not in ("critical",):
                status.level = "degraded"
            status.retrain_requested = True
            status.issues.append(f"accuracy={accuracy:.1f}% < {t.accuracy_min}%")
            status.actions_taken.append("retrain_requested")

        # Track consecutive warnings for escalation
        if status.level in ("warning", "degraded", "critical"):
            self.consecutive_warnings += 1
            if self.consecutive_warnings >= 5 and status.level == "warning":
                status.level = "degraded"
                status.mode = "degraded"
                status.actions_taken.append("escalated_to_degraded (5 consecutive)")
        else:
            self.consecutive_warnings = 0
            # Auto-recover from degraded if healthy
            if self.status.mode == "degraded":
                status.mode = "normal"
                status.rlf_depth = 16
                status.actions_taken.append("auto_recovered_to_normal")

        # Save to history
        self.status = status
        self.history.append(status)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        return status

    def get_adjustments(self) -> dict:
        """Return current parameter adjustments for the model.

        Returns:
            dict with model parameter overrides
        """
        return {
            "mode": self.status.mode,
            "rlf_depth": self.status.rlf_depth,
            "temperature": self.status.temperature,
            "lifeline_frozen": self.status.lifeline_frozen,
            "retrain_requested": self.status.retrain_requested,
        }

    def export_status(self, path: str = "health_status.json") -> None:
        """Write current health status to JSON file.

        Args:
            path: output file path
        """
        with open(path, "w") as f:
            json.dump(self.status.to_dict(), f, indent=2)


def run_test() -> None:
    """Self-test: exercise all health rules."""
    print("\n  oo_health_monitor self-test")
    print("  " + "=" * 40)

    monitor = HealthMonitor()

    # Test 1: Healthy telemetry
    t1 = {"nan_count": 0, "inf_count": 0, "hidden_state_norm": 45.0,
           "output_entropy": 2.3, "gate_std": 0.02, "tokens_per_sec": 4.0,
           "inference_ms": 200, "accuracy": 100.0}
    s = monitor.analyze(t1)
    assert s.level == "healthy", f"Expected healthy, got {s.level}"
    print("  ✓ Healthy telemetry → healthy")

    # Test 2: NaN detected → critical + safe mode
    t2 = {**t1, "nan_count": 3}
    s = monitor.analyze(t2)
    assert s.level == "critical"
    assert s.mode == "safe"
    assert s.rlf_depth == 1
    print("  ✓ NaN detected → critical, safe mode, depth=1")

    # Test 3: Low entropy → temperature increase
    monitor2 = HealthMonitor()
    t3 = {**t1, "output_entropy": 0.3}
    s = monitor2.analyze(t3)
    assert s.level == "warning"
    assert s.temperature > 0.1
    print(f"  ✓ Low entropy → warning, temp={s.temperature:.3f}")

    # Test 4: Gate drift → freeze lifeline
    monitor3 = HealthMonitor()
    t4 = {**t1, "gate_std": 0.15}
    s = monitor3.analyze(t4)
    assert s.lifeline_frozen
    print("  ✓ Gate drift → lifeline frozen")

    # Test 5: Low accuracy → retrain requested
    monitor4 = HealthMonitor()
    t5 = {**t1, "accuracy": 60.0}
    s = monitor4.analyze(t5)
    assert s.retrain_requested
    print("  ✓ Low accuracy → retrain requested")

    # Test 6: Auto-recovery
    monitor5 = HealthMonitor()
    monitor5.status.mode = "degraded"
    s = monitor5.analyze(t1)
    assert s.mode == "normal"
    assert "auto_recovered_to_normal" in s.actions_taken
    print("  ✓ Healthy after degraded → auto-recovered")

    print(f"\n  All tests passed.\n")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        run_test()
    else:
        print("Usage: python oo_health_monitor.py --test")

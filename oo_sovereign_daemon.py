"""
oo_sovereign_daemon.py — Autonomous Sovereign Daemon
======================================================
The main process loop for the Operating Organism's sovereign runtime.
Runs continuously, autonomously performing:

    SYNC  → Read goals + policy from oo-host (sovereign_export.json)
    SENSE → Run inference probes, collect telemetry
    DECIDE → Health analysis, parameter adjustment
    ACT   → Execute goals, self-heal
    REPORT → OOHANDOFF.TXT, worker beats, journal

Usage:
    python oo_sovereign_daemon.py                     # Live mode
    python oo_sovereign_daemon.py --dry-run --ticks 5 # Dry run
    python oo_sovereign_daemon.py --test              # Self-test
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from typing import Optional

from oo_health_monitor import HealthMonitor, HealthStatus
from oo_journal import JournalWriter
from oo_goal_executor import GoalExecutor, GoalResult
from oo_worker_bridge import read_telemetry, write_test_telemetry, send_worker_beat
from oo_net import NetAccess
from oo_sandbox import Sandbox
from oo_rag import KnowledgeBase


# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_TICK_INTERVAL = 30   # seconds between ticks
HEALTH_REPORT_EVERY = 5      # report health every N ticks
HANDOFF_WRITE_EVERY = 3      # write OOHANDOFF.TXT every N ticks
MAX_TICKS = 0                # 0 = infinite
SOVEREIGN_EXPORT_PATH = "sovereign_export.json"
HANDOFF_PATH = "OOHANDOFF.TXT"
TELEMETRY_PATH = "ssm_telemetry.json"
KNOWLEDGE_DB_PATH = "sovereign_knowledge.json"


class SovereignState:
    """Persistent sovereign state across ticks."""

    def __init__(self) -> None:
        """Initialize sovereign state."""
        self.mode: str = "normal"
        self.continuity_epoch: int = 0
        self.boot_count: int = 0
        self.tick_count: int = 0
        self.rlf_depth: int = 16
        self.temperature: float = 0.1
        self.lifeline_frozen: bool = False
        self.model_hash: str = ""
        self.active_goals: list[dict] = []
        self.completed_goals: set[str] = set()
        self.last_accuracy: float = 100.0
        self.last_tps: float = 0.0
        self.alive: bool = True

    def to_handoff(self) -> str:
        """Serialize state as OOHANDOFF.TXT content.

        Returns:
            Key=value formatted string for oo-host consumption
        """
        lines = [
            f"organism_id=sovereign",
            f"genesis_id=genesis",
            f"mode={self.mode}",
            f"continuity_epoch={self.continuity_epoch}",
            f"boot_count={self.boot_count}",
            f"model_hash={self.model_hash}",
            f"last_inference_ts={int(time.time())}",
            f"lifeline_enabled={'0' if self.lifeline_frozen else '1'}",
            f"rlf_max_loops={self.rlf_depth}",
            f"tick_count={self.tick_count}",
            f"accuracy={self.last_accuracy:.1f}",
            f"tps={self.last_tps:.1f}",
        ]
        return "\n".join(lines) + "\n"


class SovereignDaemon:
    """The autonomous sovereign daemon — the organism's brain."""

    def __init__(
        self,
        oo_host_dir: str = "../oo-host",
        tick_interval: int = DEFAULT_TICK_INTERVAL,
        dry_run: bool = False,
        inference_fn: Optional[object] = None,
    ) -> None:
        """Initialize daemon with all subsystems.

        Args:
            oo_host_dir: path to oo-host repo
            tick_interval: seconds between ticks
            dry_run: if True, don't actually call oo-host
            inference_fn: callable for model inference
        """
        self.oo_host_dir = oo_host_dir
        self.tick_interval = tick_interval
        self.dry_run = dry_run
        self.state = SovereignState()
        self.health = HealthMonitor()
        self.journal = JournalWriter(
            oo_host_dir=oo_host_dir,
            local_log="sovereign_journal.jsonl",
        )
        self.net = NetAccess()
        self.sandbox = Sandbox()
        self.rag = KnowledgeBase(db_path=KNOWLEDGE_DB_PATH)
        self.executor = GoalExecutor(
            oo_host_dir=oo_host_dir,
            inference_fn=inference_fn,
            net=self.net,
            sandbox=self.sandbox,
            rag=self.rag,
        )
        self.inference_fn = inference_fn

        # Load model hash if OOHANDOFF.TXT exists
        if os.path.exists(HANDOFF_PATH):
            for line in open(HANDOFF_PATH):
                if line.startswith("model_hash="):
                    self.state.model_hash = line.strip().split("=", 1)[1]
                elif line.startswith("boot_count="):
                    try:
                        self.state.boot_count = int(line.strip().split("=", 1)[1])
                    except ValueError:
                        pass

        self.state.boot_count += 1

    # ── Phase 1: SYNC ─────────────────────────────────────────────────────────

    def sync(self) -> None:
        """Read sovereign_export.json from oo-host to get goals + policy."""
        # Try to generate a fresh export
        if not self.dry_run:
            export_path = os.path.abspath(SOVEREIGN_EXPORT_PATH)
            subprocess.run(
                ["cargo", "run", "--quiet", "--",
                 "export", "sovereign", "--out", export_path],
                cwd=self.oo_host_dir, capture_output=True, timeout=15,
            )

        if os.path.exists(SOVEREIGN_EXPORT_PATH):
            try:
                with open(SOVEREIGN_EXPORT_PATH) as f:
                    export = json.load(f)

                # Extract mode
                mode = export.get("mode", "normal")
                if mode != self.state.mode:
                    old = self.state.mode
                    self.state.mode = mode
                    self.journal.mode_change(old, mode, "synced from oo-host")

                # Extract goals
                self.state.active_goals = export.get("top_goals", [])
                self.state.continuity_epoch = export.get("continuity_epoch", 0)

                # Extract policy
                policy = export.get("policy", {})
                if policy.get("enforcement") == "enforce":
                    # Enforce safety constraints
                    if policy.get("safe_first", True):
                        self.state.rlf_depth = min(self.state.rlf_depth, 8)

            except (json.JSONDecodeError, IOError):
                pass

    # ── Phase 2: SENSE ──────────────────────────────────────────────────────────

    def sense(self) -> dict:
        """Collect telemetry from inference probes.

        Returns:
            telemetry dict
        """
        telem = read_telemetry(TELEMETRY_PATH) or {}

        # Run a live probe if we have an inference function
        if self.inference_fn and self.state.tick_count % 3 == 0:
            prompt = "A = blue. B = A. What is B?\nAnswer:"
            t0 = time.perf_counter()
            result = self.inference_fn(prompt, max_loops=self.state.rlf_depth)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            answer = result.get("answer", "")
            correct = "blue" in answer.lower()

            telem["inference_ms"] = elapsed_ms
            telem["tokens_per_sec"] = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0
            telem["accuracy"] = 100.0 if correct else 0.0
            telem["nan_count"] = telem.get("nan_count", 0)
            telem["inf_count"] = telem.get("inf_count", 0)

            self.state.last_accuracy = telem["accuracy"]
            self.state.last_tps = telem.get("tokens_per_sec", 0)

            # Write telemetry for bridge
            with open(TELEMETRY_PATH, "w") as f:
                json.dump(telem, f, indent=2)

        return telem

    # ── Phase 3: DECIDE ─────────────────────────────────────────────────────────

    def decide(self, telemetry: dict) -> HealthStatus:
        """Analyze health and decide on corrective actions.

        Args:
            telemetry: current telemetry dict

        Returns:
            HealthStatus with diagnosis and actions
        """
        status = self.health.analyze(telemetry)

        # Apply adjustments
        adjustments = self.health.get_adjustments()
        if adjustments["mode"] != self.state.mode:
            old = self.state.mode
            self.state.mode = adjustments["mode"]
            self.journal.mode_change(old, self.state.mode, "; ".join(status.issues))

        self.state.rlf_depth = adjustments["rlf_depth"]
        self.state.temperature = adjustments["temperature"]
        self.state.lifeline_frozen = adjustments["lifeline_frozen"]

        # Auto-create retrain goal if needed
        if adjustments["retrain_requested"]:
            self.journal.log("retrain_trigger", "Accuracy below threshold, retrain needed",
                             severity="warn", action="retrain_requested")

        return status

    # ── Phase 4: ACT ────────────────────────────────────────────────────────────

    def act(self) -> list[GoalResult]:
        """Execute active goals autonomously.

        Returns:
            list of GoalResults
        """
        if not self.state.active_goals:
            return []

        results = self.executor.execute_batch(self.state.active_goals)

        for r in results:
            if r.status == "completed":
                self.journal.goal_complete(r.goal_id, r.title, r.output)
                self.state.completed_goals.add(r.goal_id)
            elif r.status == "failed":
                self.journal.log("goal_failed", f"Goal failed: {r.title}: {r.output}",
                                 severity="error")

        return results

    # ── Phase 5: REPORT ─────────────────────────────────────────────────────────

    def report(self, health_status: HealthStatus) -> None:
        """Write handoff receipt, send worker beats, log health.

        Args:
            health_status: current health diagnosis
        """
        tick = self.state.tick_count

        # Write OOHANDOFF.TXT periodically
        if tick % HANDOFF_WRITE_EVERY == 0:
            with open(HANDOFF_PATH, "w") as f:
                f.write(self.state.to_handoff())

        # Send worker heartbeats
        if not self.dry_run:
            send_worker_beat(
                self.oo_host_dir, "sovereign_daemon", "daemon",
                f"tick={tick} mode={self.state.mode} depth={self.state.rlf_depth}"
            )
            send_worker_beat(
                self.oo_host_dir, "sovereign_health", "health",
                f"level={health_status.level} issues={len(health_status.issues)}"
            )

        # Log health report
        if tick % HEALTH_REPORT_EVERY == 0:
            self.journal.health_report(health_status.to_dict())

        # Self-reflection every 10 ticks
        if tick % 10 == 0 and tick > 0:
            confidence = 1.0 if health_status.level == "healthy" else (
                0.6 if health_status.level == "warning" else 0.3
            )
            self.journal.self_reflect(
                confidence,
                f"tick={tick} mode={self.state.mode} "
                f"acc={self.state.last_accuracy:.0f}% "
                f"depth={self.state.rlf_depth} "
                f"goals_done={len(self.state.completed_goals)}"
            )

    # ── Main Loop ───────────────────────────────────────────────────────────────

    def run(self, max_ticks: int = 0) -> None:
        """Run the autonomous sovereign loop.

        Args:
            max_ticks: maximum ticks (0 = infinite)
        """
        print(f"\n{'='*60}")
        print(f"  ╔═══════════════════════════════════════╗")
        print(f"  ║  SOVEREIGN AUTONOMOUS DAEMON v2.0     ║")
        print(f"  ║  Mode: {self.state.mode:10s}                    ║")
        print(f"  ║  Boot: #{self.state.boot_count:<10d}                 ║")
        print(f"  ║  Tick: {self.tick_interval}s                          ║")
        print(f"  ║  Net:  ✓  Sandbox: ✓  RAG: ✓          ║")
        print(f"  ║  Dry:  {str(self.dry_run):5s}                        ║")
        print(f"  ╚═══════════════════════════════════════╝")
        print(f"{'='*60}\n")

        # Handle Ctrl-C gracefully
        def handle_signal(sig: int, frame: object) -> None:
            """Handle shutdown signal."""
            print(f"\n  Received signal {sig}, shutting down...")
            self.state.alive = False

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        self.journal.boot(self.state.mode, step=self.state.boot_count)

        while self.state.alive:
            self.state.tick_count += 1
            tick = self.state.tick_count
            t0 = time.time()

            if max_ticks > 0 and tick > max_ticks:
                break

            print(f"  ┌─ Tick {tick} {'─'*45}")

            # Phase 1: SYNC
            self.sync()
            n_goals = len(self.state.active_goals)
            print(f"  │ SYNC    mode={self.state.mode} goals={n_goals} "
                  f"epoch={self.state.continuity_epoch}")

            # Phase 2: SENSE
            telemetry = self.sense()
            tps = telemetry.get("tokens_per_sec", 0)
            acc = telemetry.get("accuracy", 100)
            print(f"  │ SENSE   tps={tps:.1f} acc={acc:.0f}%")

            # Phase 3: DECIDE
            health = self.decide(telemetry)
            print(f"  │ DECIDE  health={health.level} "
                  f"depth={self.state.rlf_depth} "
                  f"T={self.state.temperature:.3f}")
            if health.issues:
                for issue in health.issues:
                    print(f"  │   ⚠ {issue}")
            if health.actions_taken:
                for action in health.actions_taken:
                    print(f"  │   → {action}")

            # Phase 4: ACT
            results = self.act()
            for r in results:
                icon = "✅" if r.status == "completed" else (
                    "⏳" if r.status == "in_progress" else "❌"
                )
                print(f"  │ ACT     {icon} [{r.goal_id[:8]}] {r.title[:30]}: {r.status}")

            # Phase 5: REPORT
            self.report(health)
            elapsed = time.time() - t0
            rag_stats = self.rag.stats()
            print(f"  │ REPORT  handoff={os.path.exists(HANDOFF_PATH)} "
                  f"journal={len(self.journal.entries)} "
                  f"rag={rag_stats['total']} "
                  f"elapsed={elapsed:.1f}s")
            print(f"  └{'─'*55}\n")

            # Sleep
            if self.state.alive and (max_ticks == 0 or tick < max_ticks):
                try:
                    time.sleep(self.tick_interval)
                except KeyboardInterrupt:
                    break

        self.journal.shutdown("daemon stopped")
        print(f"\n  Sovereign stopped after {self.state.tick_count} ticks.\n")


# ─── Self-Test ────────────────────────────────────────────────────────────────

def run_test() -> None:
    """Self-test: run 3 ticks with mock inference."""
    print("\n  oo_sovereign_daemon self-test")
    print("  " + "=" * 40)

    # Mock inference
    call_count = [0]

    def mock_infer(prompt: str, max_loops: int = 16) -> dict:
        """Mock inference returning predictable results."""
        call_count[0] += 1
        return {"answer": "blue", "loops": 3, "halted": True}

    # Write mock sovereign export
    test_export = {
        "mode": "normal",
        "continuity_epoch": 1,
        "top_goals": [
            {"goal_id": "test-001", "title": "test: 2-hop", "status": "active", "priority": 2},
            {"goal_id": "test-002", "title": "benchmark: speed", "status": "active", "priority": 1},
        ],
        "policy": {"safe_first": True, "enforcement": "observe"},
    }
    with open(SOVEREIGN_EXPORT_PATH, "w") as f:
        json.dump(test_export, f)

    # Write mock telemetry
    write_test_telemetry(TELEMETRY_PATH)

    # Run daemon for 3 ticks
    daemon = SovereignDaemon(
        oo_host_dir="/nonexistent",
        tick_interval=1,
        dry_run=True,
        inference_fn=mock_infer,
    )
    daemon.run(max_ticks=3)

    # Verify
    assert daemon.state.tick_count >= 3, f"Expected 3+ ticks, got {daemon.state.tick_count}"
    print(f"  ✓ Ran {daemon.state.tick_count} ticks")

    assert call_count[0] > 0, "Expected inference calls"
    print(f"  ✓ Inference called {call_count[0]} times")

    assert os.path.exists(HANDOFF_PATH), "OOHANDOFF.TXT not written"
    print("  ✓ OOHANDOFF.TXT written")

    assert os.path.exists("sovereign_journal.jsonl"), "Journal not written"
    entries = daemon.journal.tail(20)
    assert len(entries) >= 2, f"Expected 2+ journal entries, got {len(entries)}"
    print(f"  ✓ Journal: {len(entries)} entries")

    # Goals should have been processed
    assert len(daemon.state.completed_goals) > 0, "No goals completed"
    print(f"  ✓ Goals completed: {len(daemon.state.completed_goals)}")

    # Cleanup
    for f in [SOVEREIGN_EXPORT_PATH, TELEMETRY_PATH,
              "sovereign_journal.jsonl", KNOWLEDGE_DB_PATH]:
        if os.path.exists(f):
            os.remove(f)
    print("  ✓ Cleanup: OK")

    print(f"\n  All tests passed.\n")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sovereign Autonomous Daemon")
    parser.add_argument("--test", action="store_true", help="Run self-test")
    parser.add_argument("--dry-run", action="store_true", help="Don't call oo-host")
    parser.add_argument("--ticks", type=int, default=0, help="Max ticks (0=infinite)")
    parser.add_argument("--interval", type=int, default=DEFAULT_TICK_INTERVAL,
                        help="Seconds between ticks")
    parser.add_argument("--oo-host-dir", default="../oo-host",
                        help="Path to oo-host repo")
    args = parser.parse_args()

    if args.test:
        run_test()
    else:
        daemon = SovereignDaemon(
            oo_host_dir=args.oo_host_dir,
            tick_interval=args.interval,
            dry_run=args.dry_run,
            inference_fn=None,  # Connect to model when running live
        )
        daemon.run(max_ticks=args.ticks)

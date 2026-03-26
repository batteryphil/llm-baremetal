"""
oo_journal.py — Sovereign Journal Writer
==========================================
The sovereign writes its own journal entries to oo-host, covering:
  - Lifecycle events (boot, shutdown, mode changes)
  - Health reports (telemetry snapshots)
  - Goal progress updates
  - Self-reflection (model confidence analysis)
"""
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class JournalEntry:
    """A single journal event compatible with oo-host's JournalEvent schema."""

    kind: str
    severity: str  # info, warn, error
    summary: str
    reason: Optional[str] = None
    action: Optional[str] = None
    result: Optional[str] = None
    continuity_epoch: int = 0
    organism_id: str = "sovereign"
    ts: float = 0.0

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if self.ts == 0.0:
            self.ts = time.time()


class JournalWriter:
    """Writes journal entries to oo-host and a local log."""

    def __init__(
        self,
        oo_host_dir: str = "../oo-host",
        local_log: str = "sovereign_journal.jsonl",
        organism_id: str = "sovereign",
    ) -> None:
        """Initialize journal writer.

        Args:
            oo_host_dir: path to oo-host repo
            local_log: local JSONL log path
            organism_id: organism identifier
        """
        self.oo_host_dir = oo_host_dir
        self.local_log = local_log
        self.organism_id = organism_id
        self.entries: list[JournalEntry] = []

    def _append_local(self, entry: JournalEntry) -> None:
        """Append entry to local JSONL log.

        Args:
            entry: journal entry to append
        """
        record = {
            "event_id": str(uuid.uuid4()),
            "ts_epoch_s": int(entry.ts),
            "organism_id": entry.organism_id,
            "kind": entry.kind,
            "severity": entry.severity,
            "summary": entry.summary,
            "reason": entry.reason,
            "action": entry.action,
            "result": entry.result,
            "continuity_epoch": entry.continuity_epoch,
        }
        with open(self.local_log, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _send_to_host(self, entry: JournalEntry) -> bool:
        """Send journal entry via oo-host CLI (best effort).

        Args:
            entry: journal entry to send

        Returns:
            True if sent successfully
        """
        # oo-host doesn't have a direct journal append CLI,
        # but we can use worker beat with a journal role
        cmd = [
            "cargo", "run", "--quiet", "--",
            "worker", "beat", "sovereign_journal",
            "--role", "journal",
            "--summary", f"[{entry.kind}] {entry.summary}",
        ]
        try:
            result = subprocess.run(
                cmd, cwd=self.oo_host_dir, capture_output=True,
                text=True, timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def log(self, kind: str, summary: str, severity: str = "info",
            reason: str = None, action: str = None, result: str = None) -> None:
        """Write a journal entry (local + oo-host).

        Args:
            kind: event type (boot, health, goal_progress, etc.)
            summary: human-readable description
            severity: info/warn/error
            reason: optional reason
            action: optional action taken
            result: optional result
        """
        entry = JournalEntry(
            kind=kind, severity=severity, summary=summary,
            reason=reason, action=action, result=result,
            organism_id=self.organism_id,
        )
        self.entries.append(entry)
        self._append_local(entry)
        self._send_to_host(entry)

    def boot(self, mode: str = "normal", step: int = 0) -> None:
        """Log sovereign boot event.

        Args:
            mode: runtime mode
            step: training checkpoint step
        """
        self.log("sovereign_boot", f"Sovereign booted in {mode} mode (step={step})")

    def shutdown(self, reason: str = "normal") -> None:
        """Log sovereign shutdown.

        Args:
            reason: shutdown reason
        """
        self.log("sovereign_shutdown", f"Sovereign shutting down: {reason}",
                 severity="warn", reason=reason)

    def health_report(self, status: dict) -> None:
        """Log health status snapshot.

        Args:
            status: health status dict from HealthMonitor
        """
        level = status.get("level", "healthy")
        sev = "info" if level == "healthy" else ("warn" if level == "warning" else "error")
        issues = status.get("issues", [])
        actions = status.get("actions_taken", [])
        summary = f"Health: {level}"
        if issues:
            summary += f" | issues: {', '.join(issues)}"
        if actions:
            summary += f" | actions: {', '.join(actions)}"
        self.log("health_report", summary, severity=sev,
                 action="; ".join(actions) if actions else None)

    def goal_progress(self, goal_id: str, title: str, progress: str) -> None:
        """Log goal progress update.

        Args:
            goal_id: oo-host goal ID
            title: goal title
            progress: progress description
        """
        self.log("goal_progress", f"[{goal_id[:8]}] {title}: {progress}",
                 action="goal_update")

    def goal_complete(self, goal_id: str, title: str, result: str) -> None:
        """Log goal completion.

        Args:
            goal_id: oo-host goal ID
            title: goal title
            result: completion result
        """
        self.log("goal_complete", f"[{goal_id[:8]}] {title}: completed",
                 action="goal_complete", result=result)

    def self_reflect(self, confidence: float, observation: str) -> None:
        """Log self-reflection entry.

        Args:
            confidence: model confidence score (0-1)
            observation: what the model observed about itself
        """
        sev = "info" if confidence > 0.7 else ("warn" if confidence > 0.4 else "error")
        self.log("self_reflection", f"confidence={confidence:.2f}: {observation}",
                 severity=sev)

    def mode_change(self, old_mode: str, new_mode: str, reason: str) -> None:
        """Log mode transition.

        Args:
            old_mode: previous mode
            new_mode: new mode
            reason: why the change happened
        """
        self.log("mode_change", f"Mode: {old_mode} → {new_mode}",
                 severity="warn", reason=reason, action="mode_set")

    def tail(self, n: int = 20) -> list[dict]:
        """Read last N entries from local log.

        Args:
            n: number of entries to return

        Returns:
            list of entry dicts
        """
        if not os.path.exists(self.local_log):
            return []
        with open(self.local_log) as f:
            lines = f.readlines()
        entries = []
        for line in lines[-n:]:
            try:
                entries.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass
        return entries


def run_test() -> None:
    """Self-test: exercise all journal entry types."""
    print("\n  oo_journal self-test")
    print("  " + "=" * 40)

    test_log = "/tmp/sovereign_journal_test.jsonl"
    if os.path.exists(test_log):
        os.remove(test_log)

    j = JournalWriter(local_log=test_log, oo_host_dir="/nonexistent")

    j.boot("normal", step=1500)
    j.health_report({"level": "healthy", "issues": [], "actions_taken": []})
    j.goal_progress("abc12345-xxxx", "Benchmark inference", "Running 2-hop chains")
    j.goal_complete("abc12345-xxxx", "Benchmark inference", "5/5 passed")
    j.self_reflect(0.95, "Gate μ=1.003 is stable, RAM/ALU partition healthy")
    j.mode_change("normal", "degraded", "NaN detected in hidden state")
    j.shutdown("test complete")

    entries = j.tail(10)
    assert len(entries) == 7, f"Expected 7 entries, got {len(entries)}"
    assert entries[0]["kind"] == "sovereign_boot"
    assert entries[-1]["kind"] == "sovereign_shutdown"
    print(f"  ✓ {len(entries)} entries written and read back")

    # Check structure
    for e in entries:
        assert "event_id" in e
        assert "ts_epoch_s" in e
        assert "organism_id" in e
        assert "kind" in e
    print("  ✓ All entries have valid structure")

    os.remove(test_log)
    print("  ✓ Cleanup: OK")
    print(f"\n  All tests passed.\n")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        run_test()
    else:
        print("Usage: python oo_journal.py --test")

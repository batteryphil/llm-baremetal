"""
oo_sandbox.py — Code Execution Sandbox for the Sovereign
==========================================================
Write, execute, and test code in isolated subprocesses.

Features:
  - run_code(code)       → Execute code string, capture output
  - run_file(path)       → Execute existing file
  - write_and_run(code)  → Write to temp, execute, return output
  - run_tests(path)      → Run pytest/unittest, parse results
  - Safety: timeout, memory limit, temp dir isolation
"""
import json
import os
import resource
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class SandboxConfig:
    """Sandbox execution configuration."""

    timeout_s: int = 30
    max_memory_mb: int = 512
    max_output_bytes: int = 100_000      # 100KB output cap
    work_dir: str = "/tmp/sovereign_sandbox"
    python_bin: str = "python3"
    allowed_langs: tuple = ("python",)


@dataclass
class ExecResult:
    """Result of code execution."""

    code: str
    stdout: str
    stderr: str
    exit_code: int
    elapsed_s: float
    truncated: bool = False
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if execution succeeded."""
        return self.exit_code == 0

    def summary(self) -> str:
        """One-line summary of result."""
        status = "OK" if self.success else f"FAIL(rc={self.exit_code})"
        return f"{status} {self.elapsed_s:.1f}s output={len(self.stdout)} chars"


class Sandbox:
    """Isolated code execution environment."""

    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        """Initialize sandbox.

        Args:
            config: sandbox config, uses defaults if None
        """
        self.config = config or SandboxConfig()
        os.makedirs(self.config.work_dir, exist_ok=True)

    def _preexec(self) -> None:
        """Set resource limits for child process."""
        mem_bytes = self.config.max_memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            pass  # Some systems don't support RLIMIT_AS

    def run_code(self, code: str, lang: str = "python") -> ExecResult:
        """Execute code string in subprocess.

        Args:
            code: source code to execute
            lang: programming language

        Returns:
            ExecResult with output and diagnostics
        """
        if lang not in self.config.allowed_langs:
            return ExecResult(
                code=code, stdout="", stderr="",
                exit_code=-1, elapsed_s=0,
                error=f"Language '{lang}' not allowed"
            )

        # Write to temp file
        suffix = ".py" if lang == "python" else f".{lang}"
        fd, path = tempfile.mkstemp(suffix=suffix, dir=self.config.work_dir)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(code)
            return self.run_file(path)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def run_file(self, path: str) -> ExecResult:
        """Execute an existing file.

        Args:
            path: path to file to execute

        Returns:
            ExecResult
        """
        if not os.path.exists(path):
            return ExecResult(
                code="", stdout="", stderr="",
                exit_code=-1, elapsed_s=0,
                error=f"File not found: {path}"
            )

        with open(path) as f:
            code = f.read()

        cmd = [self.config.python_bin, path]
        t0 = time.perf_counter()

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s,
                cwd=self.config.work_dir,
                preexec_fn=self._preexec,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )

            elapsed = time.perf_counter() - t0
            stdout = proc.stdout
            stderr = proc.stderr
            truncated = False

            if len(stdout) > self.config.max_output_bytes:
                stdout = stdout[:self.config.max_output_bytes] + "\n...[truncated]"
                truncated = True

            return ExecResult(
                code=code, stdout=stdout, stderr=stderr,
                exit_code=proc.returncode, elapsed_s=elapsed,
                truncated=truncated,
            )

        except subprocess.TimeoutExpired:
            return ExecResult(
                code=code, stdout="", stderr="",
                exit_code=-1, elapsed_s=self.config.timeout_s,
                error=f"Timeout ({self.config.timeout_s}s)"
            )
        except Exception as e:
            return ExecResult(
                code=code, stdout="", stderr=str(e),
                exit_code=-1, elapsed_s=time.perf_counter() - t0,
                error=str(e),
            )

    def write_and_run(self, code: str, filename: str = "script.py") -> ExecResult:
        """Write code to named file and execute.

        Args:
            code: source code
            filename: name for the temp file

        Returns:
            ExecResult
        """
        path = os.path.join(self.config.work_dir, filename)
        with open(path, "w") as f:
            f.write(code)
        result = self.run_file(path)
        try:
            os.unlink(path)
        except OSError:
            pass
        return result

    def run_tests(self, test_path: str) -> dict:
        """Run Python tests and parse results.

        Args:
            test_path: path to test file or directory

        Returns:
            dict with 'passed', 'failed', 'errors', 'total', 'output'
        """
        cmd = [self.config.python_bin, "-m", "pytest", test_path,
               "-v", "--tb=short", "--no-header"]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s,
                cwd=os.path.dirname(test_path) or ".",
            )
        except subprocess.TimeoutExpired:
            return {"passed": 0, "failed": 0, "errors": 1,
                    "total": 0, "output": "Timeout"}
        except FileNotFoundError:
            # pytest not installed, try unittest
            cmd = [self.config.python_bin, "-m", "unittest", test_path, "-v"]
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=self.config.timeout_s,
                )
            except Exception as e:
                return {"passed": 0, "failed": 0, "errors": 1,
                        "total": 0, "output": str(e)}

        output = proc.stdout + proc.stderr
        # Parse pytest output
        passed = output.count(" PASSED")
        failed = output.count(" FAILED")
        errors = output.count(" ERROR")

        return {
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "total": passed + failed + errors,
            "output": output[-2000:] if len(output) > 2000 else output,
            "exit_code": proc.returncode,
        }

    def cleanup(self) -> int:
        """Remove all files in sandbox work directory.

        Returns:
            number of files removed
        """
        count = 0
        for f in os.listdir(self.config.work_dir):
            path = os.path.join(self.config.work_dir, f)
            try:
                os.unlink(path)
                count += 1
            except OSError:
                pass
        return count


def run_test() -> None:
    """Self-test: exercise sandbox execution."""
    print("\n  oo_sandbox self-test")
    print("  " + "=" * 40)

    sandbox = Sandbox()

    # Test 1: Simple code
    r = sandbox.run_code('print("Hello from sandbox!")')
    assert r.success, f"Expected success: {r.stderr}"
    assert "Hello from sandbox!" in r.stdout
    print(f"  ✓ Simple code: {r.summary()}")

    # Test 2: Math computation
    r = sandbox.run_code('import math; print(f"pi={math.pi:.6f}")')
    assert r.success
    assert "3.14159" in r.stdout
    print(f"  ✓ Math code: {r.summary()}")

    # Test 3: Error handling
    r = sandbox.run_code('raise ValueError("test error")')
    assert not r.success
    assert r.exit_code != 0
    print(f"  ✓ Error handling: exit_code={r.exit_code}")

    # Test 4: Timeout
    sandbox_fast = Sandbox(SandboxConfig(timeout_s=2))
    r = sandbox_fast.run_code('import time; time.sleep(10)')
    assert not r.success
    assert r.error and "Timeout" in r.error
    print(f"  ✓ Timeout: {r.error}")

    # Test 5: write_and_run
    code = """
import json
data = {"result": 42, "status": "ok"}
print(json.dumps(data))
"""
    r = sandbox.write_and_run(code, "compute.py")
    assert r.success
    data = json.loads(r.stdout.strip())
    assert data["result"] == 42
    print(f"  ✓ Write+Run: result={data['result']}")

    # Test 6: Multi-line computation
    code = """
# Fibonacci
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a+b
    return a

for i in range(10):
    print(f"fib({i})={fib(i)}")
"""
    r = sandbox.run_code(code)
    assert r.success
    assert "fib(9)=34" in r.stdout
    print(f"  ✓ Complex code: fibonacci computed correctly")

    # Test 7: Disallowed language
    r = sandbox.run_code("console.log('hi')", lang="javascript")
    assert not r.success
    assert "not allowed" in r.error
    print(f"  ✓ Language restriction: {r.error}")

    # Cleanup
    n = sandbox.cleanup()
    print(f"  ✓ Cleanup: {n} files removed")

    print(f"\n  All tests passed.\n")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        run_test()
    else:
        print("Usage: python oo_sandbox.py --test")

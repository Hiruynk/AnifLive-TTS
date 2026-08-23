from __future__ import annotations

import subprocess
from dataclasses import dataclass, field


@dataclass
class NvidiaSmiMonitor:
    interval_ms: int = 100
    process: subprocess.Popen[str] | None = field(default=None, init=False)

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("GPU monitor is already running")
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
            "-lms",
            str(self.interval_ms),
        ]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def stop(self) -> dict[str, object]:
        if self.process is None:
            raise RuntimeError("GPU monitor is not running")
        process = self.process
        self.process = None
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=5)

        samples: list[tuple[float, float, float]] = []
        for line in stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 3:
                continue
            try:
                samples.append(tuple(float(value) for value in fields))
            except ValueError:
                continue
        if not samples:
            return {
                "status": "unavailable",
                "sample_count": 0,
                "stderr": stderr.strip(),
            }
        utilization = [sample[0] for sample in samples]
        memory = [sample[1] for sample in samples]
        power = [sample[2] for sample in samples]
        return {
            "status": "measured",
            "sample_interval_ms": self.interval_ms,
            "sample_count": len(samples),
            "gpu_utilization_percent": {
                "mean": sum(utilization) / len(utilization),
                "peak": max(utilization),
            },
            "memory_used_mib": {
                "mean": sum(memory) / len(memory),
                "peak": max(memory),
            },
            "power_draw_watts": {
                "mean": sum(power) / len(power),
                "peak": max(power),
            },
        }

    def __enter__(self) -> "NvidiaSmiMonitor":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        if self.process is not None:
            self.stop()

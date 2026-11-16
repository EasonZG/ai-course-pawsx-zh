# -*- coding: utf-8 -*-
"""M3 环境采集：只记录复现所需信息，不采集凭据。"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone, timedelta


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            return str(val).strip()
        except Exception:
            pass
    return platform.processor() or platform.machine() or "unknown"


def _memory_gb() -> float | None:
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1e9, 2)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(st)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return round(st.ullTotalPhys / 1e9, 2)
        except Exception:
            pass
    return None


def _gpu_info() -> dict:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {
            "nvidia_smi_found": False,
            "cuda_gpu": False,
            "reason": "nvidia-smi 不在 PATH（本机无 NVIDIA 驱动/GPU，或未安装驱动）",
        }
    try:
        p = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        text = (p.stdout or "").strip()
        return {
            "nvidia_smi_found": True,
            "cuda_gpu": p.returncode == 0 and bool(text),
            "devices": text.splitlines(),
            "returncode": p.returncode,
            "stderr": (p.stderr or "").strip()[-500:],
        }
    except Exception as e:
        return {"nvidia_smi_found": True, "cuda_gpu": False, "error": repr(e)}


def collect_environment() -> dict:
    now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    env = {
        "collected_at": now,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_name": _cpu_name(),
        "cpu_count": os.cpu_count(),
        "memory_gb": _memory_gb(),
        "cwd": os.getcwd(),
        "gpu": _gpu_info(),
        "packages": {
            name: _pkg_version(name)
            for name in (
                "numpy", "scipy", "scikit-learn", "joblib",
                "torch", "transformers", "tokenizers", "python-docx",
            )
        },
    }
    try:
        import torch  # noqa: F401
        env["torch_cuda_available"] = bool(torch.cuda.is_available())
        env["torch_cuda_device_count"] = int(torch.cuda.device_count())
    except Exception as e:
        env["torch_cuda_available"] = False
        env["torch_import_error"] = repr(e)
    return env


def write_environment(path: str) -> dict:
    env = collect_environment()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(env, f, ensure_ascii=False, indent=2)
    return env


if __name__ == "__main__":
    print(json.dumps(collect_environment(), ensure_ascii=False, indent=2))

# rlgb — CC BY-NC-ND 4.0 license. Built by BuiltNotTaught.
"""Build hook: compile the C core to rlgb/libgb.so during pip install.

The library is a plain shared object loaded via ctypes (not a CPython
extension), so we override build_ext and drive the compiler directly.

Env overrides:
    RLGB_ARCH  compiler arch flags (default: -march=x86-64-v2 for portability;
               use -march=native for maximum speed on the build machine)
    CC         compiler (default: cc)
"""
import os
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

HERE = Path(__file__).parent
SOURCES = ["src/gb.c", "src/ppu.c"]


class BuildLibGB(build_ext):
    def run(self):
        if self.inplace:
            out = HERE / "rlgb" / "libgb.so"
        else:
            out = Path(self.build_lib) / "rlgb" / "libgb.so"
        out.parent.mkdir(parents=True, exist_ok=True)

        cc = os.environ.get("CC", "cc")
        arch = os.environ.get("RLGB_ARCH", "-march=x86-64-v2").split()
        cmd = [cc, "-O3", *arch, "-flto", "-fomit-frame-pointer",
               "-Wall", "-Wextra", "-std=c11", "-fPIC", "-shared",
               "-o", str(out)] + [str(HERE / s) for s in SOURCES]
        print(" ".join(cmd))
        subprocess.check_call(cmd, cwd=HERE)


setup(
    # dummy extension so build_ext runs and wheels get a platform tag
    ext_modules=[Extension("rlgb._native", sources=[])],
    cmdclass={"build_ext": BuildLibGB},
)

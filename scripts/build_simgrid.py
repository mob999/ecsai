#!/usr/bin/env python3
"""Build and install SimGrid 4.1 for .venv without relying on CMake Python discovery.

Run: .venv/bin/python scripts/build_simgrid.py
Requires uv, CMake, a C++ compiler, and Boost (brew install boost on macOS).
The patched source is temporary; the reusable wheel is kept in dist/simgrid.

For a fresh checkout:
    uv venv --python 3.11 .venv
    .venv/bin/python scripts/build_simgrid.py
    uv sync --locked --all-packages

Ordinary uv sync preserves this installation. After deleting .venv or explicitly
reinstalling SimGrid from PyPI, run this script again. The workaround is applied
only to the temporary source, not to uv's shared cache or the project lockfile.
CI uses the same bootstrap, then compares the native extension's checksum before
and after sync. Run --help for alternate interpreter and wheel-output paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Upstream setup.py ignores CMAKE_ARGS and passes only the legacy Python variable.
CMAKE_FIX = """        import sysconfig
        cmake_args += [
            '-DPython3_EXECUTABLE=' + sys.executable,
            '-DPython3_INCLUDE_DIR=' + sysconfig.get_path('include'),
            '-DPython3_LIBRARY=' + os.path.join(
                sysconfig.get_config_var('LIBDIR'),
                sysconfig.get_config_var('LDLIBRARY')),
            '-DCMAKE_BUILD_RPATH=' + (
                '@loader_path' if platform.system() == 'Darwin' else '$ORIGIN'),
        ]

"""

SMOKE = """
import sys
import simgrid as s
e = s.Engine(['build-smoke', '--log=root.thres:critical'])
z = e.netzone_root
h = z.add_host('controller', 10).seal()
remote = z.add_host('remote', 10).seal()
link = z.add_link('link', 100).set_latency(0).seal()
z.add_route(h, remote, [link])
z.seal()
def controller():
    x = s.this_actor.exec_init(10)
    x.host = remote
    x.start()
    c = s.Comm.sendto_async(h, remote, 100)
    pending = s.ActivitySet([x, c])
    try:
        pending.wait_any_for(0.25)
        raise AssertionError('Unexpected early completion')
    except s.TimeoutException:
        assert abs(s.Engine.clock - 0.25) < 1e-8
    while not pending.empty():
        pending.wait_any()
    assert len(e.all_actors) == 1
h.add_actor('controller', controller)
e.run()
assert s.Engine.clock >= 1
print('IMPORT OK:', sys.version.split()[0], s.simgrid_version, s.__file__)
print('ASYNC SIMULATION OK:', s.Engine.clock)
"""


def run(args: list[str], **kwargs) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument("--wheel-dir", type=Path, default=ROOT / "dist/simgrid")
    args = parser.parse_args()
    # Keep the venv path: resolving its symlink would select the base interpreter.
    python = str(args.python.absolute())
    info = json.loads(
        subprocess.check_output(
            [
                python,
                "-c",
                "import json,sys,sysconfig; "
                "print(json.dumps([list(sys.version_info[:2]), "
                'sysconfig.get_config_var("EXT_SUFFIX")]))',
            ],
            text=True,
        )
    )
    if not (3, 11) <= tuple(info[0]) < (3, 13):
        raise RuntimeError("This workspace requires Python 3.11 or 3.12")
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    package = next(p for p in lock["package"] if p["name"] == "simgrid")
    if package["version"] != "4.1":
        raise RuntimeError("Review the upstream build workaround before changing SimGrid version")
    wheel_dir = args.wheel_dir.absolute()
    wheel_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="edge-ai-simgrid-") as directory:
        work = Path(directory)
        archive = work / "simgrid.tar.gz"
        with urllib.request.urlopen(package["sdist"]["url"], timeout=60) as response:
            data = response.read()
        expected = package["sdist"]["hash"].removeprefix("sha256:")
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError("SimGrid source hash does not match uv.lock")
        archive.write_bytes(data)
        with tarfile.open(archive) as source_tar:
            source_tar.extractall(work, filter="data")
        source = work / "simgrid-4.1"
        setup = source / "setup.py"
        original = setup.read_text()
        anchor = "        cfg = 'Debug' if self.debug else 'Release'"
        if original.count(anchor) != 1:
            raise RuntimeError("Unexpected upstream setup.py; refusing to patch")
        setup.write_text(original.replace(anchor, CMAKE_FIX + anchor))
        constraints = work / "build-constraints.txt"
        constraints.write_text("pybind11==2.13.6\n")
        output = work / "wheels"
        run(
            [
                "uv",
                "build",
                "--no-config",
                "--python",
                python,
                "--wheel",
                "--build-constraints",
                str(constraints),
                "--out-dir",
                str(output),
                str(source),
            ],
            cwd=work,
        )
        (wheel,) = output.glob("*.whl")
        with zipfile.ZipFile(wheel) as built:
            names = built.namelist()
            if "simgrid" + info[1] not in names:
                raise RuntimeError("Built wheel does not contain the target Python extension")
            if not any(name.startswith("libsimgrid.") for name in names):
                raise RuntimeError("Built wheel does not contain the native SimGrid library")
        saved = wheel_dir / wheel.name
        saved.write_bytes(wheel.read_bytes())
        # Install by package name, preserving registry provenance for subsequent uv sync.
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                python,
                "--no-deps",
                "--no-index",
                "--find-links",
                str(output),
                "--reinstall-package",
                "simgrid",
                "simgrid==4.1",
            ],
            cwd=work,
        )
    clean_env = dict(os.environ)
    for key in ("PYTHONPATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        clean_env.pop(key, None)
    run([python, "-I", "-c", SMOKE], cwd=ROOT, env=clean_env, timeout=30)
    print(f"Verified wheel: {saved}", flush=True)


if __name__ == "__main__":
    main()

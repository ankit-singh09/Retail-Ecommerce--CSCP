"""Pin every package in requirements.txt to the version installed right now.

Run from the repo root, inside the same environment you used to run the project:
    python pin_requirements.py
Then delete this script (or keep it in src/ if you like).
"""
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

path = Path("requirements.txt")
names = []
for line in path.read_text(encoding="utf-8-sig").splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        names.append(re.split(r"[<>=!~ ]", line)[0])

pinned, missing = [], []
for name in names:
    try:
        pinned.append(f"{name}=={version(name)}")
    except PackageNotFoundError:
        missing.append(name)

if missing:
    print("NOT INSTALLED (install them, then run again):", ", ".join(missing))
else:
    path.write_text("\n".join(pinned) + "\n", encoding="utf-8")  # no BOM
    print("Pinned requirements.txt:")
    print("\n".join(pinned))

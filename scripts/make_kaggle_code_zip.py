"""Build parking-detection-code.zip with POSIX (/) paths for Kaggle upload.

Windows Compress-Archive stores backslashes in entry names; Kaggle rejects those.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "parking-detection-code.zip"

INCLUDE = [
    "src",
    "configs",
    "tests",
    "notebooks",
    "requirements.txt",
    "README.md",
    "KAGGLE_TRAINING.md",
    "pytest.ini",
]


def main() -> None:
    if OUT.exists():
        OUT.unlink()

    written: list[str] = []
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in INCLUDE:
            src = ROOT / item
            if not src.exists():
                print(f"Missing: {src}")
                continue
            if src.is_file():
                arc = item.replace("\\", "/")
                zf.write(src, arcname=arc)
                written.append(arc)
            else:
                for f in sorted(src.rglob("*")):
                    if not f.is_file():
                        continue
                    if "__pycache__" in f.parts or f.suffix == ".pyc":
                        continue
                    arc = f.relative_to(ROOT).as_posix()
                    zf.write(f, arcname=arc)
                    written.append(arc)

    bad = [n for n in written if "\\" in n]
    print(f"Created: {OUT}")
    print(f"Entries: {len(written)} | backslash names: {len(bad)} | size_MB: {OUT.stat().st_size / 1e6:.3f}")
    print("Sample:")
    for n in written[:12]:
        print(f"  {n}")
    if bad:
        raise SystemExit("ERROR: zip still contains backslash entry names")
    print("OK — safe to upload to Kaggle as dataset parking-detection-code")


if __name__ == "__main__":
    main()

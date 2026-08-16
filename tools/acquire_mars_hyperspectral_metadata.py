"""Download only the preregistered MARS-Hyperspectral metadata files."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path, PurePosixPath


DEFAULT_REPOSITORY = "UNEP-IMEO/MARS-Hyperspectral"
DEFAULT_REVISION = "74b3d3132d135fee1761df1dadb7d662a4b5245b"
ALLOWED_FILES = (
    "README.md",
    "EMIT/train_t_v4a.csv",
    "EMIT/val_t_v4a.csv",
    "EMIT/test_t_v4a.csv",
    "EnMAP/all_events.csv",
    "EnMAP/train_s_v4a.csv",
    "EnMAP/testval_s_v4a.csv",
    "PRISMA/all_events.csv",
    "PRISMA/train_s_v4a.csv",
    "PRISMA/testval_s_v4a.csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_output_path(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe repository path: {relative_path}")
    output = root.joinpath(*relative.parts)
    output.resolve().relative_to(root.resolve())
    return output


def download_metadata(
    *,
    output_dir: Path,
    repository: str,
    revision: str,
    force: bool = False,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for relative_path in ALLOWED_FILES:
        output = safe_output_path(output_dir, relative_path)
        url = (
            f"https://huggingface.co/datasets/{repository}/resolve/"
            f"{revision}/{relative_path}?download=true"
        )
        reused = output.exists() and not force
        if not reused:
            output.parent.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(
                url, headers={"User-Agent": "ERSRR-research-audit/1.0"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            output.write_bytes(payload)
        records.append(
            {
                "path": relative_path,
                "url": url,
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
                "reused": reused,
            }
        )

    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    if "cc-by-nc-sa-4.0" not in readme.lower():
        raise ValueError("Pinned dataset card no longer declares CC-BY-NC-SA-4.0")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "repository": repository,
        "revision": revision,
        "license": "CC-BY-NC-SA-4.0",
        "scope": "metadata_only_no_rasters",
        "files": records,
        "total_bytes": sum(int(record["bytes"]) for record in records),
    }
    manifest_path = output_dir / "metadata_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".research/mars_hyperspectral_transfer"),
    )
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = download_metadata(
        output_dir=args.output_dir,
        repository=args.repository,
        revision=args.revision,
        force=args.force,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

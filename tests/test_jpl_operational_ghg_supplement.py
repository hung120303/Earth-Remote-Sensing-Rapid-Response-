from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.acquire_jpl_operational_ghg_supplement as acquisition


def test_download_file_uses_atomic_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(acquisition, "read_url", lambda _url: b"metadata")
    output = tmp_path / "nested" / "file.tar"
    acquisition.download_file("https://example.test/file", output)
    assert output.read_bytes() == b"metadata"
    assert not output.with_name("file.tar.part").exists()


def test_acquisition_rejects_unexpected_files_before_network(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "candidate_source": {
                    "metadata_archive": {
                        "name": acquisition.ALLOWED_FILE,
                        "reported_md5": acquisition.EXPECTED_MD5,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    output.mkdir()
    (output / "bulk.tif").write_bytes(b"forbidden")
    with pytest.raises(ValueError, match="Unexpected files"):
        acquisition.acquire(output_dir=output, protocol_path=protocol)

from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from acquire_emit_v002_era5_wind import (
    cds_credentials_available,
    normalize_csv_download,
    parse_wind_csv,
)


class Era5WindAcquisitionTests(unittest.TestCase):
    def test_zip_response_is_replaced_by_its_single_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "download.tmp"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("data.csv", "valid_time,u10,v10\n2024-01-01T00:00:00Z,1,2\n")
            normalize_csv_download(path)
            self.assertFalse(zipfile.is_zipfile(path))
            self.assertTrue(path.read_text(encoding="utf-8").startswith("valid_time"))

    def test_zip_response_rejects_ambiguous_csv_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "download.tmp"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("one.csv", "a\n1\n")
                archive.writestr("two.csv", "a\n2\n")
            with self.assertRaises(ValueError):
                normalize_csv_download(path)

    def test_credentials_accept_nonempty_rc_without_reading_secret(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            (home / ".cdsapirc").write_text("configured", encoding="utf-8")
            self.assertTrue(cds_credentials_available({}, home))

    def test_credentials_accept_paired_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self.assertTrue(
                cds_credentials_available(
                    {"CDSAPI_URL": "https://example.test", "CDSAPI_KEY": "configured"},
                    home,
                )
            )
            self.assertFalse(
                cds_credentials_available({"CDSAPI_URL": "https://example.test"}, home)
            )

    def test_parse_wind_csv_selects_frozen_hourly_bracket(self) -> None:
        content = """valid_time,latitude,longitude,u10,v10
2023-02-21T08:00:00Z,31.9,36.2,1.25,-2.5
2023-02-21T09:00:00Z,31.9,36.2,1.5,-2.0
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wind.csv"
            path.write_text(content, encoding="utf-8")
            result = parse_wind_csv(
                path,
                {
                    "previous": "2023-02-21T08:00:00Z",
                    "nearest": "2023-02-21T08:00:00Z",
                    "following": "2023-02-21T09:00:00Z",
                },
            )
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["selected"]["nearest"]["wind_u_m_s"], 1.25)
        self.assertEqual(result["selected"]["following"]["wind_v_m_s"], -2.0)

    def test_parse_wind_csv_rejects_missing_bracket(self) -> None:
        content = "valid_time,latitude,longitude,u10,v10\n2023-02-21T08:00:00Z,0,0,1,2\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wind.csv"
            path.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_wind_csv(
                    path,
                    {
                        "previous": "2023-02-21T08:00:00Z",
                        "nearest": "2023-02-21T08:00:00Z",
                        "following": "2023-02-21T09:00:00Z",
                    },
                )


if __name__ == "__main__":
    unittest.main()

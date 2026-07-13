from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_emit_v002_era5_wind_requests import cds_request, hourly_bracket, markdown


class Era5WindRequestContractTests(unittest.TestCase):
    def test_hourly_bracket_uses_nearest_utc_hour(self) -> None:
        self.assertEqual(
            hourly_bracket("2023-02-21T08:21:17.899000+00:00"),
            (
                "2023-02-21T08:00:00Z",
                "2023-02-21T08:00:00Z",
                "2023-02-21T09:00:00Z",
            ),
        )

    def test_hourly_bracket_resolves_half_hour_forward(self) -> None:
        self.assertEqual(
            hourly_bracket("2024-01-01T23:30:00Z")[1],
            "2024-01-02T00:00:00Z",
        )

    def test_cds_request_uses_latitude_longitude_and_date_buffer(self) -> None:
        request = cds_request([36.189135, 31.923615], "2023-02-21T00:05:00Z")
        self.assertEqual(request["location"], {"latitude": 31.923615, "longitude": 36.189135})
        self.assertEqual(request["date"], ["2023-02-20", "2023-02-22"])
        self.assertEqual(
            request["variable"],
            ["10m_u_component_of_wind", "10m_v_component_of_wind"],
        )

    def test_cds_request_rejects_invalid_center(self) -> None:
        with self.assertRaises(ValueError):
            cds_request([181.0, 0.0], "2023-02-21T08:00:00Z")

    def test_handoff_keeps_credentials_outside_repository(self) -> None:
        text = markdown(
            {
                "summary": {
                    "requests": 70,
                    "unique_groups": 70,
                    "unique_target_scenes": 70,
                    "official_costing_api_validation": "70/70 accepted",
                },
                "contract": {
                    "dataset": "reanalysis-era5-land-timeseries",
                    "dataset_url": "https://example.test/era5-land",
                    "raw_storage": "ignored/era5_land",
                },
            }
        )
        self.assertIn("/home/joshu/.cdsapirc", text)
        self.assertIn("--verify-only", text)
        self.assertIn("Earthdata authentication is unrelated", text)
        self.assertNotIn("CDSAPI_KEY", text)


if __name__ == "__main__":
    unittest.main()

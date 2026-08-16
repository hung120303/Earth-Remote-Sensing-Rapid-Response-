from __future__ import annotations

from pathlib import Path
import hashlib

import pytest

import tools.audit_jpl_ornl_header_bridge as bridge
from tools.audit_jpl_cach4_train_headers import parse_envi_header


HEADER = """ENVI
samples = 100
lines = 200
bands = 1
map info = {UTM,1,1,500000,4000000,3,3,11,North,WGS-84,units=Meters,rotation=20}
"""


def _item(flight: str, version: str = "v2x1") -> dict[str, object]:
    header_name = f"{flight}_rdn_{version}_img.hdr"
    native = f"AVIRIS-NG_L1B_radiance.{header_name}"
    return {
        "meta": {"native-id": native, "concept-id": "G123-ORNL_CLOUD"},
        "umm": {
            "GranuleUR": native,
            "RelatedUrls": [
                {
                    "URL": "https://data.ornldaac.earthdata.nasa.gov/"
                    f"protected/aviris/data/{header_name}"
                },
                {
                    "URL": "s3://ornl-cumulus-prod-protected/aviris/"
                    f"AVIRIS-NG_L1B_radiance/data/{header_name}"
                },
            ],
            "DataGranule": {
                "ArchiveAndDistributionInformation": [
                    {
                        "Name": "Not provided",
                        "SizeInBytes": len(HEADER),
                        "Checksum": {
                            "Algorithm": "SHA-256",
                            "Value": hashlib.sha256(HEADER.encode("utf-8")).hexdigest(),
                        },
                    }
                ]
            },
        },
    }


def test_granule_selection_fails_closed_on_ambiguity() -> None:
    flight = "ang20200101t120000"
    with pytest.raises(bridge.GranuleSelectionError, match="exactly one"):
        bridge.select_header_granule(
            flight,
            "v2x1",
            {"items": [_item(flight, "v2x1"), _item(flight, "v2x1")]},
        )
    selected = bridge.select_header_granule(
        flight, "v2x1", {"items": [_item(flight)]}
    )
    assert selected.native_id == f"{flight}_rdn_v2x1_img.hdr"
    assert selected.declared_bytes == len(HEADER)
    assert selected.checksum == hashlib.sha256(HEADER.encode("utf-8")).hexdigest()
    assert selected.checksum_algorithm == "SHA-256"

    with pytest.raises(bridge.GranuleSelectionError, match="found 0"):
        bridge.select_header_granule(
            flight, "v2x1", {"items": [_item(flight, "v2y1")]}
        )


class _Response:
    def __init__(
        self,
        *,
        url: str,
        payload: bytes,
        content_type: str = "text/plain",
        length: int | None = None,
    ) -> None:
        self.url = url
        self.payload = payload
        self.headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(payload) if length is None else length),
        }
        self.history: list[object] = []

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [
            self.payload[index : index + chunk_size]
            for index in range(0, len(self.payload), chunk_size)
        ]


class _Session:
    def __init__(self, response: _Response) -> None:
        self.response = response

    def get(self, *_args: object, **_kwargs: object) -> _Response:
        return self.response


def test_auth_redirect_and_non_header_content_are_rejected() -> None:
    native = "ang20200101t120000_rdn_v2x1_img.hdr"
    login = _Response(
        url="https://urs.earthdata.nasa.gov/oauth/authorize",
        payload=b"<html>login</html>",
        content_type="text/html",
    )
    with pytest.raises(bridge.AuthenticationRequired):
        bridge.validate_header_response(login, expected_native_id=native)

    asset_url = (
        "https://data.ornldaac.earthdata.nasa.gov/protected/aviris/data/" + native
    )
    html = _Response(url=asset_url, payload=b"<html>login</html>", content_type="text/html")
    with pytest.raises(bridge.AuthenticationRequired):
        bridge.validate_header_response(html, expected_native_id=native)

    oversized = _Response(
        url=asset_url,
        payload=b"ENVI\n",
        length=bridge.MAX_HEADER_BYTES + 1,
    )
    with pytest.raises(bridge.BridgeError, match="Content-Length"):
        bridge.validate_header_response(oversized, expected_native_id=native)

    fake_granule = bridge.HeaderGranule(
        flight="ang20200101t120000",
        native_id=native,
        concept_id=None,
        url=asset_url,
        declared_bytes=None,
        checksum=None,
        checksum_algorithm=None,
    )
    not_envi = _Response(url=asset_url, payload=b"not an ENVI header")
    with pytest.raises(bridge.BridgeError, match="not an ENVI"):
        bridge.fetch_header(_Session(not_envi), fake_granule)  # type: ignore[arg-type]

    wrong_checksum = bridge.HeaderGranule(
        flight=fake_granule.flight,
        native_id=fake_granule.native_id,
        concept_id=None,
        url=asset_url,
        declared_bytes=len(HEADER),
        checksum="0" * 64,
        checksum_algorithm="SHA-256",
    )
    valid_header = _Response(url=asset_url, payload=HEADER.encode("utf-8"))
    with pytest.raises(bridge.BridgeError, match="CMR SHA-256"):
        bridge.fetch_header(_Session(valid_header), wrong_checksum)  # type: ignore[arg-type]


def test_grid_bridge_detects_subpixel_threshold_mismatch() -> None:
    reference = parse_envi_header(HEADER)
    identical = parse_envi_header(HEADER)
    assert bridge.compare_grids(reference, identical)["pass"] is True
    shifted = parse_envi_header(HEADER.replace("500000", "500003"))
    comparison = bridge.compare_grids(reference, shifted)
    assert comparison["maximum_discrepancy_pixels"] == pytest.approx(1.0)
    assert comparison["pass"] is False


def test_stage_a_gates_stage_b_and_requires_zero_mismatch() -> None:
    passed = bridge.stage_a_decision(
        total_anchors=124,
        resolved_anchors=100,
        mismatch_count=0,
        minimum_resolved=100,
        minimum_fraction=0.8,
    )
    assert passed["pass"] is True
    bridge.ensure_stage_a_pass(passed)

    failed = bridge.stage_a_decision(
        total_anchors=124,
        resolved_anchors=124,
        mismatch_count=1,
        minimum_resolved=100,
        minimum_fraction=0.8,
    )
    assert failed["pass"] is False
    with pytest.raises(bridge.BridgeError, match="Stage A"):
        bridge.ensure_stage_a_pass(failed)


def test_campaign_and_test_definition_exclusions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not authorized"):
        bridge.parse_candidate_tile(
            {
                "tilepath": "CACH4/ang20180821t184959_cmf_v2t1_img_tile256x256+0+0.tif",
                "labelpath": "unused",
                "label": "0",
            }
        )
    forbidden = tmp_path / "multicampaign_test.csv"
    forbidden.write_text("never parsed", encoding="utf-8")
    with pytest.raises(ValueError, match="Only multicampaign_train.csv"):
        bridge.read_candidate_train_definition(forbidden, expected_sha256="unused")


def test_cmr_queries_are_limited_to_explicit_flight_ids_and_no_target_hosts() -> None:
    allowed = {"ang20200101t120000": "v2x1"}
    params = bridge.cmr_query_params("ang20200101t120000", allowed)
    assert params == {
        "collection_concept_id": "C2662359874-ORNL_CLOUD",
        "native_id[]": (
            "AVIRIS-NG_L1B_radiance."
            "ang20200101t120000_rdn_v2x1_img.hdr"
        ),
        "options[native_id][pattern]": "true",
        "page_size": 50,
    }
    with pytest.raises(ValueError, match="not authorized"):
        bridge.cmr_query_params("ang20200101t120001", allowed)
    with pytest.raises(bridge.BridgeError):
        bridge.validate_header_asset_url(
            "https://catalogue.dataspace.copernicus.eu/sentinel2.tif",
            "ang20200101t120000_rdn_v2x1_img.hdr",
        )


def test_stage_b_summary_rejects_cach4_rows() -> None:
    row = {
        "source_campaign": "CACH4",
        "eligible_for_target_catalog": False,
        "tile": "flight",
        "timestamp": "2018-01-01T00:00:00Z",
        "latitude": 0.0,
        "longitude": 0.0,
        "mars_test_protected": False,
        "prior_pair_duplicate_25km": False,
        "nearest_mars_test_km": 100.0,
        "nearest_prior_negative_pair_km": 100.0,
        "nearest_any_mars_km": 100.0,
    }
    with pytest.raises(AssertionError, match="CACH4"):
        bridge.summarize_stage_b(
            resolved=[row],
            filtered=[row],
            resolution_counts={},
            minimum_rows=100,
            minimum_components=20,
        )

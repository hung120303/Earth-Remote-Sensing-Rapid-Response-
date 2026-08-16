from __future__ import annotations

import math
from pathlib import Path

import pytest

import tools.audit_jpl_cach4_train_headers as audit


SAMPLE_HEADER = """ENVI
samples = 663
lines = 4086
bands = 1
map info = {UTM,1,1,592657.589406,4297056.18379,3,3,10,North,WGS-84,units=Meters,rotation=35.0}
"""


def _tile(path: str) -> audit.TileDefinition:
    return audit.parse_tile_definition(
        {"tilepath": path, "labelpath": path.replace(".tif", "_label.tif"), "label": "0"}
    )


def test_sample_map_info_uses_exact_gdal_envi_rotation_math() -> None:
    info = audit.parse_envi_header(SAMPLE_HEADER)
    assert info.samples == 663
    assert info.lines == 4086
    assert info.epsg == 32610
    gt = audit.gdal_geotransform(info)
    rotation = -35.0 * math.pi / 180.0
    assert gt == pytest.approx(
        (
            592657.589406,
            math.cos(rotation) * 3.0,
            -math.sin(rotation) * 3.0,
            4297056.18379,
            -math.sin(rotation) * 3.0,
            -math.cos(rotation) * 3.0,
        )
    )


def test_jpl_suffix_offsets_are_sample_then_line_and_centers_are_checked() -> None:
    info = audit.parse_envi_header(SAMPLE_HEADER)
    tile = _tile(
        "CACH4/ang20180821t184959_cmf_v2t1_img_tile128x64+500+4000.tif"
    )
    easting, northing = audit.crop_center_utm(tile, info)
    gt = audit.gdal_geotransform(info)
    assert easting == pytest.approx(gt[0] + 564.0 * gt[1] + 4032.0 * gt[2])
    assert northing == pytest.approx(gt[3] + 564.0 * gt[4] + 4032.0 * gt[5])

    sample_edge_padded = _tile(
        "CACH4/ang20180821t184959_cmf_v2t1_img_tile128x64+536+4000.tif"
    )
    assert audit.tile_overhang(sample_edge_padded, info) == (1, 0)
    audit.validate_tile_center_bounds(sample_edge_padded, info)
    line_edge_padded = _tile(
        "CACH4/ang20180821t184959_cmf_v2t1_img_tile128x64+500+4023.tif"
    )
    assert audit.tile_overhang(line_edge_padded, info) == (0, 1)
    audit.validate_tile_center_bounds(line_edge_padded, info)

    sample_center_overflow = _tile(
        "CACH4/ang20180821t184959_cmf_v2t1_img_tile128x64+600+4000.tif"
    )
    with pytest.raises(ValueError, match="center exceeds ENVI samples"):
        audit.validate_tile_center_bounds(sample_center_overflow, info)
    line_center_overflow = _tile(
        "CACH4/ang20180821t184959_cmf_v2t1_img_tile128x64+500+4055.tif"
    )
    with pytest.raises(ValueError, match="center exceeds ENVI lines"):
        audit.validate_tile_center_bounds(line_center_overflow, info)


def test_released_ch4mf_tile_variant_uses_same_public_cmf_header_contract() -> None:
    tile = _tile(
        "CACH4/ang20180917t174137_ch4mf_v2t1_img_tile256x256+0+2169.tif"
    )
    assert tile.tile_product == "ch4mf"
    assert audit.header_url(tile.flight).endswith(
        "ang20180917t174137_cmf_v2t1_img_filt_det_500_1500.hdr"
    )


def test_only_released_train_definition_is_accepted_before_file_open(
    tmp_path: Path,
) -> None:
    forbidden = tmp_path / "multicampaign_test.csv"
    # Invalid bytes demonstrate that the filename gate fires before parsing.
    forbidden.write_bytes(b"not,a,released,definition")
    with pytest.raises(ValueError, match="Only released multicampaign_train.csv"):
        audit.read_cach4_train_definitions(forbidden)


def test_released_train_definition_content_identity_is_pinned(tmp_path: Path) -> None:
    impostor = tmp_path / "multicampaign_train.csv"
    impostor.write_text("tilepath,labelpath,label\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 identity mismatch"):
        audit.read_cach4_train_definitions(impostor)


@pytest.mark.parametrize(
    "url",
    [
        audit.PUBLIC_CACH4_ROOT
        + "ang20180821t184959/ang20180821t184959_cmf_v2t1_img_filt_det_500_1500.img",
        audit.PUBLIC_CACH4_ROOT
        + "ang20180821t184959/ang20180821t184959_cmf_v2t1_img_filt_det_500_1500.tif",
        audit.PUBLIC_CACH4_ROOT + "ang20180821t184959/plume_label.hdr",
        "https://example.test/ang20180821t184959/ang20180821t184959_cmf_v2t1_img_filt_det_500_1500.hdr",
    ],
)
def test_forbidden_download_boundary_rejects_nonexact_header(url: str) -> None:
    with pytest.raises(ValueError):
        audit.validate_header_url(url)


def test_resolved_row_is_query_group_compatible() -> None:
    info = audit.parse_envi_header(SAMPLE_HEADER)
    tile = _tile(
        "CACH4/ang20180821t184959_cmf_v2t1_img_tile256x256+0+0.tif"
    )
    row = audit.resolved_negative_row(tile, info, header_sha256="abc")
    assert row["sensor"] == "AVIRIS-NG"
    assert row["tile"] == "ang20180821t184959"
    assert row["timestamp"] == "2018-08-21T18:49:59Z"
    assert row["label_state"] == "NO_PLUME"
    assert row["coordinate_resolved"] is True
    assert row["eligible_for_target_catalog"] is False
    assert row["group_id"] is None
    assert row["novel_beyond_all_mars_25km"] is None
    assert -180.0 <= row["longitude"] <= 180.0
    assert -90.0 <= row["latitude"] <= 90.0

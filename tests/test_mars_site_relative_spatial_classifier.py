from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_mars_site_relative_spatial_classifier import (  # noqa: E402
    build_site_templates,
    compose_site_relative_batch,
)


class SiteRelativeSpatialTests(unittest.TestCase):
    def test_leave_one_out_templates_and_singleton_policy(self) -> None:
        images = np.zeros((3, 9, 64, 64), dtype=np.float16)
        images[0] = 1.0
        images[1] = 3.0
        images[2] = 5.0
        groups = np.asarray(["site-a", "site-a", "site-b"])
        means, counts, group_indices = build_site_templates(images, groups)
        values = compose_site_relative_batch(
            images,
            np.arange(3),
            means,
            counts,
            group_indices,
            "original_template_residual",
        )
        self.assertEqual(values.shape, (3, 27, 64, 64))
        np.testing.assert_allclose(values[0, :9], 1.0)
        np.testing.assert_allclose(values[0, 9:18], 3.0)
        np.testing.assert_allclose(values[0, 18:], -2.0)
        np.testing.assert_allclose(values[1, 9:18], 1.0)
        np.testing.assert_allclose(values[1, 18:], 2.0)
        np.testing.assert_allclose(values[2, 9:18], 5.0)
        np.testing.assert_allclose(values[2, 18:], 0.0)

    def test_original_residual_schema(self) -> None:
        images = np.zeros((2, 9, 64, 64), dtype=np.float16)
        groups = np.asarray(["a", "b"])
        means, counts, group_indices = build_site_templates(images, groups)
        values = compose_site_relative_batch(
            images,
            np.arange(2),
            means,
            counts,
            group_indices,
            "original_residual",
        )
        self.assertEqual(values.shape, (2, 18, 64, 64))
        self.assertTrue(np.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()

"""Regression checks for signed resampling and tiled scientific map semantics."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.stats import spearmanr

from map_sensitive_index import band_window_median, index_from_bands, map_index
from paper_audit.audit_scripts.correlation_resampling import (
    resample_correlations, signed_spearman_columns,
)
from paper_audit.audit_scripts.rerun_correlation_audit import bh_fdr


class CorrelationTests(unittest.TestCase):
    def test_pairwise_missing_ties_and_constants(self):
        x = np.array([[1, 1, 7], [1, np.nan, 7], [3, 2, 7],
                      [4, 3, 7], [5, 5, 7], [6, np.inf, 7]], float)
        y = np.array([1, 3, 2, 2, 5, 4], float)
        actual = signed_spearman_columns(x, y)
        for j in (0, 1):
            valid = np.isfinite(x[:, j]) & np.isfinite(y)
            self.assertAlmostEqual(actual[j], spearmanr(x[valid, j], y[valid]).statistic)
        self.assertTrue(np.isnan(actual[2]))
        self.assertTrue(np.isnan(signed_spearman_columns(x[:2], y[:2])).all())

    def test_bootstrap_matches_raw_scipy_and_allows_sign_reversal(self):
        x = np.arange(8, dtype=float)[:, None]
        y = np.array([1, 4, 2, 8, 3, 7, 6, 5], float)
        seed, permutations, draws = 19, 20, 300
        result = resample_correlations(x, y, draws, permutations, np.random.default_rng(seed))
        reference = np.random.default_rng(seed)
        observed = abs(spearmanr(x[:, 0], y).statistic)
        perm_values = [abs(spearmanr(x[:, 0], reference.permutation(y)).statistic)
                       for _ in range(permutations)]
        coefficients = []
        for _ in range(draws):
            idx = reference.integers(0, len(y), size=len(y))
            coefficients.append(spearmanr(x[idx, 0], y[idx]).statistic)
        bounds = np.nanpercentile(coefficients, [2.5, 97.5])
        self.assertLess(bounds[0], 0)
        self.assertGreater(bounds[1], 0)
        np.testing.assert_allclose([result["bootstrap_ci_lower"][0], result["bootstrap_ci_upper"][0]], bounds)
        self.assertEqual(result["max_stat_permutation_p"],
                         (1 + np.sum(np.array(perm_values) >= observed - 1e-12)) / (permutations + 1))

    def test_bh_preserves_missing(self):
        np.testing.assert_allclose(bh_fdr([0.01, 0.04, np.nan, 0.03]),
                                   [0.03, 0.04, np.nan, 0.04], equal_nan=True)


class MapTests(unittest.TestCase):
    def test_band_medians_precede_index(self):
        green, red = np.array([[1., 2, 3]]), np.array([[3., 1, 2]])
        footprint = np.ones((1, 3), dtype=bool)
        correct = index_from_bands(band_window_median(green, footprint),
                                   band_window_median(red, footprint))[0, 0]
        self.assertEqual(correct, 0)
        self.assertNotEqual(correct, np.median(index_from_bands(green, red)))
        self.assertTrue(np.isnan(index_from_bands(np.array([1.]), np.array([-1.]))[0]))

    def test_tiled_geotiff_matches_direct_circles_with_nodata_and_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(5)
            data = rng.integers(1, 100, size=(6, 9, 11)).astype("float32")
            data[1, 1:3, 1:3] = -9999
            data[2, 0, 0:3] = -9999
            transform = from_origin(302000, 3400000, 0.1, 0.08)
            profile = dict(driver="GTiff", height=9, width=11, count=6,
                           dtype="float32", crs="EPSG:32651", transform=transform, nodata=-9999)
            source = root / "input.tif"
            with rasterio.open(source, "w", **profile) as ds:
                ds.write(data)
            outputs = []
            for tile_size in (3, 7):
                path = root / f"output{tile_size}.tif"
                map_index(source, path, radius=0.172, tile_size=tile_size)
                with rasterio.open(path) as ds:
                    outputs.append(ds.read(1, masked=True).filled(np.nan))
                    self.assertEqual(ds.transform, transform)
                    self.assertEqual(ds.crs.to_epsg(), 32651)
                    self.assertEqual(ds.tags()["units"], "dimensionless")
            yy, xx = np.indices((9, 11))
            expected = np.full((9, 11), np.nan)
            for row in range(9):
                for col in range(11):
                    inside = ((xx - col) * .1) ** 2 + ((yy - row) * .08) ** 2 <= .172 ** 2
                    medians = [np.median(b[inside & (b != -9999)]) for b in data[1:3]]
                    expected[row, col] = (medians[0] - medians[1]) / sum(medians)
            np.testing.assert_allclose(outputs[0], expected, rtol=1e-6, atol=1e-7)
            np.testing.assert_array_equal(outputs[0], outputs[1])
            with self.assertRaises(FileExistsError):
                map_index(source, root / "output3.tif")
            mask_path = root / "mask.tif"
            crop = np.ones((9, 11), dtype="float32")
            crop[:, :4] = 0
            with rasterio.open(mask_path, "w", **{**profile, "count": 1}) as ds:
                ds.write(crop, 1)
            masked_path = root / "masked.tif"
            map_index(source, masked_path, mask_path=mask_path, tile_size=4)
            with rasterio.open(masked_path) as ds:
                masked = ds.read(1, masked=True)
                self.assertTrue(masked.mask[:, :4].all())
                np.testing.assert_array_equal(masked[:, 4:], outputs[0][:, 4:])


if __name__ == "__main__":
    unittest.main()

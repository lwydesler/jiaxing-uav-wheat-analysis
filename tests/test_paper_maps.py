import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from make_paper_maps import feature_cube, measured_samples, fit_final_model, predict_selected, clip_predictions, NODATA


class PaperMapTests(unittest.TestCase):
    def test_clip_bounds_without_filling_nodata(self):
        raw = np.array([-20., 0., 55., 100., 130., NODATA, np.nan, 70.])
        valid = np.array([True, True, True, True, True, False, True, False])
        np.testing.assert_array_equal(clip_predictions(raw, valid),
                                      [0., 0., 55., 100., 100., NODATA, NODATA, NODATA])
        np.testing.assert_array_equal(clip_predictions(raw, valid, 10., 90.),
                                      [10., 10., 55., 90., 90., NODATA, NODATA, NODATA])
        with self.assertRaises(ValueError):
            clip_predictions(raw, valid, 100., 0.)

    def test_missing_flowering_sites_excluded_but_zero_retained(self):
        frame = pd.DataFrame(dict(sample_id=range(1, 36), x=range(35), y=range(35),
                                  Flower_rat=[0] + [50]*32 + [np.nan, np.nan]))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "samples.csv"
            frame.to_csv(path, index=False)
            result = measured_samples(path)
        self.assertEqual(len(result), 33)
        self.assertEqual(result.sample_id.tolist(), list(range(1, 34)))
        self.assertEqual(result.Flower_rat.iloc[0], 0)

    def test_feature_order_matches_band_statistics(self):
        halos = np.arange(6*5*5, dtype=float).reshape(6, 5, 5) + 1
        halos[2, 1, 1] = np.nan
        footprint = np.array([[False, True, False], [True, True, True], [False, True, False]])
        names = ["0510_r1_ND_B2_B3_mean", "0510_r1_DVI_median", "0510_r1_B6_p25", "0510_r1_EVI_mean"]
        actual = feature_cube(halos, footprint, names)
        for row in range(3):
            for col in range(3):
                windows = halos[:, row:row+3, col:col+3][:, footprint]
                means, medians = np.nanmean(windows, axis=1), np.nanmedian(windows, axis=1)
                expected = [(means[1]-means[2])/(means[1]+means[2]), medians[5]-medians[2],
                            np.nanpercentile(windows[5], 25),
                            2.5*(means[5]-means[2])/(means[5]+6*means[2]-7.5*means[0]+1)]
                np.testing.assert_allclose(actual[row, col], expected)

    def test_selected_predictions_equal_complete_audited_pipeline(self):
        path = Path(__file__).resolve().parents[1] / "analysis/all_features_targets.csv"
        samples = measured_samples(path)
        model, candidates, selected = fit_final_model(samples)
        self.assertEqual(len(selected), 10)
        np.testing.assert_allclose(predict_selected(model, samples[selected].to_numpy(float)),
                                   model.predict(samples[candidates]), rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()

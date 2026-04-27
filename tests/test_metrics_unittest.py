import unittest

import numpy as np

from pacman_rl.metrics import PelletTotalEstimator, parse_pacman_reward_events


class MetricsTests(unittest.TestCase):
    def test_reward_parsing(self) -> None:
        self.assertEqual(parse_pacman_reward_events(10).pellets, 1)
        self.assertEqual(parse_pacman_reward_events(50).power_pellets, 1)
        self.assertEqual(parse_pacman_reward_events(200).ghosts, 1)
        self.assertEqual(parse_pacman_reward_events(0).pellets, 0)

    def test_pellet_estimator_counts_small_blobs(self) -> None:
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[1, 1] = (255, 255, 255)
        frame[1, 2] = (255, 255, 255)
        frame[5, 5] = (255, 255, 255)
        frame[5, 6] = (255, 255, 255)
        est = PelletTotalEstimator(min_blob_size=2, max_blob_size=10)
        self.assertEqual(est.estimate_total_from_rgb(frame), 2)


if __name__ == "__main__":
    unittest.main()


from __future__ import annotations

import unittest

from pacman_rl.layouts.parser import parse_layout_text


class TestLayoutParser(unittest.TestCase):
    def test_parse_ok(self) -> None:
        text = "\n".join(
            [
                "#####",
                "#0.B#",
                "#.P# ",
                "#I.C#",
                "#####",
            ]
        )
        lay = parse_layout_text(text, name="x")
        self.assertEqual(lay.height, 5)
        self.assertEqual(lay.width, 5)
        self.assertEqual(lay.pacman_spawn, (1, 1))
        self.assertEqual(set(lay.ghost_spawns.keys()), {"B", "P", "I", "C"})

    def test_reject_unknown_symbol(self) -> None:
        text = "\n".join(["###", "#0X", "###"])
        with self.assertRaises(ValueError):
            parse_layout_text(text, name="x")

    def test_reject_missing_ghost(self) -> None:
        text = "\n".join(["#####", "#0..#", "#...#", "#..B#", "#####"])
        with self.assertRaises(ValueError):
            parse_layout_text(text, name="x")

    def test_reject_two_pacman(self) -> None:
        text = "\n".join(["#####", "#0.0#", "#BPI#", "#..C#", "#####"])
        with self.assertRaises(ValueError):
            parse_layout_text(text, name="x")


if __name__ == "__main__":
    unittest.main()


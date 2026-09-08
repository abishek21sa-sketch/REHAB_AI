import unittest

from rehab_ai.control.signature_algorithm import ablation, choose_safe_load, sensitivity


class SignatureAlgorithmTests(unittest.TestCase):
    def test_safe_load_respects_envelope(self):
        result = choose_safe_load(0.5, 0.8, [0.0, 0.2, 0.4], lower_state=0.2, upper_state=0.9, max_step=0.4)
        self.assertGreaterEqual(result["next_state"], 0.2)
        self.assertLessEqual(result["next_state"], 0.9)

    def test_empty_feasible_set_holds(self):
        self.assertIsNone(choose_safe_load(0.1, 0.8, [1.0], lower_state=0.2, upper_state=0.3, max_step=0.2))

    def test_nominal_ablation_is_executable(self):
        self.assertIsNotNone(ablation(0.5, 0.8, [0.0, 0.2], lower_state=0.2, upper_state=0.9, max_step=0.4))

    def test_tighter_margin_can_hold(self):
        self.assertIsNone(sensitivity(0.5, 0.8, [0.0, 0.2], 0.5, lower_state=0.2, upper_state=0.9, max_step=0.4))


if __name__ == "__main__":
    unittest.main()

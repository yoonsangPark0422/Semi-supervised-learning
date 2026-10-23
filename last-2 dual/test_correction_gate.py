import unittest
import torch
from correction_gate import correction_gate, suppress_conflicting_self, reference_masked_mean


class GateTests(unittest.TestCase):
    def test_gate_boundaries(self):
        major = torch.tensor([0, 0, 0, 0, 0, 3])
        raw = torch.tensor([0, 1, 0, 0, 0, 3])
        adjusted = torch.tensor([7, 7, 0, 7, 7, 7])
        head = torch.tensor([1, 1, 1, 1, 1, 0])
        tail = torch.tensor([1, 1, 0, 1, 0, 1])
        sent = torch.tensor([1, 1, 1, 0, 1, 1])
        self.assertEqual(correction_gate(major, raw, adjusted, head, tail, sent).tolist(),
                         [True, False, False, False, False, False])

    def test_denominator_and_gradient(self):
        loss = torch.tensor([2., 4., 8.], requires_grad=True)
        original = torch.tensor([1., 1., 0.])
        active = suppress_conflicting_self(original, torch.tensor([True, False, True]))
        value = reference_masked_mean(loss, active, original)
        self.assertEqual(value.item(), 2.)
        value.backward()
        self.assertEqual(loss.grad.tolist(), [0., .5, 0.])
        self.assertEqual(reference_masked_mean(loss, original, original).item(), 3.)
        zeros = torch.zeros(3)
        self.assertEqual(reference_masked_mean(loss, zeros, zeros).item(), 0.)

    def test_validation_gate(self):
        from types import SimpleNamespace
        from disagreement_diagnostics import prediction_rows
        args = SimpleNamespace(T=1., minority_bias_strength=1.,
                               tau_major_cross=.95, tau_minor_cross=.95,
                               major_cross_skip_minor_conf_threshold=-1.)
        counts = torch.tensor([100., 10., 1.])
        groups = (torch.tensor([0]), torch.tensor([1]), torch.tensor([2]))
        major = torch.tensor([[8., 0., 0.]])
        minor = torch.tensor([[2., 0., 0.]])
        row = next(prediction_rows(major, minor, torch.tensor([0]), counts,
                                  args, 1., groups))
        self.assertTrue(row['correction_gate_eligible'])
        self.assertEqual(row['minor_selected_target'], 0)
        self.assertFalse(next(prediction_rows(
            major, minor, torch.tensor([0]), counts, args, 1., groups,
            cross_enabled=False))['correction_gate_eligible'])
        args.tau_minor_cross = .5
        self.assertFalse(next(prediction_rows(
            major, minor, torch.tensor([0]), counts, args, 1., groups
        ))['correction_gate_eligible'])


if __name__ == '__main__':
    unittest.main()

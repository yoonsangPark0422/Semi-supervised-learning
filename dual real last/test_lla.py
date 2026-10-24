import io
import unittest
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from lla import LearnableLogitAdjustment, class_average, soft_cross_entropy
from correction_gate import correction_gate, suppress_conflicting_self, reference_masked_mean


class LLATests(unittest.TestCase):
    def test_formula_and_uniform_initialization(self):
        m = LearnableLogitAdjustment(3)
        z = torch.tensor([[2., 1., -1.]])
        self.assertTrue(torch.allclose(m.pseudo_labels(z), z.softmax(-1)))
        with torch.no_grad():
            m.adjustment.copy_(torch.tensor([2., 0., -1.]))
        expected = (z - m.adjustment.log_softmax(0)).softmax(-1)
        self.assertTrue(torch.allclose(m.pseudo_labels(z), expected))
        self.assertTrue(torch.allclose(expected, (z - m.adjustment).softmax(-1)))

    def test_class_average_including_missing_class(self):
        z = torch.tensor([[2., 0., -1.], [1., 0., -1.], [0., 2., -1.]])
        y = torch.tensor([0, 0, 1])
        ce = F.cross_entropy(z, y, reduction='none')
        expected = (ce[:2].mean() + ce[2]) / 3
        self.assertTrue(torch.allclose(class_average(z, F.one_hot(y, 3).float()), expected))

    def test_soft_denominators(self):
        z = torch.tensor([[2., 0., -1.], [0., 2., -1.]])
        q = torch.tensor([[.6, .3, .1], [.2, .5, .3]])
        logp = z.log_softmax(-1)
        expected = sum(-(q[:, k] * logp[:, k]).sum() / (q[:, k].sum() + 1e-8)
                       for k in range(3)) / 3
        self.assertTrue(torch.allclose(class_average(z, q), expected))
        self.assertTrue(torch.isfinite(class_average(z, torch.zeros_like(q))))

    def test_gradient_isolation(self):
        torch.manual_seed(1)
        m = LearnableLogitAdjustment(3)
        zl = torch.randn(4, 3, requires_grad=True)
        zw = torch.randn(5, 3, requires_grad=True)
        zs = torch.randn(5, 3, requires_grad=True)
        q = m.pseudo_labels(zw)
        a, b = m.class_averaged_losses(zl, torch.tensor([0, 0, 1, 2]), zs, q)
        (a + b).backward()
        self.assertIsNone(zl.grad)
        self.assertIsNone(zw.grad)
        self.assertIsNone(zs.grad)
        self.assertGreater(m.adjustment.grad.abs().sum().item(), 0)
        m.zero_grad()
        soft_cross_entropy(zs, q).mean().backward()
        self.assertIsNone(m.adjustment.grad)
        self.assertIsNone(zw.grad)
        self.assertGreater(zs.grad.abs().sum().item(), 0)

    def test_soft_target_not_hardened(self):
        z = torch.tensor([[1., -1.]])
        q = torch.tensor([[.51, .49]], requires_grad=True)
        expected = -(q.detach() * z.log_softmax(-1)).sum(-1)
        self.assertTrue(torch.allclose(soft_cross_entropy(z, q), expected))
        self.assertFalse(torch.allclose(expected, F.cross_entropy(z, torch.tensor([0]))))

    def test_gate_and_loss_denominator(self):
        m = torch.tensor([0, 0, 0])
        r = torch.tensor([0, 1, 0])
        a = torch.tensor([2, 2, 2])
        gate = correction_gate(m, r, a, torch.ones(3), torch.ones(3),
                               torch.tensor([1, 1, 0]))
        self.assertEqual(gate.tolist(), [True, False, False])
        ref = torch.tensor([1., 1., 0.])
        loss = torch.tensor([2., 4., 8.], requires_grad=True)
        value = reference_masked_mean(loss, suppress_conflicting_self(ref, gate), ref)
        value.backward()
        self.assertEqual(loss.grad.tolist(), [0., .5, 0.])

    def test_optimizer_checkpoint_roundtrip(self):
        torch.manual_seed(2)
        m = LearnableLogitAdjustment(3)
        opt = torch.optim.Adam(m.parameters(), lr=.002)
        z = torch.randn(4, 3)
        q = m.pseudo_labels(z)
        a, b = m.class_averaged_losses(z, torch.tensor([0, 1, 2, 0]), z, q)
        (a+b).backward()
        opt.step()
        buffer = io.BytesIO()
        torch.save({'m': m.state_dict(), 'opt': opt.state_dict()}, buffer)
        buffer.seek(0)
        state = torch.load(buffer, weights_only=True)
        restored = LearnableLogitAdjustment(3)
        opt2 = torch.optim.Adam(restored.parameters(), lr=.002)
        restored.load_state_dict(state['m'])
        opt2.load_state_dict(state['opt'])
        self.assertTrue(torch.equal(restored.pseudo_labels(z), m.pseudo_labels(z)))
        self.assertEqual(opt2.state_dict()['param_groups'], opt.state_dict()['param_groups'])

    def test_diagnostics_uses_learned_not_fixed_prior(self):
        from disagreement_diagnostics import prediction_rows
        m = LearnableLogitAdjustment(3)
        args = SimpleNamespace(T=1, minor_lla=m, tau_major_cross=.95,
                               tau_minor_cross=.95,
                               major_cross_skip_minor_conf_threshold=-1.)
        z = torch.tensor([[8., 0., 0.]])
        groups = (torch.tensor([0]), torch.tensor([1]), torch.tensor([2]))
        row = next(prediction_rows(z, z, torch.tensor([0]), torch.tensor([100., 10., 1.]),
                                   args, 1., groups))
        self.assertEqual(row['minor_raw_pred'], row['minor_adjusted_pred'])
        with torch.no_grad():
            m.adjustment.copy_(torch.tensor([10., 0., 0.]))
        row2 = next(prediction_rows(z, z, torch.tensor([0]), torch.tensor([100., 10., 1.]),
                                    args, 1., groups))
        self.assertNotEqual(row2['minor_adjusted_pred'], row['minor_adjusted_pred'])


if __name__ == '__main__':
    unittest.main()

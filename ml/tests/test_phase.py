import unittest

import torch

from ml.dataset import DataConfig, PhaseDataset
from ml.phase_torch import ForwardConfig, forward_stack, wavefront_rms


class TestPhaseForward(unittest.TestCase):
    def test_shape_and_normalisation(self):
        c = torch.tensor([0.5, 0.1, -0.2])
        s = forward_stack(c, ForwardConfig(n=24, noise_sigma=0.0))
        self.assertEqual(tuple(s.shape), (5, 24, 24))
        sums = s.sum(dim=(-2, -1))
        for v in sums:
            self.assertAlmostEqual(float(v), 1.0, places=5)

    def test_gradient_is_nonzero(self):
        c = torch.tensor([0.5, 0.1, -0.2], requires_grad=True)
        s = forward_stack(c, ForwardConfig(n=24, noise_sigma=0.0))
        weight = torch.linspace(0.0, 1.0, s.numel()).view_as(s)
        (s * weight).sum().backward()
        self.assertIsNotNone(c.grad)
        self.assertGreater(float(c.grad.abs().sum()), 0.0)

    def test_wavefront_rms(self):
        c = torch.tensor([0.3, 0.4, 0.0])
        self.assertAlmostEqual(float(wavefront_rms(c)), 0.5, places=6)

    def test_dataset_item(self):
        ds = PhaseDataset(DataConfig(n=24, samples=2, noise_sigma=0.01), train=True)
        x, y, yn = ds[0]
        self.assertEqual(tuple(x.shape), (5, 24, 24))
        self.assertEqual(tuple(y.shape), (3,))
        self.assertTrue(torch.allclose(y / torch.tensor([1.5, 0.5, 0.5]), yn, atol=1e-6))


if __name__ == "__main__":
    unittest.main()

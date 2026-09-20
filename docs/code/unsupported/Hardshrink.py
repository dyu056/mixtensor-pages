"""Hardshrink: NOT IMPLEMENTED as a numerical DSL adapter.

Verified configuration: Only the fixture in smoke/cases.py is verified.
Limitations: No exact lowering implemented: exp/log/reciprocal or value comparison/selection is outside NONLINEAR v0.1 modes. This is not an impossibility proof.
Inputs in smoke test: [[2, 3, 4]]
Outputs in smoke test: [{"shape": [2, 3, 4], "dtype": "torch.float64", "device": "cpu"}]
Source parameters/buffers are shared, not copied. No native source.forward call.
In-place mutation, hooks, distributed execution and export parity are not claimed.
"""
from torch import nn


class Hardshrink(nn.Module):
    """Explicit placeholder; this class is not counted as a DSL implementation."""
    def forward(self, *args, **kwargs):
        raise NotImplementedError('No exact lowering implemented: exp/log/reciprocal or value comparison/selection is outside NONLINEAR v0.1 modes. This is not an impossibility proof.')

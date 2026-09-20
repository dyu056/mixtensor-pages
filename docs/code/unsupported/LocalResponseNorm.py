"""LocalResponseNorm: NOT IMPLEMENTED as a numerical DSL adapter.

Verified configuration: Only the fixture in smoke/cases.py is verified.
Limitations: No lowering for response normalization power/reciprocal; NORM is only LayerNorm, not a universal normalization callback.
Inputs in smoke test: [[2, 4, 4, 4]]
Outputs in smoke test: [{"shape": [2, 4, 4, 4], "dtype": "torch.float64", "device": "cpu"}]
Source parameters/buffers are shared, not copied. No native source.forward call.
In-place mutation, hooks, distributed execution and export parity are not claimed.
"""
from torch import nn


class LocalResponseNorm(nn.Module):
    """Explicit placeholder; this class is not counted as a DSL implementation."""
    def forward(self, *args, **kwargs):
        raise NotImplementedError('No lowering for response normalization power/reciprocal; NORM is only LayerNorm, not a universal normalization callback.')

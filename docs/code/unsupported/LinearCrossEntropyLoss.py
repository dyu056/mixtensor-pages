"""LinearCrossEntropyLoss: NOT IMPLEMENTED as a numerical DSL adapter.

Verified configuration: Only the fixture in smoke/cases.py is verified.
Limitations: Fused loss with log-softmax/NLL; no logarithm lowering in v0.1.
Inputs in smoke test: [[3, 4], [3]]
Outputs in smoke test: [{"shape": [], "dtype": "torch.float64", "device": "cpu"}]
Source parameters/buffers are shared, not copied. No native source.forward call.
In-place mutation, hooks, distributed execution and export parity are not claimed.
"""
from torch import nn


class LinearCrossEntropyLoss(nn.Module):
    """Explicit placeholder; this class is not counted as a DSL implementation."""
    def forward(self, *args, **kwargs):
        raise NotImplementedError('Fused loss with log-softmax/NLL; no logarithm lowering in v0.1.')

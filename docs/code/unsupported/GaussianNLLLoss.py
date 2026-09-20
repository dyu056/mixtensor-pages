"""GaussianNLLLoss: NOT IMPLEMENTED as a numerical DSL adapter.

Verified configuration: Only the fixture in smoke/cases.py is verified.
Limitations: Reference smoke only; no DSL-only lowering implemented for this loss (log/exp/norm/index/branch semantics must be specified).
Inputs in smoke test: [[2, 3], [2, 3], [2, 3]]
Outputs in smoke test: [{"shape": [], "dtype": "torch.float64", "device": "cpu"}]
Source parameters/buffers are shared, not copied. No native source.forward call.
In-place mutation, hooks, distributed execution and export parity are not claimed.
"""
from torch import nn


class GaussianNLLLoss(nn.Module):
    """Explicit placeholder; this class is not counted as a DSL implementation."""
    def forward(self, *args, **kwargs):
        raise NotImplementedError('Reference smoke only; no DSL-only lowering implemented for this loss (log/exp/norm/index/branch semantics must be specified).')

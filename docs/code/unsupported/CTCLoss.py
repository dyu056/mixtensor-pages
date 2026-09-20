"""CTCLoss: NOT IMPLEMENTED as a numerical DSL adapter.

Verified configuration: Only the fixture in smoke/cases.py is verified.
Limitations: Needs log-domain dynamic programming, comparisons and variable-length indexing; SCAN alone is not its body.
Inputs in smoke test: [[5, 2, 4], [2, 2], [2], [2]]
Outputs in smoke test: [{"shape": [], "dtype": "torch.float64", "device": "cpu"}]
Source parameters/buffers are shared, not copied. No native source.forward call.
In-place mutation, hooks, distributed execution and export parity are not claimed.
"""
from torch import nn


class CTCLoss(nn.Module):
    """Explicit placeholder; this class is not counted as a DSL implementation."""
    def forward(self, *args, **kwargs):
        raise NotImplementedError('Needs log-domain dynamic programming, comparisons and variable-length indexing; SCAN alone is not its body.')

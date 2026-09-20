"""ParameterDict: NOT IMPLEMENTED as a numerical DSL adapter.

Verified configuration: abstract class / parameter or module container
Limitations: Not a standalone numerical operator; composition/parameter ownership is a host-language concern.
Inputs in smoke test: []
Outputs in smoke test: []
Source parameters/buffers are shared, not copied. No native source.forward call.
In-place mutation, hooks, distributed execution and export parity are not claimed.
"""
from torch import nn


class ParameterDict(nn.Module):
    """Explicit placeholder; this class is not counted as a DSL implementation."""
    def forward(self, *args, **kwargs):
        raise NotImplementedError('Not a standalone numerical operator; composition/parameter ownership is a host-language concern.')

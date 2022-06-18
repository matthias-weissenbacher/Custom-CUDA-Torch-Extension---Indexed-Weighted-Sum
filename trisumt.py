import math
from torch import nn
from torch.autograd import Function
import torch

import trisum_cuda

torch.manual_seed(42)


class trisumFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, idxs, idxs_bw):
        outputs = trisum_cuda.forward(input, weights, idxs, idxs_bw)
        out = outputs[:1]
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        outputs = trisum_cuda.backward(grad_out.contiguous(), *ctx.saved_variables)
        d_input, d_weights = outputs
        return d_input, d_weights


class trisum(nn.Module):
    def __init__(self,  state_size):
        super(trisum, self).__init__()

        self.state_size = state_size
        self.weights = nn.Parameter(torch.Tensor(3*state_size**2))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, idxs, idxs_bw):
        return trisumFunction.apply(input, self.weights, idxs, idxs_bw)

    
 
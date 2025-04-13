
import torch
from torch import Tensor
from megatron.training import print_rank_0
from megatron.core.utils import make_viewless_tensor

class LayerNormWithGradLogging(torch.nn.Module):
    """
    Non-fused LayerNorm implementation with gradient logging capabilities.
    This implementation ensures distinct tensors for proper gradient tracking.
    """
    def __init__(self, hidden_size, eps=1e-5, weight=None, bias=None, layer_idx=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.layer_idx = layer_idx
        
        if weight is not None:
            self.weight = weight
        else:
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            
        if bias is not None:
            self.bias = bias
        else:
            self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, input: Tensor) -> Tensor:
        return apply_layernorm_with_grad_logging(
            input, self.weight, self.bias, self.eps, self.layer_idx)


class LayerNormWithGradLoggingFunction(torch.autograd.Function):
    """
    Custom autograd function for LayerNorm that logs gradient norms during backward pass.
    This allows us to properly capture the gradient transformations that happen inside
    the layernorm operation, which isn't possible with standard hooks.
    """
    
    @staticmethod
    def forward(ctx, input, weight, bias, eps, layer_idx):
        input_clone = input.clone()
        ctx.save_for_backward(input_clone, weight, bias)
        ctx.eps = eps
        ctx.layer_idx = layer_idx
        
        activation_norm = torch.norm(input.float())
        print_rank_0(f'Layer {layer_idx} activation norm before LayerNorm: {activation_norm}, tensor ID: {id(input)}, tensor shape: {input.shape}')
        
        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (input - mean) / torch.sqrt(var + eps)
        
        if weight is not None and bias is not None:
            output = normalized * weight + bias
        else:
            output = normalized
        
        output = output.clone()
        print_rank_0(f'Layer {layer_idx} output tensor ID: {id(output)}, tensor shape: {output.shape}')
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        eps = ctx.eps
        layer_idx = ctx.layer_idx
        
        grad_output_norm = torch.norm(grad_output.float())
        print_rank_0(f'Layer {layer_idx} LayerNorm output dgrad gradnorm: {grad_output_norm}, tensor ID: {id(grad_output)}, tensor shape: {grad_output.shape}')
        
        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (input - mean) / torch.sqrt(var + eps)
        
        if weight is not None:
            grad_normalized = grad_output * weight
        else:
            grad_normalized = grad_output
            
        N = input.size(-1)
        dx_normalized = grad_normalized
        dvar = torch.sum(dx_normalized * (input - mean) * -0.5 * torch.pow(var + eps, -1.5), dim=-1, keepdim=True)
        dmean = torch.sum(dx_normalized * -1.0 / torch.sqrt(var + eps), dim=-1, keepdim=True)
        dmean += dvar * torch.sum(-2.0 * (input - mean), dim=-1, keepdim=True) / N
        dx = dx_normalized / torch.sqrt(var + eps) + dvar * 2.0 * (input - mean) / N + dmean / N
        
        dx = dx.clone()
        
        grad_input_norm = torch.norm(dx.float())
        print_rank_0(f'Layer {layer_idx} LayerNorm input dgrad gradnorm: {grad_input_norm}, tensor ID: {id(dx)}, tensor shape: {dx.shape}')
        
        if weight is not None and bias is not None:
            dw = torch.sum(grad_output * normalized, dim=0)
            db = torch.sum(grad_output, dim=0)
            return dx, dw, db, None, None
        else:
            return dx, None, None, None, None
            
def apply_layernorm_with_grad_logging(input, weight, bias, eps, layer_idx):
    """
    Apply layer normalization with gradient logging.
    
    Args:
        input: Input tensor
        weight: Weight parameter
        bias: Bias parameter
        eps: Epsilon value for numerical stability
        layer_idx: Layer index for logging
        
    Returns:
        Normalized tensor
    """
    return LayerNormWithGradLoggingFunction.apply(input, weight, bias, eps, layer_idx)


import torch
from megatron.training import print_rank_0

class LayerNormWithGradLoggingFunction(torch.autograd.Function):
    """
    Custom autograd function for LayerNorm that logs gradient norms during backward pass.
    This allows us to properly capture the gradient transformations that happen inside
    the layernorm operation, which isn't possible with standard hooks.
    """
    
    @staticmethod
    def forward(ctx, input, weight, bias, eps, layer_idx, log_activation=True):
        ctx.save_for_backward(input, weight, bias)
        ctx.eps = eps
        ctx.layer_idx = layer_idx
        
        if log_activation:
            activation_norm = torch.norm(input.float())
            print_rank_0(f'Layer {layer_idx} activation norm before LayerNorm: {activation_norm}, tensor shape: {input.shape}')
        
        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (input - mean) / torch.sqrt(var + eps)
        
        if weight is not None and bias is not None:
            output = normalized * weight + bias
        else:
            output = normalized
            
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        eps = ctx.eps
        layer_idx = ctx.layer_idx
        
        grad_output_norm = torch.norm(grad_output.float())
        print_rank_0(f'Layer {layer_idx} LayerNorm output dgrad gradnorm: {grad_output_norm}, tensor shape: {grad_output.shape}')
        
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
        
        grad_input_norm = torch.norm(dx.float())
        print_rank_0(f'Layer {layer_idx} LayerNorm input dgrad gradnorm: {grad_input_norm}, tensor shape: {dx.shape}')
        
        if weight is not None and bias is not None:
            dw = torch.sum(grad_output * normalized, dim=0)
            db = torch.sum(grad_output, dim=0)
            return dx, dw, db, None, None, None
        else:
            return dx, None, None, None, None, None
            
def apply_layernorm_with_grad_logging(input, weight, bias, eps, layer_idx, log_activation=True):
    return LayerNormWithGradLoggingFunction.apply(input, weight, bias, eps, layer_idx, log_activation)

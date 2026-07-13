import copy
import torch
import torch.nn as nn


def fuse_linear_bn(linear: nn.Linear, bn: nn.BatchNorm1d) -> nn.Linear:
    """
    Returns a new nn.Linear that reproduces linear and bn layers
    """
    fused = nn.Linear(linear.in_features, linear.out_features, bias=True)

    w = linear.weight.detach().clone()
    b = linear.bias.detach().clone() if linear.bias is not None \
        else torch.zeros(linear.out_features)

    gamma = bn.weight.detach().clone()
    beta = bn.bias.detach().clone()
    mean = bn.running_mean.detach().clone()
    var = bn.running_var.detach().clone()

    scale = gamma / torch.sqrt(var + bn.eps)

    fused.weight.data = w * scale.unsqueeze(1)
    fused.bias.data = (b - mean) * scale + beta

    return fused


def build_lrp_view(model: nn.Module) -> nn.Module:
    """
    copies model replacing linear and bn layer with equivalent linear + identity layer
    """
    model.eval()
    lrp_model = copy.deepcopy(model)

    names = [n for n, _ in lrp_model.named_children()]

    i = 0
    while i < len(names) - 1:
        name_a, name_b = names[i], names[i + 1]
        mod_a = getattr(lrp_model, name_a)
        mod_b = getattr(lrp_model, name_b)

        if isinstance(mod_a, nn.Linear) and isinstance(mod_b, nn.BatchNorm1d):
            fused = fuse_linear_bn(mod_a, mod_b)
            setattr(lrp_model, name_a, fused)
            setattr(lrp_model, name_b, nn.Identity())
            i += 2
        else:
            i += 1

    return lrp_model


def has_batchnorm(model: nn.Module) -> bool:
    return any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules())
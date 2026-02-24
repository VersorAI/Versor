import torch
import math
import gacore.kernel as kernel

class CayleyAdam(torch.optim.Optimizer):
    """
    Manifold-Aware Optimizer for Clifford multivectors.
    Updated for general Clifford signatures.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, signature=None):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(CayleyAdam, self).__init__(params, defaults)
        self.signature = signature if signature is not None else [1, 1, 1, 1, -1]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                # Update multivector state
                p.add_(exp_avg / denom, alpha=-step_size)
                
                # Manifold Normalization for multivector parameters
                # We assume a multivector parameter has a trailing dimension = 2^k
                n_dims = p.shape[-1]
                if n_dims >= 2 and (n_dims & (n_dims - 1) == 0):
                    # Only normalize if it matches the expected signature dimension
                    if n_dims == (1 << len(self.signature)):
                        p.copy_(kernel.manifold_normalization(p, self.signature))

        return loss


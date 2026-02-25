# mace/tools/swa_state_dict.py
from __future__ import annotations
import torch

class StateDictSWA:
    """
    SWA that keeps a running average of the model's state_dict, avoiding deepcopy
    and TorchScript pickling issues. It mimics the minimal API used in training:
      - update_parameters(model)
      - n_averaged (torch.LongTensor)
    Use load_into(model) to materialize the average back into a real module.
    """
    def __init__(self, model, avg_fn=None, device=None):
        self.device = device
        self.n_averaged = torch.tensor(0, dtype=torch.long, device=device)
        # average only floating-point tensors; keep others to restore unchanged
        sd = model.state_dict()
        self._avg = {
            k: v.detach().to(device).clone()
            for k, v in sd.items()
            if hasattr(v, "dtype") and getattr(v.dtype, "is_floating_point", False)
        }
        self._non_float = {k: v for k, v in sd.items() if k not in self._avg}
        # default equal-weight averaging: avg += (new - avg) / (n+1)
        self.avg_fn = avg_fn or (lambda avg, new, n: avg + (new - avg) / (n + 1))

    @torch.no_grad()
    def update_parameters(self, model):
        cur = model.state_dict()
        n = int(self.n_averaged.item())
        for k, avg in self._avg.items():
            new = cur[k].detach().to(avg.device)
            self._avg[k].copy_(self.avg_fn(avg, new, n))
        self.n_averaged += 1

    @torch.no_grad()
    def load_into(self, model):
        """Copy the running average weights into a real module."""
        tgt = model.state_dict()
        for k, v in self._avg.items():
            tgt[k].copy_(v.to(tgt[k].dtype, non_blocking=True))
        for k, v in self._non_float.items():
            tgt[k].copy_(v)

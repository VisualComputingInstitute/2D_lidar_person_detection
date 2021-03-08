from torch import optim


class Optim:
    def __init__(self, model, cfg):
        self._optim = optim.Adam(model.parameters(), amsgrad=True)
        self._lr_scheduler = _ExpDecayScheduler(**cfg["scheduler_kwargs"])

    def zero_grad(self):
        self._optim.zero_grad()

    def step(self):
        self._optim.step()

    def state_dict(self):
        return self._optim.state_dict()

    def load_state_dict(self, state_dict):
        self._optim.load_state_dict(state_dict)

    def set_lr(self, epoch):
        for group in self._optim.param_groups:
            group["lr"] = self._lr_scheduler(epoch)

    def get_lr(self):
        return self._optim.param_groups[0]["lr"]


class _ExpDecayScheduler:
    """
    Return `v0` until `e` reaches `e0`, then exponentially decay
    to `v1` when `e` reaches `e1` and return `v1` thereafter, until
    reaching `eNone`, after which it returns `None`.
    """

    def __init__(self, epoch0, lr0, epoch1, lr1):
        self._epoch0 = epoch0
        self._epoch1 = epoch1
        self._lr0 = lr0
        self._lr1 = lr1

    def __call__(self, epoch):
        if epoch < self._epoch0:
            return self._lr0
        elif epoch > self._epoch1:
            return self._lr1
        else:
            return self._lr0 * (self._lr1 / self._lr0) ** (
                (epoch - self._epoch0) / (self._epoch1 - self._epoch0)
            )

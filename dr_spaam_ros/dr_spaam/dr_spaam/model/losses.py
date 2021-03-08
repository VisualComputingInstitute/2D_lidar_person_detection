import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartiallyHuberisedBCELoss(nn.Module):
    """partially Huberised softmax cross-entrop
    https://openreview.net/pdf?id=rklB76EKPr
    """

    def __init__(self, tau=5.0):
        super(PartiallyHuberisedBCELoss, self).__init__()
        self._tau = tau
        self._log_tau = math.log(self._tau)
        self._inv_tau = 1.0 / self._tau

    def forward(self, pred, target, reduction="mean"):
        pred_logits = torch.sigmoid(pred)
        neg_pred_logits = 1.0 - pred_logits

        loss_pos = -self._tau * pred_logits + self._log_tau + 1.0
        pos_mask = pred_logits > self._inv_tau
        loss_pos[pos_mask] = -torch.log(pred_logits[pos_mask])

        loss_neg = -self._tau * neg_pred_logits + self._log_tau + 1.0
        neg_mask = neg_pred_logits > self._inv_tau
        loss_neg[neg_mask] = -torch.log(neg_pred_logits[neg_mask])

        loss = target * loss_pos + (1.0 - target) * loss_neg

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise RuntimeError


class SelfPacedLearningLoss(nn.Module):
    """Self-paced learning loss
    https://arxiv.org/abs/1712.05055
    https://papers.nips.cc/paper/3923-self-paced-learning-for-latent-variable-models
    https://arxiv.org/abs/1511.06049
    """

    def __init__(self, base_loss, lam1=0.4, lam2=0.5, alpha=1e-2):
        super(SelfPacedLearningLoss, self).__init__()
        self._base_loss = base_loss
        self._lam1 = lam1
        self._lam2 = lam2

        self._l1 = None
        self._l2 = None
        self._alpha = alpha

        self._lam1_max = 0.60
        self._lam2_max = 0.72

        self._step = -1
        self._burn_in = False
        self._burn_in_step = int(2629 * 0.5)
        self._update_step = int(2629)
        self._update_rate = 1.02

    def forward(self, pred, target, reduction="mean"):
        self._update()

        if self._burn_in:
            return self._base_loss(pred, target, reduction=reduction)

        # raw loss
        base_loss = self._base_loss(pred, target, reduction="none")

        # exponential moving average of loss percentile
        with torch.no_grad():
            l1_now = self._percentile(base_loss, self._lam1)
            self._l1 = (
                self._alpha * l1_now + (1.0 - self._alpha) * self._l1
                if self._l1 is not None
                else l1_now
            )

            l2_now = self._percentile(base_loss, self._lam2)
            self._l2 = (
                self._alpha * l2_now + (1.0 - self._alpha) * self._l2
                if self._l2 is not None
                else l2_now
            )

            # compute v
            v = (1.0 - (base_loss - self._l1) / (self._l2 - self._l1)).clamp(
                min=0.0, max=1.0
            )

        # weighted loss
        loss = base_loss * v

        # NOTE reweight the loss, it may be better to use a fixed weighting factor
        # loss = loss / (v.sum() / v.numel())
        loss = loss / (loss.sum() / base_loss.sum())

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise RuntimeError

    def _update(self):
        self._step += 1

        if self._burn_in:
            if self._step >= self._burn_in_step:
                self._burn_in = False
                self._step = 0
        else:
            if self._step >= self._update_step:
                self._lam1 = min(self._lam1_max, self._lam1 * self._update_rate)
                self._lam2 = min(self._lam2_max, self._lam2 * self._update_rate)
                self._step = 0

    def _percentile(self, t, q):
        """
        From https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30
        """
        # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
        # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
        # so that ``round()`` returns an integer, even if q is a np.float32.
        k = 1 + round(float(q) * (t.numel() - 1))
        result = t.kthvalue(k).values.item()
        return result


class SymmetricBCELoss(nn.Module):
    """Symmetric Cross Entropy loss https://arxiv.org/pdf/1908.06112.pdf
    for binary classification
    """

    def __init__(self, alpha=0.1, beta=0.5, A=-6):
        assert A < 0.0
        super(SymmetricBCELoss, self).__init__()
        self._alpha = alpha
        self._beta = beta
        self._A = A

    def forward(self, pred, target, reduction="mean"):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        log_target_pos = torch.log(target + 1e-10).clamp_min(self._A)
        log_target_neg = torch.log(1.0 - target + 1e-10).clamp_min(self._A)
        pred_logits = torch.sigmoid(pred)

        rbce_loss = -log_target_pos * pred_logits - log_target_neg * (1.0 - pred_logits)

        loss = self._alpha * bce_loss + self._beta * rbce_loss

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise RuntimeError


class FocalLoss(nn.Module):
    """From https://github.com/mbsariyildiz/focal-loss.pytorch/blob/master/focalloss.py
    """

    def __init__(self, gamma=0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)

    def forward(self, input, target, reduction="mean"):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise RuntimeError


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=-1):
        super(BinaryFocalLoss, self).__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, pred, target, reduction="mean"):
        return binary_focal_loss(pred, target, self.gamma, self.alpha, reduction)


def binary_focal_loss(pred, target, gamma=2.0, alpha=-1, reduction="mean"):
    loss_pos = -target * (1.0 - pred) ** gamma * torch.log(pred)
    loss_neg = -(1.0 - target) * pred ** gamma * torch.log(1.0 - pred)

    if alpha >= 0.0 and alpha <= 1.0:
        loss_pos = loss_pos * alpha
        loss_neg = loss_neg * (1.0 - alpha)

    loss = loss_pos + loss_neg

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise RuntimeError

from functools import partial
import torch.nn.functional as F

from .losses import (
    SymmetricBCELoss,
    SelfPacedLearningLoss,
    PartiallyHuberisedBCELoss,
)
from .dr_spaam_fn import (
    _model_fn,
    _model_eval_fn,
    _model_eval_collate_fn,
    _model_fn_mixup,
)


def get_model(cfg):
    if cfg["cls_loss"]["type"] == 0:
        cls_loss = F.binary_cross_entropy_with_logits

    elif cfg["cls_loss"]["type"] == 1:
        if "kwargs" in cfg["cls_loss"]:
            cls_loss = SymmetricBCELoss(**cfg["cls_loss"]["kwargs"])
        else:
            cls_loss = SymmetricBCELoss()

    elif cfg["cls_loss"]["type"] == 2:
        if "kwargs" in cfg["cls_loss"]:
            cls_loss = PartiallyHuberisedBCELoss(**cfg["cls_loss"]["kwargs"])
        else:
            cls_loss = PartiallyHuberisedBCELoss()

    else:
        raise NotImplementedError

    if cfg["self_paced"]:
        cls_loss = SelfPacedLearningLoss(cls_loss)

    if cfg["type"] == "drow":
        from .drow_net import DrowNet

        d = DrowNet(
            cls_loss=cls_loss,
            mixup_alpha=cfg["mixup_alpha"],
            mixup_w=cfg["mixup_w"],
            **cfg["kwargs"]
        )
        d.model_eval_fn = _model_eval_fn
        d.model_eval_collate_fn = _model_eval_collate_fn
        d.model_fn = partial(
            _model_fn, max_num_pts=1e6, cls_loss_weight=1.0 - d.mixup_w
        )
        d.model_fn_mixup = partial(
            _model_fn_mixup, max_num_pts=1e6, cls_loss_weight=d.mixup_w
        )
        return d
    elif cfg["type"] == "dr-spaam":
        from .dr_spaam import DrSpaam

        d = DrSpaam(
            cls_loss=cls_loss,
            mixup_alpha=cfg["mixup_alpha"],
            mixup_w=cfg["mixup_w"],
            **cfg["kwargs"]
        )
        d.model_eval_fn = _model_eval_fn
        d.model_eval_collate_fn = _model_eval_collate_fn
        d.model_fn = partial(
            _model_fn, max_num_pts=1000, cls_loss_weight=1.0 - d.mixup_w
        )
        d.model_fn_mixup = partial(
            _model_fn_mixup, max_num_pts=1000, cls_loss_weight=d.mixup_w
        )
        return d
    elif cfg["type"] == "detr":
        from .detr_net import DeTrNet

        return DeTrNet(**cfg["kwargs"])
    else:
        raise NotImplementedError

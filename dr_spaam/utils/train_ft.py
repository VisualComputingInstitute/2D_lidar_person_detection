import argparse
import yaml
import torch
import torch.nn as nn

from dr_spaam.dataset import get_dataloader
from dr_spaam.pipeline.pipeline import Pipeline
from dr_spaam.model import get_model


def run_training(model, pipeline, cfg):
    # main train loop
    train_loader = get_dataloader(
        split="train", shuffle=True, dataset_cfg=cfg["dataset"], **cfg["dataloader"]
    )
    val_loader = get_dataloader(
        split="val", shuffle=True, dataset_cfg=cfg["dataset"], **cfg["dataloader"]
    )
    status = pipeline.train(model, train_loader, val_loader)

    # test after training
    if not status:
        test_loader = get_dataloader(
            split="test",
            batch_size=1,
            num_workers=1,
            shuffle=False,
            dataset_cfg=cfg["dataset"],
        )
        pipeline.evaluate(model, test_loader, tb_prefix="TEST")


def run_evaluation(model, pipeline, cfg):
    val_loader = get_dataloader(
        split="val",
        batch_size=1,
        num_workers=1,
        shuffle=False,
        dataset_cfg=cfg["dataset"],
    )
    pipeline.evaluate(model, val_loader, tb_prefix="VAL")

    test_loader = get_dataloader(
        split="test",
        batch_size=1,
        num_workers=1,
        shuffle=False,
        dataset_cfg=cfg["dataset"],
    )
    pipeline.evaluate(model, test_loader, tb_prefix="TEST")


if __name__ == "__main__":
    # Run benchmark to select fastest implementation of ops.
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg", type=str, required=True, help="configuration of the experiment"
    )
    parser.add_argument("--ckpt", type=str, required=False, default=None)
    parser.add_argument("--cont", default=False, action="store_true")
    parser.add_argument("--tmp", default=False, action="store_true")
    parser.add_argument("--evaluation", default=False, action="store_true")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
        cfg["pipeline"]["Logger"]["backup_list"].append(args.cfg)
        if args.tmp:
            cfg["pipeline"]["Logger"]["tag"] += "_TMP"

    model = get_model(cfg["model"])
    model.cuda()
    model.requires_grad_(False)
    print (model)

    for i, child in enumerate(model.children()):
        if i in [3, 4, 5]:
            if i in [4, 5]:
                for m in child.modules():
                    if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                        nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
                    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
            for param in child.parameters():
                param.requires_grad = True

    pipeline = Pipeline(model, cfg["pipeline"])

    if args.ckpt:
        pipeline.load_ckpt(model, args.ckpt)
    elif args.cont and pipeline.sigterm_ckpt_exists():
        pipeline.load_sigterm_ckpt(model)

    # dirty fix to avoid repeatative entries in cfg file
    cfg["dataset"]["mixup_alpha"] = cfg["model"]["mixup_alpha"]

    # training or evaluation
    if not args.evaluation:
        run_training(model, pipeline, cfg)
    else:
        run_evaluation(model, pipeline, cfg)

    pipeline.close()

import numpy as np

import torch
import torch.nn.functional as F

import dr_spaam.utils.utils as u
import dr_spaam.utils.precision_recall as pru
from dr_spaam.utils.plotting import plot_one_frame


# TODO when to plot?
_PLOTTING = False


def _sample_or_repeat(population, n):
    """Select n sample from population, without replacement if population size
    greater than n, otherwise with replacement.

    Only work for population of 1D tensor (N,)
    """
    N = len(population)
    if N == n:
        return population
    elif N > n:
        return population[torch.randperm(N, device=population.device)[:n]]
    else:
        return population[torch.randint(N, (n,), device=population.device)]


def _balanced_sampling_reweighting(target_cls, goal_fg_ratio=0.4):
    # target_cls is 1D tensor (N, )
    N = target_cls.shape[0]
    goal_fg_num = int(N * goal_fg_ratio)
    goal_bg_num = int(N * (1.0 - goal_fg_ratio))

    inds = torch.arange(N, device=target_cls.device)
    fg_inds = inds[target_cls > 0]
    bg_inds = inds[target_cls == 0]

    if len(fg_inds) > 0:
        fg_inds = _sample_or_repeat(fg_inds, goal_fg_num)
        bg_inds = _sample_or_repeat(bg_inds, goal_bg_num)
        sample_inds = torch.cat((fg_inds, bg_inds))
    else:
        sample_inds = _sample_or_repeat(bg_inds, N)

    weights = torch.zeros(N, device=target_cls.device).float()
    weights.index_add_(0, sample_inds, torch.ones_like(sample_inds).float())

    return weights


def _model_fn(model, batch_dict, max_num_pts=1e6, cls_loss_weight=1.0):
    tb_dict, rtn_dict = {}, {}

    net_input = batch_dict["input"]
    target_cls, target_reg = batch_dict["target_cls"], batch_dict["target_reg"]

    B, N = target_cls.shape

    # train only on part of scan, if the GPU cannot fit the whole scan
    num_pts = target_cls.shape[1]
    if model.training and num_pts > max_num_pts:
        idx0 = np.random.randint(0, num_pts - max_num_pts)
        idx1 = idx0 + max_num_pts
        target_cls = target_cls[:, idx0:idx1]
        target_reg = target_reg[:, idx0:idx1, :]
        net_input = net_input[:, idx0:idx1, :, :]
        N = max_num_pts

    # to gpu
    net_input = torch.from_numpy(net_input).cuda(non_blocking=True).float()
    target_cls = torch.from_numpy(target_cls).cuda(non_blocking=True).float()
    target_reg = torch.from_numpy(target_reg).cuda(non_blocking=True).float()

    # forward pass
    rtn_tuple = model(net_input)

    # so this function can be used for both DROW and DR-SPAAM
    if len(rtn_tuple) == 2:
        pred_cls, pred_reg = rtn_tuple
    elif len(rtn_tuple) == 3:
        pred_cls, pred_reg, pred_sim = rtn_tuple
        rtn_dict["pred_sim"] = pred_sim

    target_cls = target_cls.view(B * N)
    pred_cls = pred_cls.view(B * N)

    # number of valid points
    valid_mask = target_cls >= 0
    valid_ratio = torch.sum(valid_mask).item() / (B * N)
    # assert valid_ratio > 0, "No valid points in this batch."
    tb_dict["valid_ratio"] = valid_ratio

    # cls loss
    cls_loss = (
        model.cls_loss(pred_cls[valid_mask], target_cls[valid_mask], reduction="mean")
        * cls_loss_weight
    )
    total_loss = cls_loss
    tb_dict["cls_loss"] = cls_loss.item()

    # number fg points
    # NOTE supervise regression for both close and far neighbor points
    fg_mask = torch.logical_or(target_cls == 1, target_cls == -1)
    fg_ratio = torch.sum(fg_mask).item() / (B * N)
    tb_dict["fg_ratio"] = fg_ratio

    # reg loss
    if fg_ratio > 0.0:
        target_reg = target_reg.view(B * N, -1)
        pred_reg = pred_reg.view(B * N, -1)
        reg_loss = F.mse_loss(pred_reg[fg_mask], target_reg[fg_mask], reduction="none")
        reg_loss = torch.sqrt(torch.sum(reg_loss, dim=1)).mean()
        total_loss = total_loss + reg_loss
        tb_dict["reg_loss"] = reg_loss.item()

    # # regularization loss for spatial attention
    # if spatial_drow:
    #     # shannon entropy
    #     att_loss = (-torch.log(pred_sim + 1e-5) * pred_sim).sum(dim=2).mean()
    #     tb_dict['att_loss'] = att_loss.item()
    #     total_loss = total_loss + att_loss

    rtn_dict["pred_reg"] = pred_reg.view(B, N, 2)
    rtn_dict["pred_cls"] = pred_cls.view(B, N)

    return total_loss, tb_dict, rtn_dict


def _model_fn_mixup(model, batch_dict, max_num_pts=1e6, cls_loss_weight=1.0):
    # mixup regularization for robust training against label noise
    # https://arxiv.org/pdf/1710.09412.pdf

    tb_dict, rtn_dict = {}, {}

    net_input = batch_dict["input_mixup"]
    target_cls = batch_dict["target_cls_mixup"]

    B, N = target_cls.shape

    # train only on part of scan, if the GPU cannot fit the whole scan
    num_pts = target_cls.shape[1]
    if model.training and num_pts > max_num_pts:
        idx0 = np.random.randint(0, num_pts - max_num_pts)
        idx1 = idx0 + max_num_pts
        target_cls = target_cls[:, idx0:idx1]
        net_input = net_input[:, idx0:idx1, :, :]
        N = max_num_pts

    # to gpu
    net_input = torch.from_numpy(net_input).cuda(non_blocking=True).float()
    target_cls = torch.from_numpy(target_cls).cuda(non_blocking=True).float()

    # forward pass
    rtn_tuple = model(net_input)

    # so this function can be used for both DROW and DR-SPAAM
    if len(rtn_tuple) == 2:
        pred_cls, pred_reg = rtn_tuple
    elif len(rtn_tuple) == 3:
        pred_cls, pred_reg, pred_sim = rtn_tuple
        rtn_dict["pred_sim"] = pred_sim

    target_cls = target_cls.view(B * N)
    pred_cls = pred_cls.view(B * N)

    # number of valid points
    valid_mask = target_cls >= 0
    valid_ratio = torch.sum(valid_mask).item() / (B * N)
    # assert valid_ratio > 0, "No valid points in this batch."
    tb_dict["valid_ratio_mixup"] = valid_ratio

    # cls loss
    cls_loss = (
        model.cls_loss(pred_cls[valid_mask], target_cls[valid_mask], reduction="mean")
        * cls_loss_weight
    )
    total_loss = cls_loss
    tb_dict["cls_loss_mixup"] = cls_loss.item()

    return total_loss, tb_dict, rtn_dict


def _model_eval_fn(model, batch_dict):
    _, tb_dict, rtn_dict = _model_fn(model, batch_dict)

    pred_cls = torch.sigmoid(rtn_dict["pred_cls"]).data.cpu().numpy()
    pred_reg = rtn_dict["pred_reg"].data.cpu().numpy()

    # # DEBUG use perfect predictions
    # pred_cls = batch_dict["target_cls"]
    # pred_cls[pred_cls < 0] = 1
    # pred_reg = batch_dict["target_reg"]

    fig_dict = {}
    file_dict = {}

    # postprocess network prediction to get detection
    scans = batch_dict["scans"]
    scan_phi = batch_dict["scan_phi"]
    for ib in range(len(scans)):
        # store detection, which will be used by _model_eval_collate_fn to compute AP
        dets_xy, dets_cls, _ = u.nms_predicted_center(
            scans[ib][-1], scan_phi[ib], pred_cls[ib], pred_reg[ib]
        )
        frame_id = f"{batch_dict['frame_id'][ib]:06d}"
        sequence = batch_dict["sequence"][ib]

        # save detection results for evaluation
        det_str = pru.drow_detection_to_kitti_string(dets_xy, dets_cls, None)
        file_dict[f"detections/{sequence}/{frame_id}"] = det_str

        # save corresponding groundtruth for evaluation
        anns_rphi = batch_dict["dets_wp"][ib]
        if len(anns_rphi) > 0:
            anns_rphi = np.array(anns_rphi, dtype=np.float32)
            gts_xy = np.stack(u.rphi_to_xy(anns_rphi[:, 0], anns_rphi[:, 1]), axis=1)
            gts_occluded = np.logical_not(batch_dict["anns_valid_mask"][ib]).astype(
                np.int
            )
            gts_str = pru.drow_detection_to_kitti_string(gts_xy, None, gts_occluded)
            file_dict[f"groundtruth/{sequence}/{frame_id}"] = gts_str
        else:
            file_dict[f"groundtruth/{sequence}/{frame_id}"] = ""

        # TODO When to plot
        if _PLOTTING:
            fig, ax = plot_one_frame(
                batch_dict, ib, pred_cls[ib], pred_reg[ib], dets_cls, dets_xy
            )
            fig_dict[f"figs/{sequence}/{frame_id}"] = (fig, ax)

    return tb_dict, file_dict, fig_dict


def _model_eval_collate_fn(tb_dict_list, result_dir):
    # tb_dict should only contain scalar values, collate them into an array
    # and take their mean as the value of the epoch
    epoch_tb_dict = {}
    for batch_tb_dict in tb_dict_list:
        for k, v in batch_tb_dict.items():
            epoch_tb_dict.setdefault(k, []).append(v)
    for k, v in epoch_tb_dict.items():
        epoch_tb_dict[k] = np.array(v).mean()

    sequences, sequences_results_03, sequences_results_05 = pru.evaluate_drow(
        result_dir, remove_raw_files=True
    )

    # save evaluation output to system
    epoch_dict = {}
    for n, re03, re05 in zip(sequences, sequences_results_03, sequences_results_05):
        epoch_dict[f"evaluation/{n}/result_r03"] = re03
        epoch_dict[f"evaluation/{n}/result_r05"] = re05

        # log scalar values in tensorboard
        for k, v in re03.items():
            if not isinstance(v, (np.ndarray, list, tuple)):
                epoch_tb_dict[f"{n}_{k}_r03"] = v

        for k, v in re05.items():
            if not isinstance(v, (np.ndarray, list, tuple)):
                epoch_tb_dict[f"{n}_{k}_r05"] = v

    return epoch_tb_dict, epoch_dict

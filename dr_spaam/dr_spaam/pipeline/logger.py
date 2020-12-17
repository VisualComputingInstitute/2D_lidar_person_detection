import logging
import os
import pickle
from shutil import copyfile
import time

import json
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch


def _create_logger(root_dir, file_name="log.txt"):
    log_file = os.path.join(root_dir, file_name)
    log_format = "%(asctime)s  %(levelname)5s  %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


class Logger:
    def __init__(self, cfg):
        cfg["log_dir"] = os.path.abspath(os.path.expanduser(cfg["log_dir"]))

        # main log
        if "use_timestamp" in cfg.keys() and cfg["use_timestamp"] is False:
            dir_name = cfg["tag"]
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            dir_name = f"{timestamp}_{cfg['tag']}"
        self.__log_dir = os.path.join(cfg["log_dir"], dir_name)
        os.makedirs(self.__log_dir, exist_ok=True)

        self.__log = _create_logger(self.__log_dir, cfg["log_fname"])
        self.log_debug(f"Log directory: {self.__log_dir}")

        # backup important files (e.g. config.yaml)
        self.__backup_dir = os.path.join(self.__log_dir, "backup")
        os.makedirs(self.__backup_dir, exist_ok=True)
        for file_ in cfg["backup_list"]:
            self.log_debug(f"Backup {file_}")
            copyfile(
                os.path.abspath(file_),
                os.path.join(self.__backup_dir, os.path.basename(file_)),
            )

        # for storing results (network output etc.)
        self.__output_dir = os.path.join(self.__log_dir, "output")
        os.makedirs(self.__output_dir, exist_ok=True)

        # for storing ckpt
        self.__ckpt_dir = os.path.join(self.__log_dir, "ckpt")
        os.makedirs(self.__ckpt_dir, exist_ok=True)

        # for tensorboard
        tb_dir = os.path.join(self.__log_dir, "tb")
        os.makedirs(tb_dir, exist_ok=True)
        self.__tb = SummaryWriter(log_dir=tb_dir)

        # the sigterm checkpoint
        self.__sigterm_ckpt = os.path.join(
            cfg["log_dir"], f"sigterm_ckpt_{cfg['tag']}.pth"
        )

        gpu = (
            os.environ["CUDA_VISIBLE_DEVICES"]
            if "CUDA_VISIBLE_DEVICES" in os.environ.keys()
            else "ALL"
        )
        self.log_info(f"CUDA_VISIBLE_DEVICES={gpu}")

    def flush(self):
        self.__tb.flush()

    def close(self):
        self.__tb.close()
        handlers = self.__log.handlers[:]
        for handler in handlers:
            handler.close()
            self.__log.removeHandler(handler)

    """
    Python log
    """

    def log_warning(self, s):
        self.__log.warning(s)

    def log_info(self, s):
        self.__log.info(s)

    def log_debug(self, s):
        self.__log.debug(s)

    """
    Add to tensorboard
    """

    def add_scalar(self, key, val, step):
        self.__tb.add_scalar(key, val, step)

    def add_fig(self, key, fig, step, close_fig=False):
        """Convert a python fig to np.ndarry and add it to tensorboard"""
        fig.canvas.draw()
        im = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        im = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        im = im.transpose(2, 0, 1)  # (3, H, W)
        im = im.astype(np.float32) / 255.0
        self.add_im(key, im, step)

        if close_fig:
            plt.close(fig)

    def add_im(self, key, im, step):
        """Add an image to tensorboard. The image should be as np.ndarray
        https://tensorboardx.readthedocs.io/en/latest/tutorial.html#add-image
        """
        self.__tb.add_image(key, im, step)

    """
    Save to system
    """

    def get_save_dir(self, epoch, split):
        return os.path.join(self.__output_dir, split, f"e{epoch:06d}")

    def save_dict(self, dict_, file_name, epoch, split):
        """Save the dictionary to a pickle file. Single value items in the dictionary
        are stored in addition as a json file for easy inspection.
        """
        json_dict = {}
        for key, val in dict_.items():
            if not isinstance(val, (np.ndarray, tuple, list, dict)):
                json_dict[key] = str(val)

        save_dir = self.get_save_dir(epoch, split)
        json_fname = os.path.join(save_dir, f"{file_name}.json")
        os.makedirs(os.path.dirname(json_fname), exist_ok=True)
        with open(json_fname, "w") as fp:
            json.dump(json_dict, fp, sort_keys=True, indent=4)
        self.log_info(f"Dictonary saved to {json_fname}.")

        pickle_fname = os.path.join(save_dir, f"{file_name}.pkl")
        with open(pickle_fname, "wb") as fp:
            pickle.dump(dict_, fp, protocol=pickle.HIGHEST_PROTOCOL)
        self.log_info(f"Dictonary saved to {pickle_fname}.")

    def save_fig(self, fig, file_name, epoch, split, close_fig=True):
        fname = os.path.join(self.get_save_dir(epoch, split), f"{file_name}.png")
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        fig.savefig(fname)
        if close_fig:
            plt.close(fig)

    def save_file(self, file_str, file_name, epoch, split):
        fname = os.path.join(self.get_save_dir(epoch, split), f"{file_name}.txt")
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, "w") as f:
            f.write(file_str)

    """
    Save and load checkpoints
    """

    def save_ckpt(self, fname, model, optimizer, epoch, step):
        if not os.path.dirname(fname):
            fname = os.path.join(self.__ckpt_dir, fname)

        if model is not None:
            if isinstance(model, torch.nn.DataParallel):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
        else:
            model_state = None
        optim_state = optimizer.state_dict() if optimizer is not None else None

        ckpt_dict = {
            "epoch": epoch,
            "step": step,
            "model_state": model_state,
            "optimizer_state": optim_state,
        }
        torch.save(ckpt_dict, fname)
        self.log_info(f"Checkpoint saved to {fname}.")

    def load_ckpt(self, fname, model, optimizer=None):
        ckpt = torch.load(fname)
        epoch = ckpt["epoch"] if "epoch" in ckpt.keys() else 0
        step = ckpt["step"] if "step" in ckpt.keys() else 0

        model.load_state_dict(ckpt["model_state"])

        if optimizer is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])

        self.log_info(f"Load checkpoint {fname}: epoch {epoch}, step {step}.")

        return epoch, step

    def save_sigterm_ckpt(self, model, optimizer, epoch, step):
        """Save a checkpoint, which another process can use to continue the training,
        if the current process is terminated or preempted. This checkpoint should
        be saved in a process-agnoistic directory such that it can be located by
        both processes.
        """
        self.save_ckpt(self.__sigterm_ckpt, model, optimizer, epoch, step)

    def load_sigterm_ckpt(self, model, optimizer):
        return self.load_ckpt(self.__sigterm_ckpt, model, optimizer)

    def sigterm_ckpt_exists(self):
        return os.path.isfile(self.__sigterm_ckpt)

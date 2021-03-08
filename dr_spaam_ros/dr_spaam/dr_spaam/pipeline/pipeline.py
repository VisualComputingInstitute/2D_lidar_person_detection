from .optim import Optim
from .trainer import Trainer
from .logger import Logger


class Pipeline:
    def __init__(self, model, cfg):
        self.logger = Logger(cfg["Logger"])
        self.optim = Optim(model, cfg["Optim"])
        self.trainer = Trainer(self.logger, self.optim, cfg["Trainer"])
        self.logger.log_debug("Pipeline starts.")

    def close(self):
        self.logger.log_debug("Pipeline closes.")
        self.logger.close()

    def train(self, model, train_loader, eval_loader=None):
        self.logger.log_debug("Training starts.")
        status = self.trainer.train(model, train_loader, eval_loader)
        self.logger.log_debug(f"Training ends (status {status}).")
        return status

    def evaluate(self, model, eval_loader, tb_prefix):
        self.logger.log_debug("Evaluation starts.")
        status = self.trainer.evaluate(
            model, eval_loader, tb_prefix, plotting=False
        )
        self.logger.log_debug(f"Evaluation ends (status {status}).")
        return status

    def load_ckpt(self, model, ckpt, use_ckpt_epoch=False):
        epoch, step = self.logger.load_ckpt(ckpt, model, self.optim)
        # When finetuning a pre-trained checkpoint, we don't care the previous
        # training schedule, so not setting epoch and step
        if use_ckpt_epoch:
            self.trainer.set_epoch_step(epoch, step)

    def load_sigterm_ckpt(self, model):
        epoch, step = self.logger.load_sigterm_ckpt(model, self.optim)
        self.trainer.set_epoch_step(epoch, step)

    def sigterm_ckpt_exists(self):
        return self.logger.sigterm_ckpt_exists()

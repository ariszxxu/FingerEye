import os
import copy
import tqdm
import time
import wandb
import hydra
import torch
import pathlib
from termcolor import cprint
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from accelerate import Accelerator
from fingereye.utils.torch.common import dict_apply
from fingereye.utils.torch.common import report_parameters
from fingereye.workspaces.base_workspace import BaseWorkspace
from fingereye.utils.train.common import set_seed, set_print_formatting
from fingereye.utils.train.optimizer import get_scheduler, optimizer_to
from fingereye.utils.train.checkpoint_util import TopKCheckpointManager

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("eq", lambda val1, val2: val1 == val2)


class TrainWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.accelerator = Accelerator(mixed_precision='bf16')
        super().__init__(cfg, output_dir=output_dir)
        set_seed(cfg.training.seed)
        set_print_formatting()

        self.model = hydra.utils.instantiate(cfg.policy)
        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        report_parameters(self.model)
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()
        )

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        if cfg.training.resume:
            if cfg.eval_ckpt_path is not None and pathlib.Path(cfg.eval_ckpt_path).is_file():
                cprint(f"Resuming from checkpoint {cfg.eval_ckpt_path}", "blue")
                self.load_checkpoint(path=cfg.eval_ckpt_path)
            else:
                cprint(f"No valid checkpoint found at {cfg.eval_ckpt_path}. Starting fresh training.", "yellow")
            
        # configure dataset
        dataset = hydra.utils.instantiate(cfg.setting.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        self.model, self.optimizer, train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, train_dataloader
        )
        device = self.accelerator.device
        normalizer = dataset.get_normalizer()

        # set normalizer
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure ema
        ema = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
        self.ema = ema

        # configure lr scheduler
        num_steps_per_epoch = len(train_dataloader)
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(num_steps_per_epoch * cfg.training.num_epochs),
            last_epoch=self.global_step - 1,
        )
        # Prepare scheduler too
        self.lr_scheduler = self.accelerator.prepare(lr_scheduler)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        self.wandb_run = wandb_run
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            },
            allow_val_change=True,
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk
        )

        # device transfer
        self.device = device
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # Set total training steps if model supports it
        total_steps = len(train_dataloader) * cfg.training.num_epochs
        if hasattr(self.model, 'set_total_n_training_steps'):
            self.model.set_total_n_training_steps(total_steps)
            if self.ema_model is not None:
                self.ema_model.set_total_n_training_steps(total_steps)

        start_time = time.time()
        start_epoch = self.epoch
        for _ in range(start_epoch, cfg.training.num_epochs):
            step_log = dict()
            with tqdm.tqdm(
                train_dataloader,
                desc=f"Training epoch {self.epoch}",
                leave=False,
                mininterval=cfg.training.tqdm_interval_sec,
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    # get and del batch["dataset_idx"]
                    dataset_idx = batch.pop("dataset_idx")  # (B,)

                    # compute loss
                    loss_dict = self.model.compute_loss(batch, sim_batch=None)
                        
                    loss = loss_dict["loss"]
                    self.accelerator.backward(loss)

                    # step optimizer
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lr_scheduler.step()
                    # update ema
                    if cfg.training.use_ema:
                        ema.step(self.model)
                    # logging
                    tepoch.set_postfix(loss=loss.item(), refresh=False)
                    loss_dict_to_log = {}
                    for key, value in loss_dict.items():
                        if isinstance(value, torch.Tensor):
                            if value.numel() != 1:
                                continue
                            loss_dict_to_log[f"train_{key}"] = value.item()
                        else: 
                            loss_dict_to_log[f"train_{key}"] = value
                    step_log = {
                        "time": time.time() - start_time,
                        "step": self.global_step,
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                        "lr": lr_scheduler.get_last_lr()[0],
                        **loss_dict_to_log,
                    }
                    atten_weights = getattr(self.model, "atten_weights", None)
                    if atten_weights is not None:
                        step_log["atten_weights/weights"] = {
                            f"atten_weights{i}": atten_weights[i].item()
                            for i in range(atten_weights.shape[0])
                        }
                        
                    wandb_run.log(step_log, step=self.global_step)
                    self.global_step += 1


            # checkpoint
            if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0:

                # checkpointing
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint(tag=f"epoch_{self.epoch}")
                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()

                # sanitize metric names
                metric_dict = dict()
                for key, value in step_log.items():
                    new_key = key.replace("/", "_")
                    metric_dict[new_key] = value

                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                if topk_ckpt_path is not None:
                    self.save_checkpoint(path=topk_ckpt_path)

            # log of last step is combined with validation and rollout
            wandb_run.log(step_log, step=self.global_step)
            self.epoch += 1

    def run_eval(self):

        cfg = copy.deepcopy(self.cfg)

        cprint(f"Resuming from checkpoint {cfg.eval_ckpt_path}", "blue")
        self.load_checkpoint(path=cfg.eval_ckpt_path)

        # configure ema
        ema = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
        self.ema = ema

        # device transfer
        device = torch.device(cfg.training.device)
        self.device = device
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)

        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()

        return policy


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("configs")),
    config_name="fingereye.yaml",
)
def main(cfg):
    workspace = TrainWorkspace(
        cfg,
    )
    workspace.run()


if __name__ == "__main__":
    


    main()
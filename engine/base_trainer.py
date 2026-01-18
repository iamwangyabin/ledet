import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.network_factory import get_model


class BaseTrainer(nn.Module):
    def __init__(self, opt, logger=None):
        super().__init__()
        self.opt = opt
        self.model = get_model(opt)
        self.logger = logger
        self.global_step = 0
        self.current_epoch = 0
        self._log_buffer = {}

    def log(self, key, value, step=None):
        if isinstance(value, torch.Tensor):
            value = value.detach().float().item()
        self._log_buffer[key] = value
        if self.logger is not None:
            self.logger.log({key: value}, step=step)

    def training_step(self, batch):
        raise NotImplementedError

    def validation_step(self, batch):
        raise NotImplementedError

    def on_validation_epoch_end(self):
        return {}

    def configure_optimizers(self):
        optparams = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = self.opt.train.optimizer(optparams)
        scheduler = self.opt.train.scheduler(optimizer)
        return optimizer, scheduler

    def _is_distributed(self):
        return dist.is_available() and dist.is_initialized()

    def _rank(self):
        return dist.get_rank() if self._is_distributed() else 0

    def _world_size(self):
        return dist.get_world_size() if self._is_distributed() else 1

    def gather_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self._is_distributed():
            return tensor
        tensor = tensor.detach().cpu()
        gathered = [None for _ in range(self._world_size())]
        dist.all_gather_object(gathered, tensor)
        return torch.cat(gathered, dim=0)

    def _save_checkpoint(self, output_dir, filename, optimizer=None, scheduler=None, best_metric=None):
        if self._rank() != 0:
            return
        os.makedirs(output_dir, exist_ok=True)
        model_state = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
        state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model": model_state,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_metric": best_metric,
            "opt": self.opt,
        }
        torch.save(state, os.path.join(output_dir, filename))

    def fit(self, train_loader, val_loader=None, output_dir=None, monitor_key="val_ap_epoch", device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        if self._is_distributed():
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

        optimizer, scheduler = self.configure_optimizers()

        precision = str(self.opt.train.get("precision", "16")).lower()
        use_amp = precision in {"16", "fp16", "bf16"} and device.type == "cuda"
        amp_dtype = torch.bfloat16 if precision in {"bf16", "bfloat16"} else torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

        grad_accum = int(getattr(self.opt.train, "gradient_accumulation_steps", 1))
        max_epochs = int(self.opt.train.train_epochs)

        best_metric = None
        for epoch in range(max_epochs):
            self.current_epoch = epoch
            if self._is_distributed() and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            self.train()
            optimizer.zero_grad(set_to_none=True)
            last_step = -1
            for step, batch in enumerate(train_loader):
                last_step = step
                batch = tuple(item.to(device, non_blocking=True) for item in batch)
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    loss = self.training_step(batch)
                    loss = loss / grad_accum
                scaler.scale(loss).backward()

                if (step + 1) % grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                self.global_step += 1

            if (last_step + 1) % grad_accum != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

            metrics = {}
            if val_loader is not None and (epoch + 1) % int(self.opt.train.check_val_every_n_epoch) == 0:
                self.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        batch = tuple(item.to(device, non_blocking=True) for item in batch)
                        self.validation_step(batch)
                metrics = self.on_validation_epoch_end() or {}
                for key, value in metrics.items():
                    self.log(key, value, step=self.global_step)

            if output_dir is not None:
                if metrics and monitor_key in metrics:
                    metric_value = metrics[monitor_key]
                    if best_metric is None or metric_value > best_metric:
                        best_metric = metric_value
                        self._save_checkpoint(output_dir, "best.ckpt", optimizer, scheduler, best_metric)
                self._save_checkpoint(output_dir, "last.ckpt", optimizer, scheduler, best_metric)

        if self.logger is not None:
            self.logger.finish()


Trainer = BaseTrainer

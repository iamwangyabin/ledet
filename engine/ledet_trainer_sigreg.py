import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.validate import validate
from engine.base_trainer import BaseTrainer

class SIGReg(nn.Module):
    """Global SIGReg without class conditioning or standardization."""
    def __init__(self, knots: int = 17, t_max: float = 3.0, num_slices: int = 256):
        super().__init__()
        knots = int(knots)
        t_max = float(t_max)
        num_slices = int(num_slices)

        t = torch.linspace(0.0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)
        self.num_slices = num_slices
        self.eps = 1e-12


    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.dim() != 2:
            raise ValueError("proj must be (N,K).")
        if proj.size(0) < 2:
            return proj.new_zeros(())

        A = torch.randn(proj.size(-1), self.num_slices, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0, keepdim=True) + self.eps)
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

class Trainer_LEDet_SIGReg(BaseTrainer):
    def __init__(self, opt, logger=None):
        super().__init__(opt, logger=logger)
        self.validation_step_outputs_gts, self.validation_step_outputs_preds = [], []

        self.lambda_reg = float(getattr(self.opt.train, "lambda_reg", 1e-2))
        self.sigreg = None
        self.reg_cfg = {
            "knots": int(getattr(self.opt.train, "sigreg_knots", 17)),
            "t_max": float(getattr(self.opt.train, "sigreg_t_max", 3.0)),
            "num_slices": int(getattr(self.opt.train, "sigreg_num_slices", 256)),
        }

    def _binary_logit(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 2 and logits.size(1) == 2:
            return logits[:, 1] - logits[:, 0]
        if logits.dim() == 2 and logits.size(1) == 1:
            return logits.squeeze(1)
        return logits.squeeze(-1)

    def _init_sigreg(self, device: torch.device, dtype: torch.dtype):
        self.sigreg = SIGReg(
            knots=self.reg_cfg["knots"],
            t_max=self.reg_cfg["t_max"],
            num_slices=self.reg_cfg["num_slices"],
        ).to(device=device, dtype=dtype)

    def training_step(self, batch):
        x, y = batch
        y_bin = (y % 2).long()

        output = self.model(x)

        logits = output["logits"]
        z = output["features"]

        if self.sigreg is None:
            self._init_sigreg(z.device, z.dtype)

        bin_logit = self._binary_logit(logits)
        sup = F.binary_cross_entropy_with_logits(bin_logit, y_bin.float())
        reg = self.sigreg(z)

        loss = sup + self.lambda_reg * reg
        self.log("train_loss", loss, step=self.global_step)
        self.log("train_sup", sup, step=self.global_step)
        self.log("train_sigreg", reg, step=self.global_step)
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_bin = (y % 2).long()

        output = self.model(x)

        logits = output["logits"]
        bin_logit = self._binary_logit(logits)
        self.validation_step_outputs_preds.append(bin_logit)
        self.validation_step_outputs_gts.append(y_bin)

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_step_outputs_preds, 0).to(torch.float32).sigmoid().flatten()
        all_gts = torch.cat(self.validation_step_outputs_gts, 0).to(torch.float32)
        all_preds = self.gather_tensor(all_preds).cpu().numpy()
        all_gts = self.gather_tensor(all_gts).cpu().numpy()
        acc, ap, r_acc, f_acc = validate(all_gts, all_preds)
        metrics = {
            "val_acc_epoch": acc,
            "val_ap_epoch": ap,
            "val_racc_epoch": r_acc,
            "val_facc_epoch": f_acc,
        }
        self.validation_step_outputs_preds.clear()
        self.validation_step_outputs_gts.clear()
        return metrics

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.validate import validate
from engine.base_trainer import BaseTrainer


def _trapz(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    dx = x[1:] - x[:-1]
    return (0.5 * (y[..., 1:] + y[..., :-1]) * dx).sum(dim=-1)


class EppsPulley(nn.Module):
    """Univariate CF distance to N(0,1)."""

    def __init__(self, num_points: int = 17, t_max: float = 5.0, sigma: float = 1.0):
        super().__init__()
        self.num_points = int(num_points)
        self.t_max = float(t_max)
        self.sigma = float(sigma)
        t = torch.linspace(-self.t_max, self.t_max, steps=self.num_points)
        self.register_buffer("t_grid", t, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("x must be (N,S).")
        N, _ = x.shape
        if N < 2:
            return x.new_zeros((x.shape[1],))

        t = self.t_grid.to(device=x.device, dtype=torch.float32)
        x_fp32 = x.to(dtype=torch.float32)
        xt = x_fp32[..., None] * t[None, None, :]
        phi_hat = torch.exp(1j * xt).mean(dim=0)

        phi_target = torch.exp(-0.5 * t**2)[None, :].to(phi_hat.dtype)
        w = torch.exp(-(t**2) / (2.0 * (self.sigma**2)))[None, :]

        diff2 = (phi_hat - phi_target).abs().pow(2)
        return _trapz(w * diff2, t)


class SIGReg(nn.Module):
    """Multivariate SIGReg via random slicing + univariate EP."""

    def __init__(self, num_slices: int = 64, ep_num_points: int = 17, t_max: float = 5.0, sigma: float = 1.0):
        super().__init__()
        self.num_slices = int(num_slices)
        self.ep = EppsPulley(num_points=ep_num_points, t_max=t_max, sigma=sigma)
        self.eps = 1e-12

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError("z must be (N,K).")
        N, K = z.shape
        if N < 2:
            return z.new_zeros(())

        A = torch.randn(self.num_slices, K, device=z.device, dtype=z.dtype)
        A = A / (A.norm(dim=1, keepdim=True) + self.eps)
        u = z @ A.t()

        per_slice = self.ep(u)
        return per_slice.mean()


class ClassConditionalGaussianSIGReg(nn.Module):
    """Class-conditional Gaussian standardization + SIGReg."""

    def __init__(
        self,
        feat_dim: int,
        num_classes: int = 2,
        ema_momentum: float = 0.99,
        num_slices: int = 64,
        ep_num_points: int = 17,
        t_max: float = 5.0,
        sigma: float = 1.0,
        min_count: int = 8,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.K = int(feat_dim)
        self.C = int(num_classes)
        self.m = float(ema_momentum)
        self.min_count = int(min_count)
        self.eps = float(eps)

        self.register_buffer("mu", torch.zeros(self.C, self.K), persistent=True)
        self.register_buffer("var", torch.ones(self.C, self.K), persistent=True)

        self.sigreg = SIGReg(num_slices=num_slices, ep_num_points=ep_num_points, t_max=t_max, sigma=sigma)

    @torch.no_grad()
    def update_stats(self, z: torch.Tensor, y: torch.Tensor):
        for c in range(self.C):
            mask = (y == c)
            n = int(mask.sum().item())
            if n < self.min_count:
                continue
            zc = z[mask]
            batch_mu = zc.mean(dim=0)
            batch_var = zc.var(dim=0, unbiased=False)

            self.mu[c].mul_(self.m).add_(batch_mu * (1.0 - self.m))
            self.var[c].mul_(self.m).add_(batch_var * (1.0 - self.m))

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        reg = z.new_zeros(())
        for c in range(self.C):
            mask = (y == c)
            if not mask.any():
                continue
            zc = z[mask]
            std = torch.sqrt(self.var[c] + self.eps)
            zc_std = (zc - self.mu[c]) / std
            reg = reg + self.sigreg(zc_std)
        return reg


class Trainer_LEDet(BaseTrainer):
    def __init__(self, opt, logger=None):
        super().__init__(opt, logger=logger)
        self.validation_step_outputs_gts, self.validation_step_outputs_preds = [], []

        self.lambda_reg = float(getattr(self.opt.train, "lambda_reg", 1e-2))
        self.cc_reg = None

        self.reg_cfg = {
            "ema_momentum": float(getattr(self.opt.train, "ema_momentum", 0.99)),
            "num_slices": int(getattr(self.opt.train, "num_slices", 64)),
            "ep_num_points": int(getattr(self.opt.train, "ep_num_points", 17)),
            "t_max": float(getattr(self.opt.train, "t_max", 5.0)),
            "sigma": float(getattr(self.opt.train, "sigma", 1.0)),
            "min_count": int(getattr(self.opt.train, "min_count", 8)),
            "eps": float(getattr(self.opt.train, "eps", 1e-6)),
        }

    def _binary_logit(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 2 and logits.size(1) == 2:
            return logits[:, 1] - logits[:, 0]
        if logits.dim() == 2 and logits.size(1) == 1:
            return logits.squeeze(1)
        return logits.squeeze(-1)

    def _init_cc_reg(self, feat_dim: int, device: torch.device, dtype: torch.dtype):
        self.cc_reg = ClassConditionalGaussianSIGReg(
            feat_dim=feat_dim,
            num_classes=2,
            ema_momentum=self.reg_cfg["ema_momentum"],
            num_slices=self.reg_cfg["num_slices"],
            ep_num_points=self.reg_cfg["ep_num_points"],
            t_max=self.reg_cfg["t_max"],
            sigma=self.reg_cfg["sigma"],
            min_count=self.reg_cfg["min_count"],
            eps=self.reg_cfg["eps"],
        ).to(device=device, dtype=dtype)

    def training_step(self, batch):
        x, y = batch
        y_bin = (y % 2).long()

        if hasattr(self.model, "forward_binary"):
            output = self.model.forward_binary(x)
        else:
            output = self.model(x)

        logits = output["logits"]
        z = output.get("features", output.get("z", None))

        if self.cc_reg is None:
            self._init_cc_reg(z.shape[1], z.device, z.dtype)

        bin_logit = self._binary_logit(logits)
        sup = F.binary_cross_entropy_with_logits(bin_logit, y_bin.float())

        self.cc_reg.update_stats(z.detach(), y_bin)
        reg = self.cc_reg(z, y_bin)

        loss = sup + self.lambda_reg * reg
        self.log("train_loss", loss, step=self.global_step)
        self.log("train_sup", sup, step=self.global_step)
        self.log("train_cc_sigreg", reg, step=self.global_step)
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_bin = (y % 2).long()

        if hasattr(self.model, "forward_binary"):
            output = self.model.forward_binary(x)
        else:
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

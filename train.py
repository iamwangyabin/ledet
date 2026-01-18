import os
import hydra
import argparse
import datetime

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler

import engine
import data
import networks
from utils.common import load_config_with_cli, archive_files, seed_everything


import swanlab


class SwanLabLogger:
    def __init__(self, name, project, job_type=None, group=None):
        self.name = name
        self.project = project
        self.job_type = job_type
        self.group = group
        self._init_run()

    def _init_run(self):
        init_kwargs = {"project": self.project, "name": self.name}
        try:
            swanlab.init(**init_kwargs)
        except TypeError:
            swanlab.init(project=self.project)

    def log(self, metrics, step=None):
        if step is not None:
            swanlab.log(metrics, step=step)
        else:
            swanlab.log(metrics)

    def finish(self):
        swanlab.finish()


def build_dataloader(conf, distributed=False):
    train_datasets = []
    for sub_data in conf.datasets.train.source:
        for sub_set in sub_data.sub_sets:
            train_data = eval(sub_data.target)(sub_data.data_root, conf.datasets.train.trsf,
                                               subset=sub_set, split=sub_data.split)
            train_datasets.append(train_data)
    train_datasets = ConcatDataset(train_datasets)

    val_datasets = []
    for sub_data in conf.datasets.val.source:
        for sub_set in sub_data.sub_sets:
            val_data = eval(sub_data.target)(sub_data.data_root, conf.datasets.val.trsf,
                                          subset=sub_set, split=sub_data.split)
            val_datasets.append(val_data)
    val_datasets = ConcatDataset(val_datasets)


    train_sampler = DistributedSampler(train_datasets, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_datasets, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_datasets,
        batch_size=conf.datasets.train.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=conf.datasets.train.loader_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_datasets,
        batch_size=conf.datasets.val.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=conf.datasets.val.loader_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return True, local_rank, rank, world_size
    return False, 0, 0, 1


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--cfg', type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()
    conf = load_config_with_cli(args.cfg, args_list=cfg_args)
    conf = hydra.utils.instantiate(conf)

    seed_everything(conf.train.seed)

    distributed, local_rank, rank, world_size = init_distributed()
    train_loader, val_loader = build_dataloader(conf, distributed=distributed)
    today_str = conf.name +"_"+ datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
    swanlab_logger = None
    if rank == 0:
        swanlab_logger = SwanLabLogger(name=today_str, project='DeepfakeDetection',
                                       job_type='train', group=conf.name)


    if os.getenv("LOCAL_RANK", '0') == '0':
        archive_files(today_str, exclude_dirs=['logs', 'swanlab', '.git', 'exp_results'])

    model = eval(conf.train.pipeline)(opt=conf, logger=swanlab_logger)
    torch.set_float32_matmul_precision('high')
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join('logs', today_str)
    model.fit(train_loader=train_loader, val_loader=val_loader, output_dir=output_dir, device=device)
    cleanup_distributed()

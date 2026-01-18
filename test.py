import argparse
import csv
import pickle

import hydra
import torch
import torch.utils.data

import data
import utils
from utils.common import load_config_with_cli
from utils.network_factory import get_model


def _resolve_test_datasets(conf):
    if hasattr(conf.datasets, "test"):
        return conf.datasets.test
    if hasattr(conf.datasets, "val"):
        return conf.datasets.val
    return conf.datasets


def _resolve_resume(conf, resume_path_override=None, resume_target_override=None):
    resume_cfg = getattr(conf, "resume", None)
    resume_path = resume_path_override
    resume_target = resume_target_override

    if resume_cfg is not None:
        if resume_path is None and hasattr(resume_cfg, "path"):
            resume_path = resume_cfg.path
        if resume_target is None and hasattr(resume_cfg, "target"):
            resume_target = resume_cfg.target

    if resume_path is None:
        return None, None
    if resume_target is None:
        resume_target = "utils.resume_tools.resume_checkpoint"
    return resume_target, resume_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--cfg', type=str, default=None, required=True)
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume for testing')
    parser.add_argument('--resume_target', type=str, default=None, help='Resume function path, e.g. utils.resume_tools.resume_checkpoint')
    args, cfg_args = parser.parse_known_args()
    conf = load_config_with_cli(args.cfg, args_list=cfg_args)
    conf = hydra.utils.instantiate(conf)

    model = get_model(conf)
    resume_target, resume_path = _resolve_resume(conf, args.resume, args.resume_target)
    if resume_path is not None:
        eval(resume_target)(model, resume_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_results = []
    save_raw_results = {}

    datasets_cfg = _resolve_test_datasets(conf)
    for sub_data in datasets_cfg.source:
        for sub_set in sub_data.sub_sets:
            dataset = eval(sub_data.target)(
                sub_data.data_root,
                datasets_cfg.trsf,
                subset=sub_set,
                split=sub_data.split,
            )
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=datasets_cfg.batch_size,
                num_workers=datasets_cfg.loader_workers,
                shuffle=False,
                pin_memory=True,
            )

            eval_pipeline = getattr(conf, "eval_pipeline", "utils.validate.validate_plain")
            result = eval(eval_pipeline)(model, data_loader)

            ap = result['ap']
            auc = result['auc']
            f1 = result['f1']
            r_acc0 = result['r_acc0']
            f_acc0 = result['f_acc0']
            acc0 = result['acc0']
            num_real = result['num_real']
            num_fake = result['num_fake']

            benchmark_name = getattr(sub_data, "benchmark_name", sub_data.data_root)
            print(f"{benchmark_name} {sub_set}")
            print(
                f"AP: {ap:.4f},\tF1: {f1:.4f},\tAUC: {auc:.4f},\tACC: {acc0:.4f},\t"
                f"R_ACC: {r_acc0:.4f},\tF_ACC: {f_acc0:.4f}"
            )
            all_results.append([
                benchmark_name,
                sub_set,
                ap,
                auc,
                f1,
                r_acc0,
                f_acc0,
                acc0,
                num_real,
                num_fake,
            ])
            save_raw_results[f"{benchmark_name} {sub_set}"] = result

    test_name = getattr(conf, "test_name", conf.name if hasattr(conf, "name") else "test")
    columns = ['dataset', 'sub_set', 'ap', 'auc', 'f1', 'r_acc0', 'f_acc0', 'acc0', 'num_real', 'num_fake']
    with open(test_name + '_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        for values in all_results:
            writer.writerow(values)
    with open(test_name + '.pkl', 'wb') as file:
        pickle.dump(save_raw_results, file)

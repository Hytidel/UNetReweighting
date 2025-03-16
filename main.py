from util.logger import logger

from typing import Optional, Tuple, Union, Dict, List, Set, Callable, TypeVar, Generic, NewType, Protocol

import hydra
from omegaconf import DictConfig, OmegaConf
import yaml

from util.basic_util import (
    pause, 
    set_global_variable_dict, 
    get_global_variable, set_global_variable
)


def test(
    cfg: DictConfig
):
    from task.test import test
    test(cfg)

    # `test()` done
    pass

def sample(
    cfg: DictConfig
):
    task_name = cfg["task"]["name"]

    if task_name == "sample-t2i-sd_family":
        from task.sample.t2i.sd_family import sample_sd_family
        sample_sd_family(cfg)
    elif task_name == "sample-t2i-run_sd_family":
        from task.sample.t2i.run_sd_family import sample_run_sd_family
        sample_run_sd_family(cfg)
    elif task_name == "sample-t2i-run_scheduled_sd_family":
        from task.sample.t2i.run_scheduled_sd_family import sample_run_scheduled_sd_family
        sample_run_scheduled_sd_family(cfg)

    elif task_name == "sample-t2i-sdxl_family":
        from task.sample.t2i.sdxl_family import sample_sdxl_family
        sample_sdxl_family(cfg)
    elif task_name == "sample-t2i-run_sdxl_family":
        from task.sample.t2i.run_sdxl_family import sample_run_sdxl_family
        sample_run_sdxl_family(cfg)
    elif task_name == "sample-t2i-run_scheduled_sdxl_family":
        from task.sample.t2i.run_scheduled_sdxl_family import sample_run_scheduled_sdxl_family
        sample_run_scheduled_sdxl_family(cfg)

    elif task_name == "sample-ddim_inversion-sd_family":
        from task.sample.ddim_inversion.sd_family import sample_scheduled_ddim_inversion_sd_family
        sample_scheduled_ddim_inversion_sd_family(cfg)
    elif task_name == "sample-ddim_inversion-sdxl_family":
        from task.sample.ddim_inversion.sdxl_family import sample_scheduled_ddim_inversion_sdxl_family
        sample_scheduled_ddim_inversion_sdxl_family(cfg)

    else:
        raise NotImplementedError(
            f"Unsupported task: `{task_name}`. "
        )

    # `sample()` done
    pass

def do_importance_probe(
    cfg: DictConfig
):
    task_name = cfg["task"]["name"]

    if task_name == "do_importance_probe-t2i-sd_family":
        from task.do_importance_probe.t2i.sd_family import do_importance_probe_sd_family
        do_importance_probe_sd_family(cfg)
    elif task_name == "do_importance_probe-t2i-run_sd_family":
        from task.do_importance_probe.t2i.run_sd_family import do_importance_probe_run_sd_family
        do_importance_probe_run_sd_family(cfg)

    elif task_name == "do_importance_probe-t2i-sdxl_family":
        from task.do_importance_probe.t2i.sdxl_family import do_importance_probe_sdxl_family
        do_importance_probe_sdxl_family(cfg)
    elif task_name == "do_importance_probe-t2i-run_sdxl_family":
        from task.do_importance_probe.t2i.run_sdxl_family import do_importance_probe_run_sdxl_family
        do_importance_probe_run_sdxl_family(cfg)
    
    elif task_name == "do_importance_probe-ddim_inversion-sd_family":
        from task.do_importance_probe.ddim_inversion.sd_family import do_importance_probe_ddim_inversion_sd_family
        do_importance_probe_ddim_inversion_sd_family(cfg)
    elif task_name == "do_importance_probe-ddim_inversion-sdxl_family":
        from task.do_importance_probe.ddim_inversion.sdxl_family import do_importance_probe_ddim_inversion_sdxl_family
        do_importance_probe_ddim_inversion_sdxl_family(cfg)

    else:
        raise NotImplementedError(
            f"Unsupported task: `{task_name}`. "
        )

    # `do_importance_probe()` done
    pass

def cal_importance(
    cfg: DictConfig
):
    task_name = cfg["task"]["name"]

    if task_name == "cal_importance-cal_importance_ranking":
        from task.cal_importance_ranking.cal_importance_ranking import cal_importance_ranking
        cal_importance_ranking(cfg)
    elif task_name == "cal_importance-run_cal_importance_ranking":
        from task.cal_importance_ranking.run_cal_importance_ranking import run_cal_importance_ranking
        run_cal_importance_ranking(cfg)
    elif task_name == "cal_importance-cal_importance_weight":
        from task.cal_importance_weight.cal_importance_weight import cal_importance_weight
        cal_importance_weight(cfg)
    elif task_name == "cal_importance-run_cal_importance_weight":
        from task.cal_importance_weight.run_cal_importance_weight import run_cal_importance_weight
        run_cal_importance_weight(cfg)
    
    else:
        raise NotImplementedError(
            f"Unsupported task: `{task_name}`. "
        )

    # `cal_importance()` done
    pass

def do_pruning(
    cfg: DictConfig
):
    task_name = cfg["task"]["name"]

    if task_name.startswith("do_pruning-sd_family"):
        from task.do_pruning.sd_family import do_pruning_sd_family
        do_pruning_sd_family(cfg)
    elif task_name.startswith("do_pruning-sdxl_family"):
        from task.do_pruning.sdxl_family import do_pruning_sdxl_family
        do_pruning_sdxl_family(cfg)

    else:
        raise NotImplementedError(
            f"Unsupported task: `{task_name}`. "
        )

    # `do_pruning()` done
    pass

def cal_metric(
    cfg: DictConfig
):
    task_name = cfg["task"]["name"]

    if task_name == "cal_metric-run_cal_hpsv2":
        from task.cal_metric.run_cal_hpsv2 import run_cal_hpsv2
        run_cal_hpsv2(cfg)
    elif task_name == "cal_metric-run_cal_fid_lpips":
        from task.cal_metric.run_cal_fid_lpips import run_cal_fid_lpips
        run_cal_fid_lpips(cfg)

    else:
        raise NotImplementedError(
            f"Unsupported task: `{task_name}`. "
        )

    # `cal_metric()` done
    pass

def plot(
    cfg: DictConfig
):
    task_name = cfg["task"]["name"]

    if task_name == "plot-bar_chart_voting_score":
        from task.plot.bar_chart_voting_score import bar_chart_voting_score
        bar_chart_voting_score(cfg)
    elif task_name == "plot-heatmap_importance_score":
        from task.plot.heatmap_importance_score import heatmap_importance_score
        heatmap_importance_score(cfg)
    elif task_name == "plot-scatter_fid_lpips":
        from task.plot.scatter_fid_lpips import scatter_fid_lpips
        scatter_fid_lpips(cfg)

    else:
        raise NotImplementedError(
            f"Unsupported task: `{task_name}`. "
        )

    # `plot()` donescatter_fid_lpips
    pass

def run_task(
    cfg: DictConfig
):
    task_name = cfg["task"]["name"]

    if task_name.startswith("sample"):
        sample(cfg)
    elif task_name.startswith("do_importance_probe"):
        do_importance_probe(cfg)
    elif task_name.startswith("cal_importance"):
        cal_importance(cfg)
    elif task_name.startswith("do_pruning"):
        do_pruning(cfg)
    elif task_name.startswith("cal_metric"):
        cal_metric(cfg)
    elif task_name.startswith("plot"):
        plot(cfg)

    elif task_name.startswith("test"):
        test(cfg)
    else:
        raise NotImplementedError(
            f"Unsupported task: `{task_name}`. "
        )

@hydra.main(version_base = None, config_path = "config", config_name = "cfg")
def main(
    cfg: DictConfig
):
    cfg = OmegaConf.create(cfg)
    cfg = OmegaConf.to_container(
        cfg, 
        resolve = True
    )

    set_global_variable_dict(cfg)

    exp_name = get_global_variable("exp_name")
    logger(f"Start experiment `{exp_name}`. ")

    run_task(cfg)

    logger(f"Experiment `{exp_name}` finished. ")

    # `main()` done
    pass

if __name__ == "__main__":
    main()
    
import hydra
from omegaconf import DictConfig, OmegaConf

from baseline import run_baseline
from dataloaders import test
from utils import dump_gpu_info


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    for task in cfg.tasks:
        print(task)
        if task == 'baseline':
            run_baseline(cfg.baseline)
        elif task == 'dataloader_test':
            test()
        elif task == 'dump_gpu_info':
            dump_gpu_info()


if __name__ == "__main__":
    main()
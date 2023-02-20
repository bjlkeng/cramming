import hydra
from omegaconf import DictConfig, OmegaConf

from dataloaders import test


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    if cfg.main.task == 'dataloader_test':
        test()



if __name__ == "__main__":
    main()
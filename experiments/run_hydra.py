import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import stable_pretraining

# train.py gets this from importing stable_pretraining; register it here so ${eval:...} works
# if not OmegaConf.has_resolver("eval"):
#     OmegaConf.register_new_resolver("eval", eval)

@hydra.main(
    version_base=None,
    config_path="/home/taot/github/le-wm/config/train",
    config_name="lewm",
)
def run(cfg: DictConfig) -> None:
    # OmegaConf.resolve(cfg)
    # print(OmegaConf.to_yaml(cfg))
    # print(OmegaConf.to_container(cfg))
    print(cfg.run_name)

if __name__ == "__main__":
    run()

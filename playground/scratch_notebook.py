import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    from pathlib import Path
    from dotenv import load_dotenv

    # STABLEWM_HOME (storage root) from the repo's .env, same as train.py.
    load_dotenv("/home/taot/github/le-wm/.env")

    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import DictConfig, OmegaConf, open_dict
    import stable_pretraining  # noqa: F401  registers the ${eval:...} OmegaConf resolver
    import stable_worldmodel as swm

    config_dir = str(Path("/home/taot/github/le-wm/config/train"))

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="lewm")

    print(cfg.model._target_)   # jepa.JEPA
    return OmegaConf, cfg, swm


@app.cell
def _(OmegaConf, cfg, swm):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    dataset = swm.data.load_dataset(dataset_name, transform=None, **dataset_cfg)
    return (dataset,)


@app.cell
def _(dataset):
    dataset[0]
    return


if __name__ == "__main__":
    app.run()

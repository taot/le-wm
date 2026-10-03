import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import hydra
    import json
    from omegaconf import DictConfig, OmegaConf, open_dict

    return DictConfig, OmegaConf, hydra


@app.cell
def _(DictConfig, OmegaConf, hydra):
    @hydra.main(
        version_base=None,
        config_path="/home/taot/github/le-wm/config/train",
        config_name="lewm",
    )
    def run(cfg: DictConfig) -> None:
        # print(json.dumps(cfg))
        print(OmegaConf.to_yaml(cfg))

    return (run,)


@app.cell
def _(run):
    run()
    return


@app.cell
def _(OmegaConf):
    conf = OmegaConf.load('/home/taot/github/le-wm/config/train/lewm.yaml')
    print(conf)
    return


if __name__ == "__main__":
    app.run()

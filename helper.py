from hydra import compose,initialize
def load_config():
    with initialize(version_base=None,config_path="../conf"):
        config=compose(config_name="config")
    return config

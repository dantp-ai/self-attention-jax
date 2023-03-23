from hydra import compose
from hydra import initialize
from hydra.utils import instantiate

initialize(version_base=None, config_path="../conf")

cfg = instantiate(compose(config_name="config"))

model_config = cfg.model

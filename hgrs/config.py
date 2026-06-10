import importlib_resources
import yaml


def get_config():
    configfile = importlib_resources.files(__package__).joinpath("config.yml")
    with open(configfile, "r") as file:
        config = yaml.safe_load(file)
    return config

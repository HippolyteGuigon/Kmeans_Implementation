import yaml
import os

def load_conf(path: str):
    """
    The goal of this function is loading
    a configuration file from a path given
    Arguments:
    path: str: The path of the configuration file
    """
    path=os.path.join(os.getcwd(),path)
    with open(path, "r") as ymlfile:
        configs = yaml.safe_load(ymlfile)

    return configs

import yaml
<<<<<<< HEAD

def load_conf(path:str):
    with open(path, "r") as ymlfile:
        file = yaml.safe_load(ymlfile)
    return file
=======
import os

def load_conf(path: str):
    """
    The goal of this function is loading
    a configuration file from a path given
    Arguments:
    path: str: The path of the configuration file
    """
    path=os.path.join(os.getcwd(),path)
    print("ICIIII",path)
    with open(path, "r") as ymlfile:
        configs = yaml.safe_load(ymlfile)

    return configs
>>>>>>> 17fd1cea6ba916a56f4bc966475afc2d4815fd5a

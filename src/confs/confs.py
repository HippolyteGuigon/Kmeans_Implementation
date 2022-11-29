import yaml

def load_conf(path:str)->yaml:
    """
    The goal of this function is to load the 
    params 

    Arguments: 
        path: str The path of configuration file to be loaded 

    Returns: 
        file: yaml file The configuration file loaded
    """
    with open(path, "r") as ymlfile:
        file = yaml.safe_load(ymlfile)
    return file

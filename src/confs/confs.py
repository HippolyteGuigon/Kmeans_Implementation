import yaml

def load_conf(path:str):
    with open(path, "r") as ymlfile:
        file = yaml.safe_load(ymlfile)
    return file
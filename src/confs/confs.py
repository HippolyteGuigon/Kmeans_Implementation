import yaml


def load_conf(path: str) -> yaml:
    """
    The goal of this function is to load the
    params
    Arguments:
        -path: str The path of configuration file to be loaded
    Returns:
        -file: yaml file The configuration file loaded
    """
    with open(path, "r") as ymlfile:
        file = yaml.safe_load(ymlfile)
    return file

def load_default_params(dict_params:dict(),path_default_params="configs/default_params.yml")->dict():
    """
    The goal of this function is, for all parameters of the 
    KMeans not entered by the user, to enter instead default 
    params stored in a yml file 
    
    Arguments:
        -dict_params: dict(): The dictionnary containing the 
        parameters entered by the user 
        -path_default_params: The path of the yml file where the 
        default parameters are stored
        
    Returns:
        -dict_params: dict(): The dictionnary of parameters after it
        was updated with default parameters"""

    
    default_params=load_conf(path_default_params)
    for key in default_params.keys():
        if key not in dict_params.keys():
            dict_params[key]=default_params[key]

    return dict_params
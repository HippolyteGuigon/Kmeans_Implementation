import yaml
import os
import numpy as np
import ruamel.yaml

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

model_params=load_conf("configs/model_params.yml")

def load_default_params(
    dict_params: dict(), path_default_params="configs/default_params.yml"
) -> dict():
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
        was updated with default parameters
    """

    default_params = load_conf(path_default_params)
    for key in default_params.keys():
        if key not in dict_params.keys():
            dict_params[key] = default_params[key]

    return dict_params


def updating_parameter(dict_conf, points) -> None:
    """
    The goal of this function is, if the user decides to cluster its own
    data, to modify the yml file of the model so that the pipeline can run
    without problem and appropriate parameters can be applicated

    Arguments:
        -dict_conf: dict(str): The current dictionnary with the parameters
        of the yml file
        -points: np.array(float): The points entered by the user to be clustered

    Returns:
        None
    """
    data = np.load(os.path.join(os.getcwd(), model_params["to_cluster_data_path"]))
    yaml = ruamel.yaml.YAML()
    with open(os.path.join(os.getcwd(), model_params["path_config_file"])) as fp:
        data = yaml.load(fp)
    data["number_dimension"] = points.shape[1]
    with open(model_params["path_config_file"], "w") as file:
        documents = yaml.dump(data, file)
    dict_conf["number_of_individuals"] = points.shape[0]
    dict_conf["number_dimension"] = points.shape[1]
    dict_conf["limit_min"] = points.min()
    dict_conf["limit_max"] = points.max()

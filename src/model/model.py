# As a user, I want to have my random-generated data returned
# As a user, I want to have K points generated within the plan
# As a user, I want every point to be assigned to the nearest cluster point

import numpy as np
import sys
import yaml
import os

sys.path.insert(0, os.path.join(os.getcwd(), "src/data_generator"))

from data_generator import Data_Generator


class Model(Data_Generator):
    """
    The goal of this class is to implement the KMeans
    algorithm step by step
    """

    def __init__(
        self,
        path_config_model="configs/model_params.yml",
        path_config_file="configs/data_params.yml",
    ):
        with open(path_config_model, "r") as ymlfile:
            configs_model = yaml.safe_load(ymlfile)

        with open(path_config_file, "r") as ymlfile:
            configs_file = yaml.safe_load(ymlfile)

        self.configs = configs_file
        self.configs_model = configs_model
        self.K = self.configs_model["K"]
        self.data = super().generate_data()

    def generate_initial_K(self):
        """
        The goal of this function is to initialize K centroids
        randomly that will be then used to allocate points between
        the different clusters
        """
        lim_min = self.configs["limit_min"]
        lim_max = self.configs["limit_max"]
        initial_cluster_coordinates = np.random.uniform(
            low=lim_min, high=lim_max, size=(self.K, self.configs["number_dimension"])
        )
        self.initial_coordinates = initial_cluster_coordinates
        return self.initial_coordinates


a = Model()
a.generate_initial_K()

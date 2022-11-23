import numpy as np
import sys
import os
from sklearn.metrics.pairwise import pairwise_distances
from src.confs.confs import load_conf

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

        self.configs = load_conf(path_config_file)
        self.configs_model = load_conf(path_config_model)
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

    def compute_distances(self):
        distances = pairwise_distances(self.data, self.initial_coordinates)
        cluster_belonging = np.argmin(distances, axis=1)
        full_data = np.hstack(self.initial_coordinates, cluster_belonging)
        np.save(self.configs_model["save_path"], full_data)
        return cluster_belonging


a = Model()
a.generate_initial_K()
a.compute_distances()

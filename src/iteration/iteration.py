import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), "src/confs"))
from confs import load_conf
from scipy import spatial


class Generate_Region:
    def __init__(
        self,
        path_config_model="configs/model_params.yml",
        path_config_file="configs/data_params.yml",
    ):

        self.configs = load_conf(path_config_file)
        self.configs_model = load_conf(path_config_model)

    def initiate_region_points(self):
        """
        Generates the data randomly according to the configs
        file
        """
        n_rows = 1000 * self.configs["number_of_individuals"]
        n_columns = self.configs["number_dimension"]
        lim_min = self.configs["limit_min"]
        lim_max = self.configs["limit_max"]
        data_generated = np.random.uniform(
            low=lim_min, high=lim_max, size=(n_rows, n_columns)
        )
        return data_generated

    def compute_centroid(self, data_points):
        """
        The goal of this function is to determine  each cluster zone of
        influence that will then be used to compute the centroids. Points are generated
        randomly and attributed to nearest cluster's appartenance points.
        Regions emerging from this operation are then used to compute centroid

        Args:
        data_points: np.array(): The original points with attributed clusters at a cetrain step

        Return:
        np.array() with the centroids of each clusters"""
        data_region = self.initiate_region_points()
        clusters = data_points[:, -1]
        data_points = data_points[:, :-1]
        closest_points = spatial.KDTree(data_points).query(data_region)[1]
        attributed_regions = clusters[closest_points]
        data_region = np.column_stack((data_region, attributed_regions))
        unique_centroids = np.unique(clusters)
        return data_region

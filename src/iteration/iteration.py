import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), "src/confs"))
from confs import load_conf
from scipy import spatial
import json

class Generate_Region:
    """
    The goal of this class is, at each step of the KMeans algorithm,
    after points are attributed to the nearest cluster, to determine
    the influence region of each cluster and then compute the barycenter
    of each of these regions.
    """

    def __init__(
        self,
        path_config_model="configs/model_params.yml",
        path_config_file="configs/data_params.yml",
        path_final_params="configs/final_params.json"
    ):

        self.configs = load_conf(path_config_file)
        self.configs_model = load_conf(path_config_model)
        f=open(path_final_params)
        self.final_params=json.load(f)
    def initiate_region_points(self) -> np.array(float):
        """
        Generates the data randomly according to the configs
        file with a uniform law.

        Arguments:
            None

        Returns:
            -data_generated : np.array(float) : Set of points
            that will be used to determine the centroids of
            the regions
        """
        n_rows = 1000 * self.configs["number_of_individuals"]
        n_columns = self.configs["number_dimension"]
        lim_min = self.configs["limit_min"]
        lim_max = self.configs["limit_max"]
        data_generated = np.random.uniform(
            low=lim_min, high=lim_max, size=(n_rows, n_columns)
        )
        return data_generated

    def compute_centroid(self, data_points) -> np.array(float):
        """
        The goal of this function is to determine  each cluster zone of
        influence that will then be used to compute the centroids. Points are generated
        randomly and attributed to nearest cluster's appartenance points.
        Regions emerging from this operation are then used to compute barycentre

        Arguments:
            -data_points: np.array(float): The original points with attributed
            clusters at a cetrain step

        Returns:
            -new_centroids_coordinates: np.array(float) with the coordinates of
            new-computed centroids
        """
        data_region = self.initiate_region_points()
        clusters = data_points[:, -1]
        data_points = data_points[:, :-1]
        closest_points = spatial.KDTree(data_points).query(data_region)[1]
        attributed_regions = clusters[closest_points]
        data_region = np.column_stack((data_region, attributed_regions))
        unique_centroids = np.unique(clusters)
        new_centroids_coordinates = np.zeros(
            shape=(self.final_params["n_clusters"], self.configs["number_dimension"])
        )

        for i, centroid in enumerate(unique_centroids):
            centroid_region = data_region[data_region[:, -1] == centroid][:, :-1]
            length = centroid_region.shape[0]
            new_centroids_coordinates[i] = np.array(
                [
                    np.sum(centroid_region[:, c]) / length
                    for c in range(centroid_region.shape[1])
                ]
            )
        return new_centroids_coordinates

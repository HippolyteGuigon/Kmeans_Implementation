import numpy as np
import sys
import os
from sklearn.metrics.pairwise import pairwise_distances
from scipy import spatial
from src.confs.confs import load_conf

sys.path.insert(0, os.path.join(os.getcwd(), "src/confs"))
sys.path.insert(0, os.path.join(os.getcwd(), "src/data_generator"))
sys.path.insert(0, os.path.join(os.getcwd(), "src/iteration"))

from data_generator import Data_Generator
from iteration import Generate_Region

class Model(Data_Generator,Generate_Region):
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
        self.data_region=super().initiate_region_points()

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

    def first_attribution(self):
        distances = pairwise_distances(self.data, self.initial_coordinates)
        cluster_belonging = np.argmin(distances, axis=1)
        full_data=np.column_stack((self.data,cluster_belonging))
        np.save(self.configs_model["path_save"],full_data)
        return full_data

    def compute_centroid(self,data_points):
        """
        The goal of this function is to determine  each cluster zone of 
        influence that will then be used to compute the centroids. Points are generated 
        randomly and attributed to nearest cluster's appartenance points.
        Regions emerging from this operation are then used to compute centroid
        
        Args:
        data_points: np.array(): The original points with attributed clusters at a cetrain step
        
        Return:
        np.array() with the centroids of each clusters"""

        full_data=self.first_attribution()
        data_points=full_data[:,:-1]
        attributed_regions=spatial.KDTree(data_points).query(self.data_region)[1]
        print(attributed_regions)



a = Model()
a.generate_initial_K()
a.first_attribution()
a.compute_centroid(a.first_attribution())
import numpy as np
import sys
import os
from sklearn.metrics.pairwise import pairwise_distances
from src.confs.confs import load_conf

sys.path.insert(0, os.path.join(os.getcwd(), "src/confs"))
sys.path.insert(0, os.path.join(os.getcwd(), "src/data_generator"))
sys.path.insert(0, os.path.join(os.getcwd(), "src/iteration"))

from data_generator import Data_Generator
from iteration import Generate_Region


class Model(Data_Generator, Generate_Region):
    """
    The goal of this class is to implement the KMeans
    algorithm step by step
    """

    def __init__(
        self,
        path_config_model="configs/model_params.yml",
        path_config_file="configs/data_params.yml",
        randomly_generated_data=True,
    ):

        self.configs = load_conf(path_config_file)
        self.configs_model = load_conf(path_config_model)
        self.K = self.configs_model["K"]
        self.data = super().generate_data(random=randomly_generated_data)
        self.data_region = super().initiate_region_points()
        self.generate_region = Generate_Region()

        if not randomly_generated_data:
            self.configs["number_of_individuals"] = self.data.shape[0]
            self.configs["number_dimension"] = self.data.shape[1]
            self.configs["limit_min"] = self.data.min()
            self.configs["limit_max"] = self.data.max()

    def generate_initial_K(self, random_initialisation=True, *args) -> np.array(float):
        """
        The goal of this function is to initialize K centroids
        randomly that will be then used to allocate points between
        the different clusters

        Arguments:
            -random_initialisation: bool : If True, initial_K are generated
            randomly, else, the user must enter an np.array(float) containing
            the coordinates wished as initial cluster.
            -args : np.array(float): If random_initialisation is False, then
            args corresponds to the initial cluster coordinates entered by the
            user.

        Returns:
            -initial_coordinates: np.array(float): The coordinates of the
            K centroids
        """

        if random_initialisation:
            lim_min = self.configs["limit_min"]
            lim_max = self.configs["limit_max"]
            initial_cluster_coordinates = np.random.uniform(
                low=lim_min,
                high=lim_max,
                size=(self.K, self.configs["number_dimension"]),
            )
            self.initial_coordinates = initial_cluster_coordinates
            return self.initial_coordinates

        else:

            initial_coordinates = args[0]
            K = self.configs_model["K"]
            dimension = self.configs["number_dimension"]
            if (
                initial_coordinates.shape[0] != K
                or initial_coordinates.shape[1] != dimension
            ):
                raise ValueError(
                    f"The shapes of entered cluster must be [{K},{dimension}]"
                )
            self.initial_coordinates=initial_coordinates
            return initial_coordinates

    def first_attribution(self) -> np.array(float):
        """
        The goal of this function is to attribute the
        points to the centroids generated in the previous
        function as a first iteration

        Arguments:
            None

        Returns:
            -full_data: np.array(float): Numpy array with the
            original points and a column representing the nearest
            cluster
        """
        distances = pairwise_distances(self.data, self.initial_coordinates)
        cluster_belonging = np.argmin(distances, axis=1)
        full_data = np.column_stack((self.data, cluster_belonging))
        np.save(self.configs_model["path_save"], full_data)
        self.current_repartition=full_data
        return full_data

    def cluster_attribution(self, centroids) -> np.array(float):
        """
        The goal of this function is to return, at each step of
        the algorithm, to attribute each point to the nearest
        centroid.

        Arguments:
            -centroids: np.array(float): The coordinates of the
            centroids at each step

        Returns:
            -full_data: np.array(float): Numpy array with the
            original points and a column representing the nearest
            cluster
        """
        distances = pairwise_distances(self.data, centroids)
        cluster_belonging = np.argmin(distances, axis=1)
        full_data = np.column_stack((self.data, cluster_belonging))
        np.save(self.configs_model["path_save"], full_data)
        return full_data

    def fit(self) -> np.array(float):
        """
        The goal of this function is, at each step of the
        algorithm, to compute the region of influence of each
        centroid and its centroid

        Arguments:
            None

        Returns:
            -current_repartition: np.array(float): The final repartition
            of points among calculated clusters
        """
        self.current_cluster_position = self.initial_coordinates
        while not np.allclose(
            self.current_cluster_position,
            self.generate_region.compute_centroid(self.current_repartition),
            atol=0.5,
        ):
            self.current_repartition = self.cluster_attribution(
                self.current_cluster_position
            )
            self.current_cluster_position = self.generate_region.compute_centroid(
                self.current_repartition
            )
        return self.current_repartition

    def save_final_clustering(self) -> None:
        """
        The goal of this function is, after having launched
        the iteration, to save the final clustering for the
        final user.

        Arguments:
            None

        Returns:
            None
        """
        np.save("data/final_clustered_data.npy", self.current_repartition)

    def get_final_cluster_position(self) -> np.array(float):
        """
        The goal of this function is to return the
        final clusters positions after the iterations
        have been performed

        Arguments:
            None

        Returns:
            -current_cluster_position: np.array(float): The
            final cluster positions after iterations have been
            performed.
        """
        return self.current_cluster_position

    def labels(self)->np.array(int):
        """
        The goal of this function is to return an array 
        with all the predicted labels after the clustering
        was performed
        
        Arguments:
            None 
            
        Returns:
            -label: np.array(int): The predicted labels
        """

        label=self.current_repartition[:,-1]
        return label

a = Model()
a.generate_initial_K()
a.first_attribution()
a.fit()

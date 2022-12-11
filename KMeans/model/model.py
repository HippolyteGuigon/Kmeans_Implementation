import numpy as np
import sys
import os
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.exceptions import NotFittedError
from KMeans.confs.confs import load_conf, load_default_params, updating_parameter
from KMeans.data_generator.data_generator import Data_Generator
from KMeans.iteration.iteration import Generate_Region
from KMeans.trackcalls.trackcalls import trackcalls
import json
from scipy import spatial
import random
import ruamel.yaml

model_params=load_conf("configs/model_params.yml")

class KMeans(Data_Generator, Generate_Region):
    """
    The goal of this class is to implement the KMeans
    algorithm step by step

    Arguments:
        randomly_generated_data: bool: Initiates data randomly
        or not according to user's choice
        **kwargs: key, value: The model parameters chosen by the
        user
    """

    def __init__(
        self,
        path_config_model=model_params["path_config_model"],
        path_config_file=model_params["path_config_file"],
        path_config_default=model_params["path_config_default"],
        **kwargs,
    ):

        self.default_dict_params = load_conf(path_config_default)

        self.dict_params = {}

        for param, value in kwargs.items():
            if param not in self.default_dict_params.keys():
                raise AttributeError(f"The KMeans model has no attribute {param}")
            else:
                self.dict_params[param] = value
        
        self.dict_params = load_default_params(self.dict_params)

        if type(self.dict_params["init"]) == np.ndarray:
            self.dict_params["init"] = self.dict_params["init"].tolist()
        with open(model_params["final_parameter_path"], "w") as fp:
            json.dump(self.dict_params, fp)

        self.configs = load_conf(path_config_file)
        self.configs_model = load_conf(path_config_model)

        if not self.dict_params["randomly_generated_data"]:
            if not os.path.exists(model_params["to_cluster_data_path"]):
                raise OSError(
                    "The user must upload his data under the\
                         path data/data_to_cluster.npy"
                )
            self.data = np.load(model_params["to_cluster_data_path"])
            updating_parameter(self.configs, self.data)

        self.K = self.dict_params["n_clusters"]
        self.data = super().generate_data(
            random=self.dict_params["randomly_generated_data"]
        )
        self.data_region = super().initiate_region_points(self.data)
        self.generate_region = Generate_Region()

    def generate_initial_K(self, *args) -> np.array(float):
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

        if self.default_dict_params["init"] == "random":
            lim_min = self.configs["limit_min"]
            lim_max = self.configs["limit_max"]
            initial_cluster_coordinates = np.random.uniform(
                low=lim_min,
                high=lim_max,
                size=(self.K, self.configs["number_dimension"]),
            )
            self.initial_coordinates = initial_cluster_coordinates
            return self.initial_coordinates

        elif type(self.default_dict_params["init"]) == np.ndarray:

            initial_coordinates = args[0]
            K = self.dict_params["n_clusters"]
            dimension = self.configs["number_dimension"]
            if (
                initial_coordinates.shape[0] != K
                or initial_coordinates.shape[1] != dimension
            ):
                raise ValueError(
                    f"The shapes of entered cluster must be [{K},{dimension}]"
                )
            self.initial_coordinates = initial_coordinates
            return initial_coordinates

        elif self.default_dict_params["init"] == "k-means++":
            K = self.dict_params["n_clusters"]
            centroids = np.random.uniform(
                low=self.configs["limit_min"],
                high=self.configs["limit_max"],
                size=(1, self.configs["number_dimension"]),
            )

            while len(centroids) < K:
                distance_closest_point = spatial.KDTree(centroids).query(self.data)[0]
                point_choice = random.choices(
                    distance_closest_point,
                    weights=(
                        i / sum(distance_closest_point) for i in distance_closest_point
                    ),
                )[0]
                new_cluster = np.array(
                    [self.data[np.where(distance_closest_point == point_choice)[0][0]]]
                )
                centroids = np.append(centroids, new_cluster, axis=0)

            self.initial_coordinates = centroids
            return centroids

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
        self.current_repartition = full_data
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

    @trackcalls
    def fit(self, X=np.array([])) -> None:
        """
        The goal of this function is, at each step of the
        algorithm, to compute the region of influence of each
        centroid and its centroid

        Arguments:
            -X: np.array(float): The data the model must be fitted on

        Returns:
            None
        """
        if self.dict_params["randomly_generated_data"]:
            X = self.data
            self.generate_initial_K()
            self.first_attribution()
        elif not self.dict_params["randomly_generated_data"] and np.any(X):
            updating_parameter(self.configs, X)
            self.data = X
            self.generate_initial_K()
            self.first_attribution()
        self.current_cluster_position = self.initial_coordinates
        iter = 0
        while (
            not np.allclose(
                self.current_cluster_position,
                self.generate_region.compute_centroid(self.current_repartition),
                atol=0.5,
            )
            and iter < self.dict_params["max_iter"]
        ):

            self.current_repartition = self.cluster_attribution(
                self.current_cluster_position
            )
            self.current_cluster_position = self.generate_region.compute_centroid(
                self.current_repartition
            )
            iter += 1

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
        np.save(model_params["final_data_save_path"], self.current_repartition)

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

    def labels(self) -> np.array(int):
        """
        The goal of this function is to return an array
        with all the predicted labels after the clustering
        was performed

        Arguments:
            None

        Returns:
            -label: np.array(int): The predicted labels
        """

        label = self.current_repartition[:, -1]
        return label

    def predict(self, X: np.array(float)) -> np.array(float):
        """
        The goal of this function is, after the model
        has been fitted, to predict the cluster appartenance
        of a set of points X

        Arguments:
            -X: np.array(float): Set of points to be clustered

        Returns:
            -clustered_data: np.array(float): The points after
            clustering was performed"""

        if not self.fit.has_been_called:
            raise NotFittedError("The KMeans model has to be fitted first")

        if X.shape[1] != self.configs["number_dimension"]:
            raise ValueError(
                f"The data to be clustered has to be of dimension (,{self.configs['number_dimension']})"
            )

        distances = pairwise_distances(X, self.current_cluster_position)
        cluster_belonging = np.argmin(distances, axis=1)
        clustered_data = np.column_stack((X, cluster_belonging))
        return clustered_data

    def get_params(self):
        """
        This function returns the params chosen by the
        user.

        Arguments:
            None

        Returns:
            -dict_params: dict[str]: The dictionnary containing
            the parameters of the model
        """
        return self.dict_params

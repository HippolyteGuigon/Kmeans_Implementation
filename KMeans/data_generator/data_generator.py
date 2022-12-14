import sys
import numpy as np
import os

from KMeans.confs.confs import load_conf

model_params=load_conf("configs/path_params.yml")

class Data_Generator:
    """
    The goal of this class is generating random data according
    to parameters entered by the User in appropriate yml file
    """

    def __init__(
        self,
        path_config="configs/data_params.yml",
        path_config_model="configs/path_params.yml",
    ) -> None:
        self.configs = load_conf(path_config)
        self.configs_model = load_conf(path_config_model)

    def generate_data(self,random=True) -> np.array:
        """
        Generates the data randomly according to the configs
        file

        Arguments:
            None

        Returns:
            -data_generated : np.array(float): Data randomly-generated
        """
        if random:
            n_rows = self.configs["number_of_individuals"]
            n_columns = self.configs["number_dimension"]
            lim_min = self.configs["limit_min"]
            lim_max = self.configs["limit_max"]
            data_generated = np.random.uniform(
                low=lim_min, high=lim_max, size=(n_rows, n_columns)
            )

        else:
            data_generated=np.load(model_params["to_cluster_data_path"])
            
        return data_generated

    def save_data(self) -> None:
        """
        Saves the data randomly produced under
        the following path: data/data_generated.npy

        Arguments:
            None

        Returns:
            Save data under following_path: data/data_generated.npy
        """
        data_generated = self.generate_data()
        np.save(self.configs_model["path_save"], data_generated)


if __name__ == "__main__":
    loader = Data_Generator()
    loader.save_data()

import yaml
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "../configs")

class Data_Generator:
    """
    The goal of this class is generating random data according
    to parameters entered by the User in appropriate yml file
    """

    def __init__(self, path_config="configs/data_params.yml") -> None:
        with open(path_config, "r") as ymlfile:
            configs = yaml.safe_load(ymlfile)

        self.configs = configs

    def generate_data(self) -> np.array:
        """
        Generates the data randomly according to the configs
        file
        """
        n_rows = self.configs["number_of_individuals"]
        n_columns = self.configs["number_dimension"]
        lim_min = self.configs["limit_min"]
        lim_max = self.configs["limit_max"]
        data_generated = np.random.uniform(
            low=lim_min, high=lim_max, size=(n_rows, n_columns)
        )
        return data_generated

    def save_data(self) -> None:
        """
        Saves the data randomly produced under
        the following path: data/data_generated.npy
        """
        data_generated = self.generate_data()
        np.save("data/data_generated.npy", data_generated)


if __name__ == "__main__":
    loader = Data_Generator()
    loader.save_data()

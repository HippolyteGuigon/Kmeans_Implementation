import numpy as np
import sys 
import os
sys.path.insert(0, os.path.join(os.getcwd(), "src/confs"))
from confs import load_conf

class Generate_Region:

    def __init__(self,
        path_config_model="configs/model_params.yml",
        path_config_file="configs/data_params.yml"):

        self.configs = load_conf(path_config_file)
        self.configs_model = load_conf(path_config_model)

    def initiate_region_points(self):
        """
        Generates the data randomly according to the configs
        file
        """
        n_rows = 1000*self.configs["number_of_individuals"]
        n_columns = self.configs["number_dimension"]
        lim_min = self.configs["limit_min"]
        lim_max = self.configs["limit_max"]
        data_generated = np.random.uniform(
            low=lim_min, high=lim_max, size=(n_rows, n_columns)
        )
        return data_generated

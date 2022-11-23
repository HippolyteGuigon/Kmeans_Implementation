import unittest
import yaml
from src.data_generator.data_generator import Data_Generator
from src.model.model import Model
from src.confs.confs import load_conf

test = Data_Generator()
model_test = Model()

class Test(unittest.TestCase):
    """
    The goal of this class is to implement unnitest
    and check everything commited makes sense
    """

    def __init__(self,path_config="configs/data_params.yml",path_model="configs/model_params.yml"):
        self.configs=load_conf(path_config)
        self.configs_model=load_conf(path_model)

    def test_generated_data(self):
        test_data = test.generate_data()

        n_rows = self.configs["number_of_individuals"]
        n_columns = self.configs["number_dimension"]
        lim_min = self.configs["limit_min"]
        lim_max = self.configs["limit_max"]

        self.assertEqual(test_data.shape[0], n_rows)
        self.assertEqual(test_data.shape[1], n_columns)
        self.assertGreaterEqual(test_data.min(), lim_min)
        self.assertLessEqual(test_data.max(), lim_max)

    def test_initial_centroids(self):
        test_centroid = model_test.generate_initial_K()
        self.assertEqual(test_centroid.shape[0], self.configs_model["K"])
        self.assertEqual(test_centroid.shape[1], self.configs_model["number_dimension"])


if __name__ == "__main__":
    unittest.main()

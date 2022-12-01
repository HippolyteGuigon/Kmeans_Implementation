import unittest
from src.data_generator.data_generator import Data_Generator
from src.model.model import Model
from src.confs.confs import load_conf
import numpy as np
import os

test = Data_Generator()
model_test = Model()

configs = load_conf("configs/data_params.yml")
configs_model = load_conf("configs/model_params.yml")


class Test(unittest.TestCase):
    """
    The goal of this class is to implement unnitest
    and check everything commited makes sense
    """

    def test_generated_data(self):
        """
        The goal of this test is to check if randomly-generated
        data have the appropriate dimension as entered by the user
        in the config file

        Arguments:
            None

        Returns:
            -bool: True or False
        """
        test_data = test.generate_data()

        n_rows = configs["number_of_individuals"]
        n_columns = configs["number_dimension"]
        lim_min = configs["limit_min"]
        lim_max = configs["limit_max"]

        self.assertEqual(test_data.shape[0], n_rows)
        self.assertEqual(test_data.shape[1], n_columns)
        self.assertGreaterEqual(test_data.min(), lim_min)
        self.assertLessEqual(test_data.max(), lim_max)

    def test_initial_centroids(self):
        """
        The goal of this test function is checking if the centroids
        generated have the appropriate dimensions compared to the
        generated-data

        Arguments:
            None

        Returns:
            -bool: True or False
        """
        test_centroid = model_test.generate_initial_K()
        model_test.first_attribution()
        self.assertEqual(test_centroid.shape[0], configs_model["K"])
        self.assertEqual(test_centroid.shape[1], configs["number_dimension"])

    def test_manual_entrance_centroids(self):
        """
        The goal of this test-function is checking if the full KMeans
        pipeline works when clusters are entered manually by the user

        Arguments:
            None

        Returns:
            -bool: True or False
        """
        n_columns = configs["number_dimension"]
        K = configs_model["K"]

        try:
            model_test.generate_initial_K(False, np.random.uniform(size=(K, n_columns)))
            model_test.first_attribution()
            model_test.launch_iteration()
        except:
            self.fail("Error detected")

    def test_full_pipeline(self):
        """
        The goal of this test-function is checking if the full KMeans
        pipeline works when centroids are randomly generated and not
        chosen by the user

        Arguments:
            None

        Returns:
            -bool: True or False
        """
        try:
            model_test.generate_initial_K()
            model_test.first_attribution()
            model_test.launch_iteration()
        except:
            self.fail("Error detected")

    def test_full_pipeline_own_data(self):
        """
        The goal of this test-function is to check that the
        clustering works with randomly generated data, even if
        the yml file has not been changed explicitely

        Arguments:
            None

        Returns:
            -bool: True or False
        """
        data_generated = np.random.uniform(
            low=-1000,
            high=1000,
            size=(np.random.randint(low=1, high=10), np.random.randint(low=1, high=10)),
        )
        np.save("data/data_to_cluster.npy", data_generated)
        try:
            os.system("python main.py Hippolyte own_data")
        except:
            self.fail("Error detected")


if __name__ == "__main__":
    unittest.main()

import unittest
from KMeans.data_generator.data_generator import Data_Generator
from KMeans.model.model import KMeans
from KMeans.confs.confs import load_conf
import numpy as np
import os

test = Data_Generator()

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
        model_test = KMeans(randomly_generated_data=True, max_iter=10)
        K = model_test.get_params()["n_clusters"]
        test_centroid = model_test.generate_initial_K()
        model_test.first_attribution()
        configs = load_conf(configs_model["path_config_file"])
        self.assertEqual(test_centroid.shape[0], K)
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

        try:
            model_test = KMeans(
                randomly_generated_data=True,
                max_iter=10,
                init=np.random.uniform(size=(2, n_columns)),
            )
            model_test.generate_initial_K(False)
            model_test.first_attribution()
            model_test.fit()
        except:
            self.fail("Error detected")

    def test_kmeans_plus_plus(self):
        """
        The goal of this function is to check if
        KMeans ++ works as initialization for the first
        centroids

        Arguments:
            None

        Returns:
            boolean: True or False"""

        try:
            model_test = KMeans(
                randomly_generated_data=True, max_iter=10, init="k-means++"
            )
            model_test.generate_initial_K(False)
            model_test.first_attribution()
            model_test.fit()
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
            model_test = KMeans(randomly_generated_data=True, max_iter=10)
            model_test.generate_initial_K()
            model_test.first_attribution()
            model_test.fit()
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
        np.random.seed(42)
        data_generated = np.random.uniform(
            low=-1000,
            high=1000,
            size=(1000, np.random.randint(low=1, high=10))
        )
        np.save(configs_model["to_cluster_data_path"], data_generated)
        try:
            os.system("python main.py Hippolyte own_data")
        except:
            self.fail("Error detected")

    def test_predict_without_fit(self):
        """
        The goal of this function is to check
        that the predict function of the KMeans
        fails if the model is not fitted first.

        Arguments:
            None

        Returns:
            bool: True or False"""
        model = KMeans(randomly_generated_data=True)

        self.assertRaises(
            AttributeError,
            model.predict,
            "'KMeans' object has no attribute 'current_cluster_position'",
        )

    def test_predict_with_fit(self):
        """
        The goal of this function is to check if the predict function
        works with random data.

        Arguments:
            None

        Returns:
            bool: True or False
        """

        model = KMeans(randomly_generated_data=True, max_iter=5)
        model.generate_initial_K()
        model.first_attribution()
        model.fit()
        configs = load_conf(configs_model["path_config_file"])
        np.random.seed(42)
        X = np.random.uniform(
            low=-100, high=100, size=(100, configs["number_dimension"])
        )

        try:
            model.predict(X)
        except:
            self.fail("Error detected")

    def test_invalid_argument(self):
        """
        The goal of this test function is to check
        wheter an AttributeError is raised whenever
        an invalid argument in given to the function

        Arguments:
            None

        Returns:
            bool: True or False
        """
        self.assertRaises(AttributeError, lambda: KMeans(faux_argument=3))

    def test_fit_function(self):
        """
        The goal of this function is to check that the fitting function 
        works when given an array as input
        
        Arguments:
            -X:np.array(float): The set of points to be clustered
            
        Returns:
            bool: True or False
        """

        X=np.random.uniform(low=-100, high=100,size=(np.random.randint(low=500,high=1000),configs["number_dimension"]))
        model=KMeans(randomly_generated_data=False)
        y=np.random.uniform(low=-1000,high=1000,size=X.shape)
        try:
            model.fit(X)
            model.labels
            model.predict(y)
        except:
            self.fail("Error detected")

    def testing_manual_fit_predict(self):
        """
        The goal of this testing function is to make sure
        that the fit and predict manual function of the KMeans
        algorithm works and returns data with appropriate shape.
        
        Arguments:
            None

        Returns:
            bool: True or False
        """
        dimension=np.random.randint(1, 10)
        number_of_rows=np.random.randint(low=500,high=1000)
        np.random.seed(42)
        X=np.random.uniform(low=-100, high=100,size=(number_of_rows,dimension))
        y=np.random.uniform(low=-1000,high=1000,size=(number_of_rows,dimension))
        model=KMeans(randomly_generated_data=False)
        model.fit(X)
        self.assertEqual(model.labels().shape[0], number_of_rows)
        self.assertEqual(model.predict(y)[:,-1].shape[0], number_of_rows)

if __name__ == "__main__":
    unittest.main()

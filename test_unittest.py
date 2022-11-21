import unittest
import yaml
from src.data_generator.data_generator import Data_Generator

test = Data_Generator()

with open("configs/data_params.yml", "r") as ymlfile:
    configs = yaml.safe_load(ymlfile)


class Test(unittest.TestCase):
    """
    The goal of this class is to implement unnitest
    and check everything commited makes sense
    """

    def test_generated_data(self):
        test_data = test.generate_data()

        n_rows = configs["number_of_individuals"]
        n_columns = configs["number_dimension"]
        lim_min = configs["limit_min"]
        lim_max = configs["limit_max"]

        self.assertEqual(test_data.shape[0], n_rows)
        self.assertEqual(test_data.shape[1], n_columns)
        self.assertGreaterEqual(test_data.min(), lim_min)
        self.assertLessEqual(test_data.max(), lim_max)


if __name__ == "__main__":
    unittest.main()

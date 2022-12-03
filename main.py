import argparse
import logging
import logging.handlers
import sys
import numpy as np

sys.path.insert(0, "src/logs")
sys.path.insert(0, "src/model")
from model import KMeans
from logs import main

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

main()

parser = argparse.ArgumentParser()
parser.add_argument(
    "Name",
    help="The name entered by the user to easily find your own iteration in the logs",
    type=str,
)

parser.add_argument(
    "data_type",
    help="The kind of data to be clustered. If you want a random set of data to be clustered,enter random, else enter own_data",
    type=str,
)

args = parser.parse_args()


def model_launch() -> None:
    """
    The goal of this function is to launch
    the all model pipeline and cluster the points
    entered as input

    Args:
        None

    Return:
        None
    """
    if args.data_type == "random":
        logger.info(f"KMeans will be performed with random data {args.Name}")
        model = KMeans(n_iter=5, randomly_generated_data=True)
    elif args.data_type == "own_data":
        logger.info(f"KMeans will be performed with pre-charged data {args.Name}")
        model = KMeans(n_iter=5, randomly_generated_data=False)
    logger.info(f"Model Charged {args.Name}")
    model.generate_initial_K(False, np.array([[0, 0], [30, 30], [60, 60]]))
    logger.info(f"Initial centroïds initialized {args.Name}")
    model.first_attribution()
    model.fit()
    logger.info(f"Model has converged {args.Name}")
    model.save_final_clustering()
    logger.info(f"Clustering is over and your data has been saved {args.Name}")
    np.save("data/final_cluster_position.npy", model.get_final_cluster_position())

if __name__ == "__main__":
    model_launch()

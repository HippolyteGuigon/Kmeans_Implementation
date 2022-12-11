import argparse
import logging
import logging.handlers
import sys
import numpy as np
from KMeans.confs.confs import load_conf
from KMeans.logs.logs import main
from KMeans.model.model import KMeans

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

model_params=load_conf("configs/model_params.yml")

parser = argparse.ArgumentParser()
parser.add_argument(
    "Name",
    help="The name entered by the user to easily find your own iteration in the logs.",
    nargs="?",
    const="Hippolyte",
    type=str,
)

parser.add_argument(
    "data_type",
    help="The kind of data to be clustered. \
        If you want a random set of data to be \
            clustered,enter random, else enter own_data.",
    nargs="?",
    const="own_data",
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
    model = KMeans(max_iter=5, randomly_generated_data=True)
    if args.data_type == "random":
        logger.info(f"KMeans will be performed with random data {args.Name}")
        model = KMeans(max_iter=5, randomly_generated_data=True)
    elif args.data_type == "own_data":
        logger.info(f"KMeans will be performed with pre-charged data {args.Name}")
        model = KMeans(max_iter=5, randomly_generated_data=False)
    logger.info(f"Model Charged {args.Name}")
    model.generate_initial_K()
    logger.info(f"Initial centroïds initialized {args.Name}")
    model.first_attribution()
    model.fit()
    logger.info(f"Model has converged {args.Name}")
    model.save_final_clustering()
    logger.info(f"Clustering is over and your data has been saved {args.Name}")
    np.save(model_params["final_save_path"], model.get_final_cluster_position())


if __name__ == "__main__":
    main()
    model_launch()

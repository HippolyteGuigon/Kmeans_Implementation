import argparse
import logging
import logging.handlers
import sys

sys.path.insert(0, "src/logs")
sys.path.insert(0, "src/model")
from model import Model
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
    model = Model()
    logger.info(f"Model Charged {args.Name}")
    model.generate_initial_K()
    logger.info(f"Initial centroïds initialized {args.Name}")
    model.first_attribution()
    logger.info(f"Model has converged {args.Name}")


if __name__ == "__main__":
    model_launch()

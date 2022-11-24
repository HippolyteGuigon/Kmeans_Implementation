# coding=utf-8
import argparse
import logging
import logging.handlers
import sys

sys.path.insert(0,"src/logs")
sys.path.insert(0,"src/model")
from model import Model
from logs import main

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

main()

parser=argparse.ArgumentParser()
parser.add_argument("Name",
help="The name entered by the user to easily find your own iteration in the logs",
type=str)
args=parser.parse_args()

def model_launch():
    model = Model()
    logger.info(f"Model Charged {args.Name}")
    model.generate_initial_K()
    logger.info(f"Initial centroïds initialized {args.Name}")
    model.compute_distances()
    logger.info(f"First iteration done {args.Name}")

if __name__=="__main__":
    model_launch()
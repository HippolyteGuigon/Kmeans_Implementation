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

def model_launch():
    model = Model()
    logger.info("Model Charged")
    model.generate_initial_K()
    logger.info("Initial centroïds initialized")
    model.compute_distances()
    logger.info("First iteration done")

if __name__=="__main__":
    model_launch()
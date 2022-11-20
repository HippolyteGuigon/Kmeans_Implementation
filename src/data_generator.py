# As a user, I want to have n points in dimenion m
# generated randomly

import yaml
from yaml.loader import SafeLoader
import os

current_path = os.getcwd()
with open(os.path.join(current_path, "configs/data_params.yml"), "r") as f:
    data_search = list(yaml.load_all(f, Loader=SafeLoader))[0]

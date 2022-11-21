# Implementation of a KMeans algorithm in Python 

The goal of this repository is to code an implementation of the KMeans algorithm. 

The Kmeans algorithm is a non-supervised Machine Learning Clustering algorithm 
which takes as input a set of points. The KMeans algorithm will create K groups 
among which those points will be reparted, K being chosen by the user.

## Build Status

For the moment, the project is still being coded, altghouth the first setps are in place.
You can have the first iteration of the algorithm which allocate the points among the K clusters,
but not as a final step of the algorithm.

## Code style 

The all project was coded under PEP-8 (https://peps.python.org/pep-0008/) and flake8 (https://pypi.org/project/flake8/) compliancy. Such compliance is verified during commits with pre-commits file ```.pre-commit-config.yaml```

## Installation

- This project uses a specific conda environment, to get it, run the following command: 
```conda env create -f environment_droplet.yml```
 
- To install all necessary libraries, run the following code: ```pip install -r requirements.txt```

## Screenshot 

![alt text](https://github.com/HippolyteGuigon/Kmeans_Implementation/blob/features_updating_documentation_configs/K_means.png)

## How to use ? 

To choose the parameters of the K-means algorithm you want to launch (number of clusters, 
number of points, number of dimensions) you have to specify them in the following configs file:  

  -```configs/data_params.yml```  
  -```configs/model_params.yml```

Then, run the following command: ```python run main.py```
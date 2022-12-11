# Implementation of a KMeans algorithm in Python 

The goal of this repository is to code an implementation of the KMeans algorithm. 

The Kmeans algorithm is a non-supervised Machine Learning Clustering algorithm 
which takes as input a set of points. The KMeans algorithm will create K groups 
among which those points will be reparted, K being chosen by the user.

## Build Status

For the moment, the KMeans algorithm is ready, and you can choose between clustering your own imported data on run the all Pipeline on randomly generated data. The user can also manually enter the centroid he wishes for the first iteration.
Next steps are to allow the user to enter his own data as well as implementing other KMeans 
classical functionnalities (Choose different distances,add other hyperparameters).

If you see any improvements that could be made in the code, do not hesitate to reach out at 
Hippolyte.guigon@hec.edu

## Code style 

The all project was coded under PEP-8 (https://peps.python.org/pep-0008/) and flake8 (https://pypi.org/project/flake8/) compliancy. Such compliance is verified during commits with pre-commits file ```.pre-commit-config.yaml```

## Installation

- This project uses a specific conda environment, to get it, run the following command: 
```conda env create -f environment_droplet.yml```
 
- To install all necessary libraries, run the following code: ```pip install -r requirements.txt```

## Screenshot 

![alt text](https://github.com/HippolyteGuigon/Kmeans_Implementation/blob/main/ressources/K_means.png)

## How to use ? 

You have two options for running this KMeans algorithm:


- The first one is to run in through the shell following the next steps:

To choose the parameters of the K-means algorithm you want to launch (number of clusters, 
number of points, number of dimensions) you have to specify them in the following configs file:  

  -```configs/data_params.yml```  
  
As well as in the model once it is launched. You can check the available parameters at the following 
path:

  -```configs/default_params.yml```  

Then, you have two options:
  -If you want to run the KMeans algorithm with random data, run the following command: ```python run main.py user_name random```

  -If you want to cluster your own data, put them under the following path: ```data/data_to_cluster.npy``` and run the following command:  ```python run main.py user_name own_data```

In both cases, you'll find your clustered data under the following path: ```data/final_clustered_data.npy``` and the centroids under the path ```data/final_cluster_position.npy```


- The second method is to run it like the classical sklearn KMeans model

For this, run the following steps:

  -```from KMeans.model.model import KMeans```

  -```model=KMeans()```
  
  -```model.fit(X)``` with X the numpy array of the data to cluster

  -```model.labels()``` to get the clusters of the clustered data
  
  -```model.predict(y)``` with y another dataset of points to cluster 




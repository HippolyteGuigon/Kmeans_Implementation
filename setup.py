from setuptools import setup, find_packages

setup(
    name='KMeans',
    version='0.1.0',
    packages=find_packages(include=['KMeans', 'KMeans.*']),
    description='A python implementation of the KMeans algorithm',
    author='Hippolyte Guigon',
    author_email='Hippolyte.guigon@hec.edu',
    url='https://github.com/HippolyteGuigon/Kmeans_Implementation'
)

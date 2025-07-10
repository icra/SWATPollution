# setup.py
from setuptools import setup, find_packages

setup(
    name="SWATPollution",
    version="1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'networkx',
        'ipykernel',
        'pandas',
        'sqlalchemy',
        'geopandas',
        'scikit-learn',
        'plotly',
        'bokeh',
        'geoviews',
        'openpyxl',
    ],    
    include_package_data=True,
    author='Joan Saló',
    description='Python class to interact with SWAT pollution module',
    url="https://github.com/icra/SWATPollution.git",
)
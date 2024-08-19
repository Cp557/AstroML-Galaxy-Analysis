# AstroML Galaxy Analysis

## Overview

This repository contains two interconnected machine learning projects focused on galaxy analysis using data from the Sloan Digital Sky Survey (SDSS):

1. **Redshift Regression**: Predicts galaxy redshifts using photometric data.
2. **Galaxy Morphological Classification**: Classifies galaxies into three morphological types (merger, elliptical, spiral) using various features including predicted redshift.

These projects demonstrate the application of machine learning techniques to astronomical data, aiming to accelerate research in galaxy evolution and cosmology.

![Galaxy Types](galaxies.png)
*Figure: Examples of different galaxy types used in the classification model.*

## Repository Contents

- `redshift_regression.ipynb`: Jupyter notebook for the redshift prediction model
- `galaxy_classification.ipynb`: Jupyter notebook for the galaxy classification model
- `sdss_galaxy_colors.npy`: NumPy file containing SDSS galaxy color data
- `galaxy_catalogue.npy`: NumPy file containing galaxy catalogue data
- `redshift_model.joblib`: Saved redshift prediction model
- `galaxies.png`: Image file showing examples of galaxy types

## Features

- Data preprocessing and exploratory data analysis of SDSS galaxy data
- Implementation of XGBoost for redshift prediction
- Deep learning model using TensorFlow/Keras for galaxy classification
- Integration of redshift predictions into the classification model
- Comprehensive model evaluation and visualization

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/AstroML-Galaxy-Analysis.git
   ```
2. Navigate to the project directory:
   ```
   cd AstroML-Galaxy-Analysis
   ```
3. Install required packages:
   ```
   pip install numpy pandas matplotlib seaborn tensorflow scikit-learn xgboost joblib
   ```

## Usage

1. Open and run the Jupyter notebooks:
   ```
   jupyter notebook redshift_regression.ipynb
   jupyter notebook galaxy_classification.ipynb
   ```

Note: Ensure you have Jupyter Notebook installed and that the .npy data files are in the same directory as the notebooks.

## Results

### Redshift Regression
XGBoost model:
- Mean Squared Error: 0.058631 
- R-squared score: 0.797278

Ridge Regression:
- Mean Squared Error: 0.157324
- R-squared score 0.443480

### Galaxy Classification
- Test Accuracy: 0.8564
- ROC AUC Score: 0.9369

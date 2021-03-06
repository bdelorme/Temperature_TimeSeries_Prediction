# Temperature TimeSeries Prediction

**Author: Bertrand Delorme**

**Date: 23/05/2018**

## General

### Goal of the project

Predicting the temperature of a subway station at time *h+1* based on historical data from the beginning of time to time *h*.

### Dataset

The dataset is provided by RATP ([link](https://data.ratp.fr)) under open data Etalab license and contains air property measurements from a subway station in csv format. It has 7 features: day and time of measurement, temperature, humidity, particulates concentration and concentration of 3 chemical tracers (NO, NO2, CO2). 

### Implementation choices

Two implementations have been tested to predict the temperature at *h+1*:
- a random forest ensemble algorithm based on the value of temperature at *h* and *h-1*, the hour of the day, and the day of the month.
- a Seasonal AutoRegressive Integrated Moving Average (SARIMA) algorithm. 

As the random forest algorithm performs better in our experiments, we use it as finale response for the coding challenge.

## Installation

### Requirements

The dependencies needed are:

- numpy
- scipy
- pandas
- scikit-learn
- jupyter
- statsmodels 

as shown in [config.yml](config.yml).

### Setup instructions

We suggest using [Anaconda](https://www.anaconda.com/) to create a conda environment with the required dependencies as: 
```bash
cd Temperature_TS_Prediction
conda env create -f config.yml
source activate Temperature_TS_Prediction
```

## Project Overview

### Libraries used

- numpy for efficient data structures and functions for scientific computing.
- pandas for data manipulation and data analysis.
- matplotlib for data visualization.
- scikit-learn for Random Forest implementation.
- statsmodels for Seasonal Arima implementation.

### Architecture

- `data/`: contains the raw historical dataset.
- `src/`: contains the source code to make the prediction.
- `config.yml`: conda config file.
- `predict`: executable to get the prediction.
- `exploration.ipynb`: jupyter notebook showing the exploration process and justification of implementation choices.

## How it works

Given a dataset with historical data until time *h*, you get a prediction for time *h+1* by simply doing: 
```bash
./predict path_to_historical_dataset
```

This should last less than a minute and print the prediction at time *h+1*. To store the result in a file *result.txt*, do:
```bash
./predict path_to_historical_dataset > result.txt
```

We suggest that you first go over the jupyter [notebook](exploration.ipynb) to understand the choices made.

## Improvements to do

- test files
- Random Forest in Fourier Space
- implementation of RNN with LSTM

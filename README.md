# Samsung Coding Challenge 
### Author: Bertrand Delorme
### Date: 23/05/2018

I have implemented a Random Forest algorithm with Python3 to predict the temperature of a subway station at time "h+1" based on historical values from the beginning of time to "h".

## Project structure
- data contains the raw historical dataset
- notebook contains a jupyter notebook showing the exploration process (this is where I justify my implementation choices)
- src contains the source code (modules).

## Set-up instructions
The dependencies needed to run the experiment are defined in [config.yml](config.yml).

## Experiment
Given a dataset with historical data from the beginning of time to "h", you can get a prediction for the next hour, "h+1", by simply doing: 
```bash
python predict.py path_to_historical_dataset
```
This function prints the prediction at time "h+1", and store it in a file *result.txt*.


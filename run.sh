#!/bin/bash

python experiment.py --skip_hyperparameter_tuning -p entire
python experiment.py --skip_hyperparameter_tuning --code_generation -p entire
python experiment.py --skip_hyperparameter_tuning -p input_only 
python experiment.py --skip_hyperparameter_tuning --code_generation -p input_only 

python visualize.py -p entire
python visualize.py --code_generation -p entire
python visualize.py -p input_only 
python visualize.py --code_generation -p input_only 

python generalize.py -p entire

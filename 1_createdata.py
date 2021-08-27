from Step_1_create_dataset.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from util import util
from pathlib import Path
import copy
import os
import sys

# directory paths for the current experiment (change it if needed)
FOLDER_PATH = Path('./datasets/motor_imagery/')
RESULT_PATH = Path('./intermediate_datafiles/motor_imagery/step1_result/')
RESULT_PATH.mkdir(exist_ok=True, parents=True)
GRANULARITY = 100 #milisecond per instance; we settle at 100 ms.


RESULT_PATH.mkdir(exist_ok=True, parents=True)
for instance in os.scandir(FOLDER_PATH): # go through all instances of experiments  
    instance_path = instance.path
    print(f'Creating numerical datasets for {instance_path} using granularity {GRANULARITY}.')
    dataset = CreateDataset(instance_path, GRANULARITY)

    # We add the brain wave data, for each sensor, for each brain wave
    # and aggregate the values per timestep by averaging the values
    dataset = dataset.add_data(instance_path, ['Delta_TP9','Delta_AF7','Delta_AF8','Delta_TP10',
    'Theta_TP9','Theta_AF7','Theta_AF8','Theta_TP10',
    'Alpha_TP9','Alpha_AF7','Alpha_AF8','Alpha_TP10',
    'Beta_TP9','Beta_AF7','Beta_AF8','Beta_TP10',
    'Gamma_TP9','Gamma_AF7','Gamma_AF8','Gamma_TP10'], ['label_left', 'label_right'], 'avg')

    # Plot the data
    DataViz = VisualizeDataset(__file__)

    # 1. Boxplot to check if amplitude of delta is higher than gamma
    #DataViz.plot_dataset_boxplot(dataset, ['Delta_TP9','Theta_TP9', 'Alpha_TP9', 'Beta_TP9', 'Gamma_TP9',
    #'Delta_AF8','Theta_AF8','Alpha_AF8','Beta_AF8','Gamma_AF8'], instance.name.split('--')[0])

    # 2. Plot brainwaves and labels
    DataViz.plot_dataset(dataset, ['Gamma_','Beta_', 'Alpha_', 'Theta_', 'Delta_', 'label_'],
                                    ['like', 'like', 'like', 'like', 'like', 'like'],
                                    ['line', 'line', 'line', 'line', 'line', 'line'], instance.name.split('_')[1])
    
    # 3. And we print a summary of the dataset.
    #util.print_statistics(dataset)

    # Store the dataset we generated.
    dataset.to_csv(Path(str(RESULT_PATH) + '/' + instance.name))
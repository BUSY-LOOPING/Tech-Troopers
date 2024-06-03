import os
import sys

import pandas as pd
import numpy as np

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

import globals
from helpers import file_helper, results_helper

def result_worker(mode, algo_type, file_dict):
    try:

        mode_results_temp_path = file_helper.create_path_from_params(globals.cwd, 'results_temp', mode, algo_type)
        mode_results_path = file_helper.create_path_from_params(globals.cwd, 'results', mode, algo_type)

        for file_name, file_dict_obj in file_dict.items():
            df = results_helper.build_data_frame(mode, algo_type, file_name, file_dict_obj)
            
            results_temp_csv_path = file_helper.create_path_from_params(mode_results_temp_path, file_name + '.csv')
            df.to_csv(results_temp_csv_path, index=False)
    
            results_helper.build_ipynb(results_temp_csv_path, mode_results_path, file_name)
    except Exception as err:
        print('result_worker: err', err)
    pass
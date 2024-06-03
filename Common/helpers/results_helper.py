import pandas as pd
import numpy as np
import pickle

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

import globals
from helpers import file_helper


def build_data_frame(mode, algo, file_name, file_dict_obj):
    df = pd.DataFrame(dtype=np.float32)
    
    min_ind = file_dict_obj['min']
    max_ind = file_dict_obj['max']

    #build file_paths
    for i in range(min_ind, max_ind):
        file_path_list = [file_name, globals.results_file_mid_text + str(i), globals.results_file_extension]
        file_name_path = '.'.join(file_path_list)


        file_actual_path = os.path.join(globals.cwd, mode, algo, file_name_path)

        with open(file_actual_path, 'rb') as file:
            data = pickle.load(file)
            for items in data:
                #print(items['vals'])
                col_names = []
                values = []

                for k,v in items.items():
                    if isinstance(v, dict):
                        for k1, v1 in v.items():
                            col_names.append(k1)
                            values.append(v1)
                    else:
                        col_names.append(k)
                        values.append(v)

                df.loc[df.shape[0], col_names] = values
                print('successful insertion')
    return df





initial_text = """\
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from xgboost import XGBClassifier

from huggingface_hub import hf_hub_download

import warnings
warnings.filterwarnings("ignore")
"""

def read_csv(file_path):
    line_list = []
    line_list.append("df = pd.read_csv('{file}')".format(file = file_path))
    line_list.append("df.dropna(axis=1, how='all')")
    line_list.append("df.head()")
    return "\n".join(line_list)

def build_ipynb(csv_path, result_path, file_name):
    try:
        #create the ipynb
        nb = nbformat.v4.new_notebook()
        initial_cell_content = initial_text
        
        cell = nbformat.v4.new_code_cell(initial_cell_content)
        nb.cells.append(cell)

        cell = nbformat.v4.new_code_cell(read_csv(csv_path))
        nb.cells.append(cell)

        with open(file_helper.create_path_from_params(result_path, file_name + '.ipynb'), 'w') as f:
            nbformat.write(nb, f)

        pass
    except Exception as err:
        print('err building ipynb', err)
        raise err        

def build_clf_num_ipynb(cv_path):
    try:
        pass
    except Exception as err:
        print('err building ipynb', err)
        raise err
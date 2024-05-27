import pandas as pd
import numpy as np
import pickle
import os

def load_results_to_dataframe(folder_path, datasetname):
    # List to store each file's data
    df = pd.DataFrame(dtype=np.float32)

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl') and datasetname in filename:
            file_path = os.path.join(folder_path, filename)
            # Load the pickle file
            with open(file_path, 'rb') as file:
                data = pickle.load(file) #list of dctionaries
                for item in data :
                    col_names = []
                    values = []
                    for key, value in item.items() :
                        if key == 'hyperparameters':
                            for k2, v2 in value.items() :
                                col_names.append(k2)
                                values.append(v2)
                        else :
                            col_names.append(key)
                            values.append(value)
                    df.loc[df.shape[0], col_names] = values
                                
    return df

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
style.use('ggplot')
import seaborn as sns
import re
import os
import pickle
from glob import glob
from drone_troopers.preprocessing import preprocess, create_time_series_features

# Define column indices for PCA
pca_column_indices = list(range(0, 18))
pca_column_indices.remove(2) # Indices for the columns from 'Motor Speed (RPM)' to 'Cooling Pump Speed (RPM)'

# Column index to scale using MinMaxScaler
throttle_column_indices = [2]  # Index for 'Throttle (%)'

# Columns to keep unchanged (by indices)
unchanged_column_indices = list(range(18, 23))  # Indices for 'flightId', 'hour', 'minute', 'second', 'microsecond'

def get_resource_path(filename):
    """Get the absolute path of a resource file."""
    return os.path.join(os.path.dirname(__file__), filename)

def scale_n_process(time_series_data) :
    with open(get_resource_path('preprocessor.pkl'), 'rb') as file:
        preprocessor = pickle.load(file)
    
    with open(get_resource_path('data_column_names.pkl'), 'rb') as file:
        data_column_names = pickle.load(file)
        
    transformed_data = preprocessor.transform(time_series_data)
    # Convert the transformed data back to a DataFrame for better readability
    # The output column names after PCA will be 'PC1', 'PC2', ..., 'PCn' where n is the number of components
    pca_feature_names = [f'PC{i+1}' for i in range(transformed_data.shape[1] - len(unchanged_column_indices) - 1)]
    final_column_names = pca_feature_names + ['Throttle (%)'] + [data_column_names[i] for i in unchanged_column_indices]

    final_df = pd.DataFrame(transformed_data, columns=final_column_names)
    return final_df
    
def read_files(lst) :
    time_series_data = np.empty((0, 23))
    for idx, x in enumerate(lst) :
        if x.lower().endswith('.csv') :
            df = preprocess(pd.read_csv(x))
        else :
            df = preprocess(pd.read_excel(x))
        if df is not None and len(df) > 0 :
            df['flightId'] = idx
            
            df = create_time_series_features(df)        
            time_series_data = np.concatenate((time_series_data, df.values), axis=0)
    return time_series_data

def predict(lst) :
    time_series_data = read_files(lst)
    final_df = scale_n_process(time_series_data)
    
    with open(get_resource_path('voting_classifier_model.pkl'), 'rb') as file:
        voting_clf = pickle.load(file)
        
    flightId_grouped = final_df.groupby('flightId').size()
    y_pred = voting_clf.predict(final_df.drop('flightId', axis=1))
    final_y = []
    start_idx = 0
    for idx, val in flightId_grouped.items() :
        
        dataset_pred = y_pred[start_idx: start_idx + val]
        mode = pd.Series(dataset_pred).value_counts().sort_values(ascending=False).index[0]
        final_y.append(mode)
        start_idx += val
    return final_y

if __name__ == '__main__' :
    path1 = r'C:\Python Programs\Tech-Troopers\Dhruv\Drone\logs\selected\-1\50 throttle not enough power(annotated).xlsx'
    path3 = r'C:\Python Programs\Tech-Troopers\Dhruv\Drone\logs\selected\-1\NoEngineCranking\B3P 041\LOG00000\LOG.csv'
    path2 = r'C:\Python Programs\Tech-Troopers\Dhruv\Drone\logs\selected\1\1\LOG00000\LOG.csv'
    path4 = r'C:\Python Programs\Tech-Troopers\Dhruv\Drone\logs\selected\1\1\LOG00001\LOG.csv'
    
    print(predict([path1, path2, path3, path4]))
    
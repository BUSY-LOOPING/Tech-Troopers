import pickle
import os
import numpy as np
from drone_troopers.preprocessing import preprocess, create_time_series_features, add_lags
from drone_troopers import COLUMNS_CONSIDERED
import pandas as pd

def get_resource_path(filename):
    """Get the absolute path of a resource file."""
    return os.path.join(os.path.dirname(__file__), filename)

def get_filename_from_path(file_path, without_extension=False):
    """
    Extracts the filename from a given file path.
    
    Args:
    file_path (str): The full path to the file.
    
    Returns:
    str: The filename extracted from the path.
    """
    file_name = os.path.basename(file_path)
    if without_extension :
        file_name, _ = os.path.splitext(file_name)
    return file_name


def read_files(lst) :
    final_df = None
    for idx, x in enumerate(lst) :
        if x.lower().endswith('.csv') :
            df = preprocess(pd.read_csv(x))
        else :
            df = preprocess(pd.read_excel(x))
        df['flightId'] = idx
        
        df = create_time_series_features(df) 
        df = df.reset_index(drop=True)

        for col in COLUMNS_CONSIDERED:
            df = add_lags(df, col, range(1, 10), fill_na=True)

        if final_df is None :
            final_df = df.copy()
        else :
            final_df = pd.concat([final_df, df], axis=0)


    return final_df

with open(get_resource_path(os.path.join("std_scaler_samples_model.pkl")),"rb") as file:
    std_scaler = pickle.load(file)

def scale_n_process(df) :
    return std_scaler.transform(df)

def inverse_transform(data_scaled, columns):
    df_original = std_scaler.inverse_transform(data_scaled)
    return pd.DataFrame(df_original, columns=columns)

def predict(lst) :
    final_df = read_files(lst)
    flight_Ids = final_df['flightId']
    final_df = final_df.drop(columns='flightId')
    columns = final_df.columns
    scaled_data = scale_n_process(final_df)
    
    scaled_data = pd.DataFrame(scaled_data, columns=columns)
    # scaled_data['flightId'] = flight_Ids

    with open(get_resource_path(os.path.join("voting_classifier_samples_model.pkl")), "rb") as file:
        model = pickle.load(file)
    y_pred = model.predict(scaled_data.values)
    scaled_data['alert'] = y_pred

     # Group by flightId
    grouped_predictions = scaled_data.groupby(flight_Ids)

    # Create a dictionary to hold dataframes for each flightId
    prediction_dfs = {flight_id: group for flight_id, group in grouped_predictions}
    return [x['alert'].values for x in list(prediction_dfs.values())]

def predict_n_save(lst, save_paths=None) :
    if save_paths is not None :
        assert len(lst) == len(save_paths), 'Saved paths list must be same length as input list'
    final_df = read_files(lst)
    flight_Ids = final_df['flightId']
    final_df = final_df.drop(columns='flightId')
    columns = final_df.columns
    scaled_data = scale_n_process(final_df)
    scaled_data = pd.DataFrame(scaled_data, columns=columns)

    with open(get_resource_path(os.path.join("..", "models", "voting_classifier_samples_model.pkl")), "rb") as file:
        model = pickle.load(file)
    y_pred = model.predict(scaled_data.values)
    scaled_data['alert'] = y_pred

     # Group by flightId
    grouped_predictions = scaled_data.groupby(flight_Ids)

    # Create a dictionary to hold dataframes for each flightId
    for idx, (flight_id, group) in enumerate(grouped_predictions) :
        if save_paths is not None :
            path_to_save = save_paths[idx]
        else :
            file_name = get_filename_from_path(lst[idx], without_extension=True)
            path_to_save = f'{file_name}_annotated.csv'

        group_transformed = inverse_transform(group.drop(columns='alert').values, columns)
        group_transformed = group_transformed[COLUMNS_CONSIDERED]
        group = pd.concat((group_transformed, group['alert']), axis=1)
        group.to_csv(path_to_save, index=False)
        print(f'Saved {path_to_save}')


if __name__ == '__main__' :
    path1 = r'C:\Python Programs\Tech-Troopers\Dhruv\Drone\logs\selected\-1\50 throttle not enough power(annotated).xlsx'
    path3 = r'C:\Python Programs\Tech-Troopers\Dhruv\Drone\logs\selected\-1\NoEngineCranking\B3P 041\LOG00000\LOG.csv'
    path2 = r'C:\Python Programs\Tech-Troopers\Dhruv\Drone\logs\selected\1\1\LOG00000\LOG.csv'
    path4 = r'C:\Python Programs\Tech-Troopers\Dhruv\Drone\logs\selected\1\1\LOG00001\LOG.csv'
    
    # print(predict([path2]))
    predict_n_save([path2], ['path2_annotated.csv'])






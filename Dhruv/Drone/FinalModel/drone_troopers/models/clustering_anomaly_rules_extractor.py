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
import warnings
warnings.filterwarnings('ignore')

def get_resource_path(filename):
    """Get the absolute path of a resource file."""
    return os.path.join(os.path.dirname(__file__), filename)

def generate_rules(row, cluster_centers, anomaly_cluster, original_features, scaler):
    if row['is_anomaly']:
        # Inverse transform the cluster centers and the row
        inverse_cluster_centers = scaler.inverse_transform(cluster_centers)
        inverse_row = scaler.inverse_transform([row[original_features].values])[0]
        
        anomaly_center = inverse_cluster_centers[anomaly_cluster]
        other_centers = np.delete(inverse_cluster_centers, anomaly_cluster, axis=0)
        
        rules = []
        for i, feature in enumerate(original_features):
            feature_value = inverse_row[i]
            mean_non_anomaly = np.mean(other_centers[:, i])
            std_non_anomaly = np.std(other_centers[:, i])
            
            if feature_value < mean_non_anomaly - std_non_anomaly:
                rule = f"{feature} was below {mean_non_anomaly - std_non_anomaly:.2f}"
                rules.append(rule)
            elif feature_value > mean_non_anomaly + std_non_anomaly:
                rule = f"{feature} was above {mean_non_anomaly + std_non_anomaly:.2f}"
                rules.append(rule)
                
        return ', '.join(rules)
    return np.nan

def get_tree_rules(tree, sample, feature_names):
    if sample['is_anomaly'] :
        sample = sample[feature_names].values
        node_indicator = tree.decision_path(sample.reshape(1, -1))
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold

        # Collect rules
        rules = []
        for node in range(node_indicator.shape[1]):
            if node_indicator[0, node]:
                if feature[node] != -2:  # -2 indicates a leaf node
                    rule = f"{feature_names[feature[node]]} <= {threshold[node]:.2f}" if sample[feature[node]] <= threshold[node] else f"{feature_names[feature[node]]} > {threshold[node]:.2f}"
                    rules.append(rule)
        return ' AND '.join(rules)
    return np.nan


def read_df(lst) :
    original_df = pd.DataFrame(dtype=np.float32)
    for idx, x in enumerate(lst) :
        if x.lower().endswith('.csv') :
            df = pd.read_csv(x)
        else :
            df = pd.read_excel(x)
        if df is not None and len(df) > 0:
            df['flightId'] = idx
            
            original_df = pd.concat((original_df, df), axis=0, ignore_index=True)

    return original_df


def get_anomaly_n_rules(lst) :
    original_df = read_df(lst)
    flight_Ids = original_df['flightId']

    processed_df, corrupted_idx = preprocess(original_df.copy(), return_corrupted=True)
    flight_Ids = flight_Ids.drop(index=corrupted_idx)
    original_df = original_df.drop(index=corrupted_idx)

    processed_df = create_time_series_features(processed_df)
    processed_df = processed_df.reset_index(drop=True)
    processed_df_columns = processed_df.columns

    df_scaled = scaler.transform(processed_df)

    processed_df['cluster'] = kmeans.predict(df_scaled)

    # Label anomalies
    anomaly_clusters = [4, 3]
    processed_df['is_anomaly'] = processed_df['cluster'].isin(anomaly_clusters)

    # Calculate cluster centers
    cluster_centers = kmeans.cluster_centers_

    anomalies = processed_df[processed_df['is_anomaly']]

    if len(anomalies) > 0 :
        processed_df['general_rules'] = processed_df.apply(generate_rules, axis=1, cluster_centers=cluster_centers, anomaly_cluster=anomaly_clusters, original_features=processed_df_columns, scaler=scaler).values
        
        processed_df['tree_rules'] = processed_df.apply(lambda row: get_tree_rules(dt_classifier, row, processed_df_columns), axis=1).values
        
        # Group by flightId
        grouped_predictions = processed_df.groupby(flight_Ids)

        # Create a dictionary to hold dataframes for each flightId
        prediction_dfs = {flight_id: group for flight_id, group in grouped_predictions}


        return [group[group['is_anomaly'] == 1][['general_rules', 'tree_rules']] for group in prediction_dfs.values()]
    return []




with open(get_resource_path(os.path.join('clustering_anomaly', 'scaler.pkl')), 'rb') as file:
    scaler = pickle.load(file)

with open(get_resource_path(os.path.join('clustering_anomaly', 'kmeans_model.pkl')), 'rb') as file:
    kmeans = pickle.load(file)

with open(get_resource_path(os.path.join('clustering_anomaly', 'dt_model.pkl')), 'rb') as file:
    dt_classifier = pickle.load(file)

if __name__ == '__main__' :
    path1 = r'C:\Python Programs\Tech-Troopers\Dhruv\Drone\logs\selected\-1\50 throttle not enough power(annotated).xlsx'
    path3 = r'C:\Python Programs\Tech-Troopers\Dhruv\Drone\logs\selected\-1\NoEngineCranking\B3P 041\LOG00000\LOG.csv'
    path2 = r'C:\Python Programs\Tech-Troopers\Dhruv\Drone\logs\selected\1\1\LOG00000\LOG.csv'
    path4 = r'C:\Python Programs\Tech-Troopers\Dhruv\Drone\logs\selected\1\1\LOG00001\LOG.csv'

    print(get_anomaly_n_rules([path3]))
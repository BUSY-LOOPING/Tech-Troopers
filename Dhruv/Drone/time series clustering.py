#!/usr/bin/env python
# coding: utf-8

# In[106]:


from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import os
import re
from kneed import KneeLocator
import pickle

from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax, TimeSeriesResampler
from tslearn.clustering import TimeSeriesKMeans, KShape, KernelKMeans
from tslearn.utils import to_time_series_dataset
from dtaidistance import dtw, clustering
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.metrics import silhouette_score, silhouette_samples

from scipy.cluster.hierarchy import dendrogram, linkage


# In[7]:


def get_file_paths(directory, extensions=['.csv', '.xlsx']):
    file_paths = []
    # Iterate over each extension
    for ext in extensions:
        # Use glob to find all files with the given extension in the directory and subdirectories
        file_paths.extend(glob(os.path.join(directory, '**', f'*{ext}'), recursive=True))
    return file_paths

# Specify the root directory
root_directory = 'logs'

# Get the list of file paths
file_paths = get_file_paths(root_directory)
print(f'Found {len(file_paths)} log files.')


# In[86]:


column_patterns = {
    'Time (s)': r'Time.*s',
    'Motor Speed (RPM)': r'Motor.*RPM',
    'Engine Speed (RPM)': r'Engine.*Speed.*RPM',
    'Throttle (%)': r'Throttle.*%',
    'Intake Temperature (C)': r'\s*Intake\s*Temp(?:erature)?\s*\(\s*C\s*\)',
    'Engine Coolant Temperature 1 (C)': r'\s*Engine\s*Coolant\s*(?:Temperature|Temp)\s*1?\s*\(\s*C\s*\)',
    'Engine Coolant Temperature 2 (C)': r'\s*Engine\s*Coolant\s*(?:Temperature|Temp)\s*2?\s*\(\s*C\s*\)',
    'Barometric Pressure (kpa)': r'Barometric.*Pressure.*kpa',
    'Fuel Trim': r'Fuel.*Trim',
    'Fuel Consumption (g/min)': r'Fuel.*Consumption.*g.*min',
    'Fuel Consumed (g)': r'Fuel.*Consumed.*g',
    'Expected Max Power (W)': r'Expected.*Max.*Power.*W',
    'Bus Voltage (V)': r'Bus.*Voltage.*V',
    'Battery Current (A)': r'Battery.*Current.*A',
    'Power Generated (W)': r'Power.*Generated.*W',
    'Inverter Temperature (C)': r'\s*Inverter\s*(?:Temperature|MAX)\s*\(\s*C\s*\)',
    'Target Fuel Pressure (bar)': r'Target.*Fuel.*Pressure.*bar',
    'Fuel Pressure (bar)': r'Fuel.*Pressure.*bar',
    'Fuel Pump Speed (RPM)': r'Fuel.*Pump.*Speed.*RPM',
    'Cooling Pump Speed (RPM)': r'Cooling.*Pump.*Speed.*RPM',
}

def standardize_columns(df, column_patterns):
    standardized_columns = {}
    for standard_col, pattern in column_patterns.items():
        for col in df.columns:
            if re.match(pattern, col, re.IGNORECASE):
                standardized_columns[col] = standard_col
                break
    df = df.rename(columns=standardized_columns)
    
    return df[list(column_patterns.keys())]

def preprocess(df) :
    df = df.copy()
    df = standardize_columns(df, column_patterns)
    if df['Throttle (%)'].max() < 80 :
        return None
    reference_date = pd.to_datetime('1970-01-01')
    df['Time (s)'] = reference_date + pd.to_timedelta(df['Time (s)'], unit='s')
    df = df.set_index('Time (s)')
    df = df.dropna()
    # df = df.resample('1S').mean()
    return df

def create_time_series_features(df) :
    df = df.copy()
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['second'] = df.index.second
    df['microsecond'] = df.index.microsecond
    return df


# In[87]:


time_series_data = []
file_paths_considered = []

for x in file_paths :
    if len(time_series_data) >= 200 :
        break
    if x.lower().endswith('.csv') :
        df = preprocess(pd.read_csv(x))
    else :
        df = preprocess(pd.read_excel(x))
    if df is not None:
        file_paths_considered.append(x)
        time_series_data.append(df.values)


# In[88]:


X = to_time_series_dataset(time_series_data)
X.shape


# In[91]:


X_scaled = TimeSeriesResampler(sz=1000).fit_transform(X)
X_scaled.shape


# In[92]:


X_scaled = TimeSeriesScalerMinMax().fit_transform(X_scaled)
X_scaled.shape


# In[36]:


# knn = KNeighborsTimeSeriesClassifier(n_neighbors=1)


# In[93]:


# Assuming X_scaled is your time series data
wcss = []
k_values = range(1, 11)  # Adjust the range as needed

for k in k_values:
    model = TimeSeriesKMeans(n_clusters=k, metric="dtw", verbose=0, random_state=0)
    model.fit(X_scaled)
    wcss.append(model.inertia_)

# Plotting the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(k_values, wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.show()


# In[ ]:


# Automatically find the "elbow" point
kneedle = KneeLocator(k_values, wcss, curve='convex', direction='decreasing')
n_clusters = kneedle.elbow

print(f'The optimal number of clusters is: {n_clusters}')


# In[97]:


n_clusters = 4
model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", verbose=True, random_state=0)
model.fit(X_scaled)


# In[98]:


with open('timeseries_kmeans_model.pkl', 'wb') as f:
    pickle.dump(model, f)


# In[ ]:


# To load the model back
# with open('timeseries_kmeans_model.pkl', 'rb') as f:
#     model = pickle.load(f)


# In[99]:


# Step 3: Get cluster labels
cluster_labels = model.labels_


# In[109]:


# Reshape X for silhouette score calculation
X_reshaped = X_scaled.reshape(X.shape[0], -1)

# Compute the silhouette score for each sample
silhouette_vals = silhouette_samples(X_reshaped, cluster_labels)

# Compute the overall silhouette score
overall_silhouette_score = silhouette_score(X_reshaped, cluster_labels)
print(f'Overall Silhouette Score: {overall_silhouette_score}')


# In[110]:


def get_top_series_per_cluster(X, labels, silhouette_vals, top_n=5):
    top_series_indices = []
    for cluster in range(n_clusters):
        # Get indices of samples in the current cluster
        cluster_indices = np.where(labels == cluster)[0]
        # Get silhouette scores for these samples
        cluster_silhouette_vals = silhouette_vals[cluster_indices]
        # Get indices of the top_n samples with the highest silhouette scores
        top_indices = cluster_indices[np.argsort(cluster_silhouette_vals)[-top_n:]]
        top_series_indices.append(top_indices)
    return top_series_indices

top_series_indices = get_top_series_per_cluster(X_scaled, cluster_labels, silhouette_vals)


# In[117]:


def plot_clusters_for_column(X, labels, n_clusters, column_index, top_series_indices, model):
    plt.figure(figsize=(15, 10))
    for cluster in range(n_clusters):
        plt.subplot(n_clusters, 1, cluster + 1)
        for idx in top_series_indices[cluster]:
            plt.plot(X[idx, :, column_index].flatten(), "k-", alpha=0.5)
        plt.plot(model.cluster_centers_[cluster, :, column_index].flatten(), "r-", linewidth=2)
        plt.title(f"Cluster {cluster + 1} for {list(column_patterns.keys())[column_index + 1]}")
    plt.tight_layout()
    plt.show()


# In[118]:


num_columns = X_scaled.shape[2]
for col in range(num_columns):
    plot_clusters_for_column(X_scaled, cluster_labels, n_clusters, col, top_series_indices, model)


# In[ ]:


## KShape
## KernelKMeans


# In[119]:


model_kshape = KShape(n_clusters=n_clusters, verbose=True, random_state=0)
model_kshape.fit(X_scaled)


# In[120]:


with open('kshape_model.pkl', 'wb') as f:
    pickle.dump(model_kshape, f)


# In[121]:


cluster_labels_kshape = model_kshape.labels_


# In[122]:


# Reshape X for silhouette score calculation
X_reshaped = X_scaled.reshape(X.shape[0], -1)

# Compute the silhouette score for each sample
silhouette_vals_kshape = silhouette_samples(X_reshaped, cluster_labels_kshape)

# Compute the overall silhouette score
overall_silhouette_score_kshape = silhouette_score(X_reshaped, cluster_labels_kshape)
print(f'Overall Silhouette Score: {overall_silhouette_score_kshape}')


# In[128]:


top_series_indices_kshape = get_top_series_per_cluster(X_scaled, cluster_labels_kshape, silhouette_vals_kshape, top_n=5)


# In[129]:


top_series_indices_kshape


# In[124]:


num_columns = X_scaled.shape[2]
for col in range(num_columns):
    plot_clusters_for_column(X_scaled, cluster_labels_kshape, n_clusters, col, top_series_indices_kshape, model_kshape)


# In[130]:


model_kernel_kmeans = KernelKMeans(n_clusters=n_clusters, kernel="gak", random_state=0)
model_kernel_kmeans.fit(X_scaled)


# In[131]:


with open('kernel_kmeans_model.pkl', 'wb') as f:
    pickle.dump(model_kernel_kmeans, f)


# In[132]:


cluster_labels_kernel_kmeans = model_kernel_kmeans.labels_


# In[133]:


# Reshape X for silhouette score calculation
X_reshaped = X_scaled.reshape(X.shape[0], -1)

# Compute the silhouette score for each sample
silhouette_vals_kernel_kmeans = silhouette_samples(X_reshaped, cluster_labels_kernel_kmeans)

# Compute the overall silhouette score
overall_silhouette_score_kernel_kmeans = silhouette_score(X_reshaped, cluster_labels_kernel_kmeans)
print(f'Overall Silhouette Score: {overall_silhouette_score_kernel_kmeans}')


# In[ ]:


top_series_indices_kernel_kmeans = get_top_series_per_cluster(X_scaled, cluster_labels_kernel_kmeans, silhouette_vals_kernel_kmeans, top_n=5)


# In[ ]:


num_columns = X_scaled.shape[2]
for col in range(num_columns):
    plot_clusters_for_column(
        X_scaled, 
        cluster_labels_kernel_kmeans, 
        n_clusters, 
        col, 
        top_series_indices_kernel_kmeans, 
        model_kernel_kmeans
    )


# In[ ]:





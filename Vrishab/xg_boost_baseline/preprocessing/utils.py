import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder

def remove_pseudo_categorical(X, y):
    """Remove columns where most values are the same"""
    pseudo_categorical_cols_mask = X.nunique() < 10
    print("Removed {} columns with pseudo-categorical values on {} columns".format(sum(pseudo_categorical_cols_mask),
                                                                                   X.shape[1]))
    X = X.drop(X.columns[pseudo_categorical_cols_mask], axis=1)
    return X, y

def remove_rows_with_missing_values(X, y):
    missing_rows_mask = pd.isnull(X).any(axis=1)
    print("Removed {} rows with missing values on {} rows".format(sum(missing_rows_mask), X.shape[0]))
    X = X[~missing_rows_mask]
    y = y[~missing_rows_mask]
    
    return X, y

def remove_missing_values(X, y, threshold=0.7):
    """Remove columns where most values are missing, then remove any row with missing values"""
    missing_cols_mask = pd.isnull(X).mean(axis=0) > threshold
    print("Removed {} columns with missing values on {} columns".format(sum(missing_cols_mask), X.shape[1]))
    X = X.drop(X.columns[missing_cols_mask], axis=1)
    missing_rows_mask = pd.isnull(X).any(axis=1)
    print("Removed {} rows with missing values on {} rows".format(sum(missing_rows_mask), X.shape[0]))
    X = X[~missing_rows_mask]
    y = y[~missing_rows_mask]
    
    return X, y
    
def remove_high_cardinality(X, y, categorical_mask, threshold=20):
    high_cardinality_mask = (X.nunique() > threshold).values
    print("high cardinality columns: {}".format(X.columns[high_cardinality_mask * categorical_mask]))
    n_high_cardinality = sum(categorical_mask * high_cardinality_mask)
    X = X.drop(X.columns[categorical_mask * high_cardinality_mask], axis=1)
    print("Removed {} high-cardinality categorical features".format(n_high_cardinality))
    categorical_mask = [categorical_mask[i] for i in range(len(categorical_mask)) if not (high_cardinality_mask[i] and categorical_mask[i])]

    return X, y

def balance(X, y) :
    freq_count = y.value_counts().sort(ascending=False)
    X = X[y in freq_count.index[:2]]
    y = y[y in freq_count.index[:2]]
    
def transform_target(y, keyword):
    if keyword == "log":
        return np.sign(y) * np.log(1 + np.abs(y))
    elif keyword == "none":
        return y
    elif pd.isnull(keyword):
        return y
    
def is_heavy_tailed(data):
    """
    Checks if the distribution of the given data is heavy-tailed.
    
    Parameters:
    data (array-like): The data for which to check the tail behavior.
    
    Returns:
    bool: True if the distribution is heavy-tailed, False otherwise.
    """
    skewness = skew(data)
    kurt = kurtosis(data)
    
    # Thresholds for skewness and kurtosis to determine if distribution is heavy-tailed
    skew_threshold = 0
    kurtosis_threshold = 3
    
    return skewness > skew_threshold or kurt > kurtosis_threshold
    
    
def preprocess_data(X, y, isCategorical=False):
    if isCategorical :
        le = LabelEncoder()
        y = le.fit_transform(y)
    elif is_heavy_tailed(y) :
        y = transform_target(y, keyword='log')


    X, y = remove_rows_with_missing_values(X, y)
    X, y = remove_missing_values(X, y)
    X, y = remove_pseudo_categorical(X, y)
    categorical_mask = [(X[col].dtype == 'object' or len(X[col].unique()) < 20) for col in X.columns]
    X, y = remove_high_cardinality(X, y, categorical_mask)
    
    return X, y

def split_data(X, y, isCategorical=None):
    # Shuffle the data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    try :
        X = X.values
    except :
        pass
    
    try :
        y = y.values
    except :
        pass
    X = X[indices]
    y = y[indices]
    
    stratify = None if not isCategorical else y
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42, stratify=stratify)
    stratify = None if not isCategorical else y_temp
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.7, random_state=42, stratify=stratify)
    
    num_val = min(X_val.shape[0], 50000)
    num_test = min(X_test.shape[0], 50000)
    
    X_val, y_val = X_val[:num_val], y_val[:num_val]
    X_test, y_test = X_test[:num_test], y_test[:num_test]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def plot_distribution(data):
    """
    Plots the distribution of the given data.
    
    Parameters:
    data (array-like): The data to be plotted.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=30, color='blue', alpha=0.7)
    plt.title('Distribution Plot')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
    
def affine_renormalization_classification(results):
    """
    Perform affine renormalization on classification results.
    
    Parameters:
    results (list): List of original classification results between 0 and 1.
    
    Returns:
    list: List of renormalized classification results between 0 and 1.
    """
    # Find the top-performing model's accuracy
    top_accuracy = max(results)
    
    # Find the accuracy corresponding to the 10th percentile
    quantile_accuracy = np.percentile(results, 10)
    
    # Calculate the range of accuracies for renormalization
    range_accuracy = top_accuracy - quantile_accuracy
    
    # Perform affine renormalization for each accuracy
    renormalized_results = [(accuracy - quantile_accuracy) / range_accuracy for accuracy in results]
    
    return renormalized_results
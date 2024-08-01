#!/usr/bin/env python
# coding: utf-8

# In[257]:


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

plt.style.use('fivethirtyeight')

import warnings
warnings.filterwarnings('ignore')


# In[163]:


df = pd.read_excel('50 throttle not enough power(annotated).xlsx')
df.head()


# In[164]:


reference_date = pd.to_datetime('1970-01-01')
df['Time (s)'] = reference_date + pd.to_timedelta(df['Time (s)'], unit='s')
df['Time (s)'].dtype


# In[165]:


df = df.set_index('Time (s)')


# In[166]:


print(f'The samples are collected almost every {df.index.diff().mean().total_seconds()} seconds.')


# In[167]:


print(f'The samples are collected for almost {df.index.max().minute} minutes.')


# In[168]:


df['Engine Shut Command'] = df[' PWM Uptime (s)'].map(lambda x : x == 0).astype(np.int32)
df = df.drop(columns=[' PWM Uptime (s)', ' Fans On (qty)'])


# ## Scaling data

# In[169]:


min_max = {
    'Motor Speed (RPM)' : (-1, 8800),
    'Engine Speed (RPM)' : (3000, 8192),
    'Throttle (%)' : (5, 100),
    'Intake Temperature (C)' : (-40, 60),
    'Engine Coolant Temperature 1 (C)' : (-40, 91),
    'Engine Coolant Temperature 2 (C)' : (-40, 91),
    'Barometric Pressure (kpa)' : (70, 110),
    'Fuel Trim' : (0.8, 1.2),
    'Fuel Consumption (g/min)' : None,  #derived value based on the time the injector is open and the fuel pressure
    'Fuel Consumed (g)' : None,
    'Expected BSFC (g/kW.hr)' : None,
    'Actual BSFC (g/kW.hr)' : None,
    'Expected Max Power (W)' : None, #expected power under current circumstances if throttle at 100
    'Bus Voltage (V)' : (42, 50.4),
    'GCU Current (A)' : (-20, 80),
    'Battery Current (A)' : (-30, 30), #current that is coming out of or going into the battery
    'Power Generated (W)' : None,
    'Inverter Temperature (C)' : None,
    'Target Fuel Pressure (bar)' : (1, 4.5),
    'Fuel Pressure (bar)' : (1, 4.5),
    'Fuel Pump Speed (RPM)' : None,
    'Cooling Pump Speed (RPM)' : None
}


# In[170]:


def scale_column(column, min_max):
    if min_max is None:
        scaler = StandardScaler()
        scaled_column = scaler.fit_transform(column.values.reshape(-1, 1))
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(min_max.reshape(-1, 1))
        
        scaled_column = scaler.transform(column.values.reshape(-1, 1))
    return scaled_column.flatten()

for column, min_max_range in min_max.items():
    df[column] = scale_column(df[column], np.array(list(min_max_range)) if min_max_range else None)


# In[171]:


df.head()


# ## Time Series Cross Validation

# In[172]:


from sklearn.model_selection import TimeSeriesSplit


# In[173]:


n_splits = 5
tss = TimeSeriesSplit(n_splits=n_splits, test_size=5*60*4, gap=int(0.5*60*4)) #predict 5 min ahead, gap = 2 minutes
df = df.sort_index()


# In[174]:


def plot_splits_for_col(col_name) :
    fig, axs = plt.subplots(n_splits, 1, figsize=(15, 15), sharex=True)
    
    fold = 0
    for train_idx, val_idx in tss.split(df):
        train = df.iloc[train_idx]
        test = df.iloc[val_idx]
        train[col_name].plot(ax=axs[fold],
                              label='Training Set',
                              title=f'Data Train/Test Split Fold {fold}')
        test[col_name].plot(ax=axs[fold],
                             label='Test Set')
        axs[fold].axvline(test.index.min(), color='black', ls='--')
        axs[fold].legend()
        fold += 1
    plt.show()


# In[175]:


plot_splits_for_col('Engine Speed (RPM)')


# ## Forecasting horizon
# Length of time into the future for which forecasts are to be prepared

# In[176]:


def create_time_series_features(df) :
    df = df.copy()
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['second'] = df.index.second
    df['microsecond'] = df.index.microsecond
    return df


# In[177]:


df = create_time_series_features(df)


# ## Lag Features

# In[178]:


target_cols = list(min_max.keys())
target_cols.remove('Throttle (%)')


# In[179]:


def add_lags(df, col_name):
    shift = 60*4
    df[f'{col_name}_lag1'] = df[col_name].shift(2*shift)  # Shift by 2 minutes
    df[f'{col_name}_lag2'] = df[col_name].shift(4*shift)  # Shift by 4 minutes
    df[f'{col_name}_lag3'] = df[col_name].shift(8*shift)  # Shift by 8 minutes
    return df


# In[180]:


for x in target_cols :
    df = add_lags(df, x)


# ## Training using Cross Validation

# In[181]:


# fold = 0
# preds = []
# scores = []
# lag_pattern = re.compile(r'_lag[123]$')

# for fold, (train_idx, val_idx) in enumerate(tss.split(df)):
#     train = df.iloc[train_idx]
#     test = df.iloc[val_idx]

#     train = create_time_series_features(train)
#     test = create_time_series_features(test)
#     FEATURES = ['Throttle (%)', 'Engine Shut Command']
#     FEATURES += df.columns[df.columns.str.contains(lag_pattern)].tolist()
#     TARGET = target_cols.copy()

#     X_train = train[FEATURES]
#     y_train = train[TARGET]

#     X_test = test[FEATURES]
#     y_test = test[TARGET]

#     reg = MultiOutputRegressor(xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
#                            n_estimators=5000,
#                            # early_stopping_rounds=100,
#                            objective='reg:squarederror',
#                            # max_depth=3,
#                            learning_rate=0.01))
#     reg.fit(X_train, y_train,
#             # eval_set=[(X_train, y_train), (X_test, y_test)],
#             verbose=100)

#     y_pred = reg.predict(X_test)
#     preds.append(y_pred)
#     score = np.sqrt(mean_squared_error(y_test, y_pred))
#     scores.append(score)
    
#     print(f"Fold {fold} Score: {score}")


# In[182]:


fold = 0
preds = []
scores = []
lag_pattern = re.compile(r'_lag[123]$')

for fold, (train_idx, val_idx) in enumerate(tss.split(df)):
    train = df.iloc[train_idx]
    test = df.iloc[val_idx]

    train = create_time_series_features(train)
    test = create_time_series_features(test)

    FEATURES = ['Throttle (%)', 'Engine Shut Command', 'hour', 'minute', 'second', 'microsecond']
    FEATURES += df.columns[df.columns.str.contains(lag_pattern)].tolist()
    TARGET = target_cols.copy()

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    # Convert the dataset into DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Define parameters for XGBoost
    params = {
        'base_score': 0.5,
        'booster': 'gbtree',
        'objective': 'reg:squarederror',
        'learning_rate': 0.01,
        'n_estimators': 5000,
        'early_stopping_rounds': 100,
        # 'max_depth': 3,  # Uncomment and adjust based on your needs
    }

    # Train the model
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    reg = xgb.train(params, dtrain, num_boost_round=params['n_estimators'], evals=evals, early_stopping_rounds=params['early_stopping_rounds'], verbose_eval=100)

    # Predict
    y_pred = reg.predict(dtest)
    preds.append(y_pred)
    
    # Reshape y_test to match the predicted shape if necessary
    if len(y_test.shape) == 1:
        y_test = y_test.reshape(-1, 1)
        
    # Calculate score
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    scores.append(score)
    
    print(f"Fold {fold} Score: {score}")

print("Final Scores:", scores)
print("Average Score:", np.mean(scores))


# In[183]:


print(f'Score across folds {np.mean(scores):0.4f}')
print(f'Fold scores:{scores}')


# ## Retraining the model

# In[184]:


# Retrain on all data
df = create_time_series_features(df)

X_all = df[FEATURES]
y_all = df[TARGET]

# Convert the dataset into DMatrix for XGBoost
dtrain = xgb.DMatrix(X_all, label=y_all)

# Define parameters for XGBoost
params = {
    'base_score': 0.5,
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    'learning_rate': 0.01,
    'n_estimators': 500,
    'early_stopping_rounds': 100,
    # 'max_depth': 3,  # Uncomment and adjust based on your needs
}

# Train the model
evals = [(dtrain, 'train'), (dtrain, 'eval')]
reg = xgb.train(params, 
                dtrain, 
                num_boost_round=params['n_estimators'], 
                evals=evals, 
                early_stopping_rounds=params['early_stopping_rounds'], 
                verbose_eval=100)


# # TODO

# In[121]:


FEATURES = ['Throttle (%)', 'Engine Shut Command']

mask = df['Motor Speed (RPM)_lag3'].isna() & df['Motor Speed (RPM)_lag3'].shift(-1).notna()
first_non_nan_index = mask.idxmax()
first_non_nan_index = df.index.get_loc(first_non_nan_index) + 1

pred_indices = df.iloc[first_non_nan_index:].index #skeleton indices
pred_df = pd.DataFrame(index=pred_indices)  
pred_df['isPred'] = True
df['isPred'] = False
df_and_pred = pd.concat([df[:first_non_nan_index], pred_df])
df_and_pred[FEATURES] = df[FEATURES].copy()
df_and_pred = create_time_series_features(df_and_pred)
for x in target_cols :
    df_and_pred = add_lags(df_and_pred, x)


# In[ ]:


FEATURES = ['Throttle (%)', 'Engine Shut Command']


# In[124]:


pred_w_features = df_and_pred.query('isPred').copy()


# In[58]:


FEATURES = ['Throttle (%)', 'Engine Shut Command']
lag_pattern = re.compile(r'_lag[123]$')
FEATURES += df.columns[df.columns.str.contains(lag_pattern)].tolist()

X = pred_w_features[FEATURES]
try :
    dmat_X = xgb.DMatrix(X)
    ypred = reg.predict(dmat_pred_w_features)
except :
    ypred = reg.predict(X)


# In[59]:


TARGET = target_cols.copy()
TARGET = [col + '_pred' for col in TARGET]

ypred_df = pd.DataFrame(ypred, columns=TARGET, index=pred_w_features.index)


# In[206]:


# Define batch size (2 minutes)
batch_size = int(2 * 60 * 4)  # 480 samples

FEATURES = ['Throttle (%)', 'Engine Shut Command', 'hour', 'minute', 'second', 'microsecond']
lag_pattern = re.compile(r'_lag[123]$')
FEATURES += df.columns[df.columns.str.contains(lag_pattern)].tolist()

TARGET = target_cols.copy()

predictions = []
index_list = []
for start in range(batch_size, len(df), batch_size):
    end = start + batch_size
    batch = df.iloc[start:end]
    batch = create_time_series_features(batch)
    for x in TARGET :
        batch = add_lags(batch, x)

    # # Drop rows with NaN values in the target columns
    # batch = batch.dropna(subset=target_columns)

    # if batch.empty:
    #     continue

    # Prepare the features (exclude the target columns)
    X_batch = batch[FEATURES]
    y_true = batch[TARGET]

    # Make predictions
    try :
        dmat_X = xgb.DMatrix(X_batch)
        y_pred = reg.predict(dmat_X)
    except :
        y_pred = reg.predict(dmat_X)

    # Store predictions
    predictions.extend(y_pred)
    index_list.extend(batch.index)


# In[208]:


y_pred_df = pd.DataFrame(predictions, index=index_list, columns=[f'{x}_pred' for x in TARGET])


# In[235]:


pred_w_X = pd.concat([df[FEATURES + TARGET], y_pred_df], axis=1)


# In[276]:


# columns_to_plot = [(col, f'{col}_pred') for col in TARGET]

# # Plot each pair of columns
# for col_true, col_pred in columns_to_plot:
#     if col_true in pred_w_X.columns and col_pred in pred_w_X.columns:
#         residuals = pred_w_X[col_true] - pred_w_X[col_pred]
        
#         # Compute mean and standard deviation of residuals
#         mean_residual = residuals.mean()
#         std_residual = residuals.std()

#         # Define threshold for anomalies (e.g., 3 standard deviations)
#         anomaly_threshold = 3 * std_residual

#         # Mark anomalies based on threshold
#         pred_w_X[f'{col_true}_anomaly'] = np.abs(residuals) > anomaly_threshold
        
        
#         fig, ax = plt.subplots(figsize=(10, 4))
#         plt.title(f'{col_true}')
#         pred_w_X[col_true].plot(linestyle='-', label='True', linewidth=1, alpha=1, ax=ax)
#         pred_w_X[col_pred].plot(linestyle='-', label='Pred', linewidth=1, c='orange', alpha=1, ax=ax)
        
#         # anomalies = pred_w_X[pred_w_X[f'{col_true}_anomaly']]
#         # plt.scatter(anomalies.index, anomalies[col_true], c='red', label='Anomaly', s=50)
        
#         in_anomaly = False
#         start_idx = None
#         alpha = 0.5
#         for idx, is_anomaly in pred_w_X[f'{col_true}_anomaly'].items():
#             if is_anomaly:
#                 if not in_anomaly:
#                     in_anomaly = True
#                     start_idx = idx
#             else:
#                 if in_anomaly:
#                     ax.axvspan(start_idx, idx, color='red', alpha=alpha)
#                     in_anomaly = False
#         # Check if anomaly continues until the end of the series
#         if in_anomaly:
#             ax.axvspan(start_idx, pred_w_X.index[-1], color='red', alpha=alpha)
        
#         handles, labels = ax.get_legend_handles_labels()

#         # manually define a new patch 
#         patch = mpatches.Patch(color='red', label='Anomaly', alpha=alpha)

#         # handles is a list, so append manual patch
#         handles.append(patch) 
        
#         plt.legend(handles=handles)
#         plt.show()


# In[280]:


columns_to_plot = [(col, f'{col}_pred') for col in TARGET]

# Create a common x-axis
fig, axs = plt.subplots(len(columns_to_plot), 1, figsize=(10, 4*len(columns_to_plot)), sharex=True)

# Iterate over each column pair and plot
for idx, (col_true, col_pred) in enumerate(columns_to_plot):
    if col_true in pred_w_X.columns and col_pred in pred_w_X.columns:
        residuals = pred_w_X[col_true] - pred_w_X[col_pred]
        
        # Compute mean and standard deviation of residuals
        mean_residual = residuals.mean()
        std_residual = residuals.std()

        # Define threshold for anomalies (e.g., 3 standard deviations)
        anomaly_threshold = 3 * std_residual

        # Mark anomalies based on threshold
        pred_w_X[f'{col_true}_anomaly'] = np.abs(residuals) > anomaly_threshold
        
        # Plotting on the respective axis
        ax = axs[idx]
        ax.set_title(f'{col_true}')
        
        # Plot true and predicted values
        # pred_w_X['Throttle (%)'].plot(linestyle='-', label='Throttle', c='brown', linewidth=1, alpha=0.5, ax=ax)
        pred_w_X[col_true].plot(linestyle='-', label='True', linewidth=1, alpha=1, ax=ax)
        pred_w_X[col_pred].plot(linestyle='-', label='Pred', linewidth=1, c='orange', alpha=1, ax=ax)
        
        # Plot anomalies using axvspan
        in_anomaly = False
        start_idx = None
        alpha = 0.5
        for idx, is_anomaly in pred_w_X[f'{col_true}_anomaly'].items():
            if is_anomaly:
                if not in_anomaly:
                    in_anomaly = True
                    start_idx = idx
            else:
                if in_anomaly:
                    ax.axvspan(start_idx, idx, color='red', alpha=alpha)
                    in_anomaly = False
        # Check if anomaly continues until the end of the series
        if in_anomaly:
            ax.axvspan(start_idx, pred_w_X.index[-1], color='red', alpha=alpha)
        
        handles, labels = ax.get_legend_handles_labels()

        # manually define a new patch 
        patch = mpatches.Patch(color='red', label='Anomaly', alpha=alpha)

        # handles is a list, so append manual patch
        handles.append(patch) 
        
        ax.legend(handles=handles)

# Adjust layout
plt.tight_layout()
plt.show()


# ## Saving the model

# In[243]:


reg.save_model('lagged_regressor.json')


# In[24]:


reg = xgb.XGBRegressor()
reg.load_model('lagged_regressor.json')


# In[25]:


del reg_new


# In[ ]:





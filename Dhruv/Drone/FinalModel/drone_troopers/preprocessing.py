import numpy as np
import re
import pandas as pd


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
    # 'Expected Max Power (W)': r'Expected.*Max.*Power.*W',
    'Bus Voltage (V)': r'Bus.*Voltage.*V',
    'Battery Current (A)': r'Battery.*Current.*A',
    'Power Generated (W)': r'Power.*Generated.*W',
    'Inverter Temperature (C)': r'\s*Inverter\s*(?:Temperature|MAX)\s*\(\s*C\s*\)',
    'Target Fuel Pressure (bar)': r'Target.*Fuel.*Pressure.*bar',
    'Fuel Pressure (bar)': r'Fuel.*Pressure.*bar',
    'Fuel Pump Speed (RPM)': r'Fuel.*Pump.*Speed.*RPM',
    'Cooling Pump Speed (RPM)': r'Cooling.*Pump.*Speed.*RPM',
}

def add_lags(df, col_name, lags, fill_na=False):
    df = df.copy()
    for lag in lags:
        df[f"{col_name}_lag_{lag}"] = df[col_name].shift(lag)
        if fill_na:
            df[f"{col_name}_lag_{lag}"] = df[f"{col_name}_lag_{lag}"].fillna(0)
    return df


def mark_state_ON(
    df, window_size=10, rolling_current_thresh=20, rolling_pow_thresh=30000
):
    df["State_ON"] = 0
    rolling_battery_current_sum = (
        df["Battery Current (A)"].rolling(window=window_size).sum()
    )
    rolling_power_generated_sum = (
        df["Power Generated (W)"].rolling(window=window_size).sum()
    )

    rc_exceeding_thresh = rolling_battery_current_sum > rolling_current_thresh
    pow_exceeding_thresh = rolling_power_generated_sum > rolling_pow_thresh

    index_exceeding = df.index[(rc_exceeding_thresh & pow_exceeding_thresh) == True]
    if len(index_exceeding) > 0:
        last_idx = df.index[-1]
        df.loc[index_exceeding[0] :, "State_ON"] = 1
        df.loc[last_idx:, "State_ON"] = 0

    return df


def find_consecutive_indices(arr):
    arr = np.asarray(arr)
    diff = np.diff(arr)
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]

    if arr[0] == 1:
        starts = np.insert(starts, 0, 0)
    if arr[-1] == 1:
        ends = np.append(ends, len(arr) - 1)

    result = []
    for start, end in zip(starts, ends):
        if start == end:
            start -= 1 if start > 0 else 0
        result.append((start, end))

    return result

def standardize_columns(df, column_patterns, retain_all_columns=False):
    standardized_columns = {}
    for standard_col, pattern in column_patterns.items():
        for col in df.columns:
            if re.match(pattern, col, re.IGNORECASE):
                standardized_columns[col] = standard_col
                break
    df = df.rename(columns=standardized_columns)
    
    return df if retain_all_columns else df[list(column_patterns.keys())]

def preprocess(df, filter_throttle_values=None, retain_all_columns=False) :
    df = df.copy()
    df = standardize_columns(df, column_patterns, retain_all_columns)
    if filter_throttle_values is not None and df['Throttle (%)'].max() < filter_throttle_values :
        return None
    if retain_all_columns :
        time_col_original = df['Time (s)']
    df = df.dropna()
    reference_date = pd.to_datetime('1970-01-01')
    df['Time (s)'] = reference_date + pd.to_timedelta(df['Time (s)'], unit='s')
    df = df.set_index('Time (s)')
    if retain_all_columns :
        df = pd.concat([time_col_original, df], axis=1)
    return df

def create_time_series_features(df) :
    df = df.copy()
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['second'] = df.index.second
    df['microsecond'] = df.index.microsecond
    return df
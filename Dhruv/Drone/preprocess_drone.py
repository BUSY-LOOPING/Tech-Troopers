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

def standardize_columns(df, column_patterns):
    standardized_columns = {}
    for standard_col, pattern in column_patterns.items():
        for col in df.columns:
            if re.match(pattern, col, re.IGNORECASE):
                standardized_columns[col] = standard_col
                break
    df = df.rename(columns=standardized_columns)
    
    return df[list(column_patterns.keys())]

def preprocess(df, filter_throttle_values=None) :
    df = df.copy()
    df = standardize_columns(df, column_patterns)
    if filter_throttle_values is not None and df['Throttle (%)'].max() < filter_throttle_values :
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
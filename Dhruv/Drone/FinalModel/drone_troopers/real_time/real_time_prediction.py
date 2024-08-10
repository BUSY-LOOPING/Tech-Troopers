import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from drone_troopers.preprocessing import preprocess, create_time_series_features
import pickle
import os
import numpy as np
import time
from drone_troopers import COLUMNS_CONSIDERED
import shap

def get_resource_path(filename):
    """Get the absolute path of a resource file."""
    return os.path.join(os.path.dirname(__file__), filename)


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

def model_predict(X):
    return model.predict_proba(X)

with open(get_resource_path(os.path.join("..", "models", "voting_classifier_samples_model.pkl")), "rb") as file:
    model = pickle.load(file)

with open(get_resource_path(os.path.join("..", "models", "std_scaler_samples_model.pkl")),"rb") as file:
    std_scaler = pickle.load(file)

# Variables to store the simulation parameters
START_IDX = 100  # Define the starting index for simulation
MAX_WINDOW_SIZE = 99  # Maximum number of rows in current_df
i = START_IDX  # Start from START_IDX

samples_label = []
data_model_columns = None

# Load initial data from Excel file
excel_file_path = r"C:\Python Programs\Tech-Troopers\Dhruv\Drone\logs\selected\-1\50 throttle not enough power(annotated).xlsx"
csv_file_path = r'C:\Python Programs\Tech-Troopers\Dhruv\Drone\logs\selected\-1\100% throttle no power.csv'
# df = preprocess(pd.read_excel(excel_file_path))
df = preprocess(pd.read_csv(csv_file_path))
df = mark_state_ON(df)
data_model = create_time_series_features(df.drop(columns='State_ON'))
data_model = data_model.reset_index(drop=True)

for col in COLUMNS_CONSIDERED:
    data_model = add_lags(data_model, col, range(1, 10), fill_na=True)
data_model_columns = data_model.columns
data_model = std_scaler.transform(data_model)
df['alert'] = model.estimators_[-1].predict(data_model)

# Calculate SHAP values
explainer = shap.Explainer(model.estimators_[-1], data_model)
shap_values = explainer(data_model)
del data_model
# del data_model

# Create a Dash web application
app = Dash(__name__, 
            meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ])

# Define the layout of the application with a 2x2 grid
app.layout = html.Div(
    [
        html.Div(
            [
                dcc.Graph(id=f"plot-{column}", style={"height": "400px"})
                for column in COLUMNS_CONSIDERED[:4]
            ],  # First row
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, 1fr)",
                "gap": "10px",
            },
        ),
        html.Div(
            [
                dcc.Graph(id=f"plot-{column}", style={"height": "400px"})
                for column in COLUMNS_CONSIDERED[4:8]
            ],  # Second row
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, 1fr)",
                "gap": "10px",
            },
        ),
        html.Div(
            [
                dcc.Graph(id=f"plot-{column}", style={"height": "400px"})
                for column in COLUMNS_CONSIDERED[8:12]
            ],  # Third row
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, 1fr)",
                "gap": "10px",
            },
        ),
        html.Div(
            [
                dcc.Graph(id=f"plot-{column}", style={"height": "400px"})
                for column in COLUMNS_CONSIDERED[12:]
            ],  # Fourth row
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, 1fr)",
                "gap": "10px",
            },
        ),
        dcc.Interval(
            id="interval-component", interval=2000, n_intervals=0  # in milliseconds
        ),
    ]
)


# Define callback to update the plots in real-time
@app.callback(
    [Output(f"plot-{column}", "figure") for column in COLUMNS_CONSIDERED],
    [Input("interval-component", "n_intervals")],
)
def update_graph(n_intervals):
    global i, std_scaler, model, samples_label
    i += 1

    # Ensure we don't exceed the bounds of df
    if i >= len(df):
        i = len(df) - 1

    # Create current_data based on the starting index and max window size
    start_index = max(0, i - MAX_WINDOW_SIZE + 1)  # Calculate start index
    current_data = df.iloc[start_index : i + 1]  # Slicing current_data based on indices
    if len(samples_label) == 0:
        samples_label = [0] * (len(current_data) - 1)
    else:
        samples_label = samples_label[-(len(current_data) - 1) :]

    assert len(current_data) - 1 == len(
        samples_label
    ), f"Length mismatch: current_data length is {len(current_data) - 1}, samples_label length is {len(samples_label)}"
    change_indices = current_data.index[
        current_data["State_ON"] != current_data["State_ON"].shift()
    ]

    # Separate into two lists
    indices_0_to_1 = change_indices[
        current_data.loc[change_indices, "State_ON"] == 1
    ].tolist()
    indices_1_to_0 = change_indices[
        current_data.loc[change_indices, "State_ON"] == 0
    ].tolist()

    inFlight = current_data.iloc[-1]["State_ON"] == 1
    if True:
        label = int(current_data.iloc[-1]['alert'])
    else:
        label = 0

    samples_label.append(label)

    # Create a list to store figures for each column
    figures = []

    # Loop through the columns and create a trace for each
    for column in COLUMNS_CONSIDERED:
        if column in current_data.columns:  # Check if the column exists
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=current_data.index.time,  # Use the time part of the index
                    y=current_data[column],
                    mode="lines",
                    name=column,
                )
            )

            # Update layout for individual plot
            fig.update_layout(
                title=column,
                xaxis_title="Time",
                yaxis_title="Values",
                legend_title="Metrics",
                template="plotly",
                xaxis=dict(
                    tickangle=-45, tickfont=dict(size=8)
                ),  # Smaller x-axis tick labels
                yaxis=dict(tickfont=dict(size=8)),  # Smaller y-axis tick labels
                height=400,  # Set the height of the plot
            )

            for idx in indices_0_to_1:
                fig.add_vline(
                    idx.time(),
                    line_width=2,
                    line_dash="dash",
                    line_color="green",
                    layer="below",
                    opacity=0.5,
                )
            for idx in indices_1_to_0:
                if current_data.index[0] != idx:
                    fig.add_vline(
                        idx.time(),
                        line_width=2,
                        line_dash="dash",
                        line_color="red",
                        layer="below",
                        opacity=0.5,
                    )

            if len(samples_label) != len(current_data):
                samples_label = samples_label[-len(current_data) :]
            assert len(current_data) == len(
                samples_label
            ), f"Length mismatch2: current_data length is {len(current_data)}, samples_label length is {len(samples_label)}"
            consecutive_marked_indices = find_consecutive_indices(samples_label)
            for start_idx, end_idx in consecutive_marked_indices:
                if end_idx < len(current_data):
                    highlight_x0 = current_data.index[start_idx].time()
                    highlight_x1 = current_data.index[end_idx].time()
                    highlight_y_min = current_data[column].min()
                    highlight_y_max = current_data[column].max()
                    fig.add_shape(
                        type="rect",
                        xref="x",
                        yref="y",
                        x0=highlight_x0,
                        y0=highlight_y_min,
                        x1=highlight_x1,
                        y1=highlight_y_max,
                        fillcolor="yellow",
                        opacity=0.5,
                        line_width=0,
                        layer="below",
                    )

            # Add annotation for the label
            fig.add_annotation(
                text=f"Label: {'Normal Operation' if label == 0 else 'Non Normal Operation'}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.1,
                showarrow=False,
                font=dict(size=16, color="black"),
                align="center",
                bgcolor="white",    
                opacity=0.8,
            )
            
            current_sample_index = i
            shap_val = shap_values[current_sample_index]    
            shap_df = pd.DataFrame(shap_val.values.reshape(1, -1), columns=data_model_columns)
                
            # Identify the most important feature
            most_important_feature = shap_df.abs().mean().idxmax()

            # Remove lag suffix if it exists
            if '_lag_' in most_important_feature:
                most_important_feature = most_important_feature.split('_lag_')[0]
                
            if most_important_feature in ['hour', 'second', 'minute']:
                # If it's one of the time features, find the next best feature
                second_most_important_feature = shap_df.abs().mean().nlargest(2).idxmin()
                feature_to_display = second_most_important_feature
            else:
                feature_to_display = most_important_feature
                
            # Add the feature importance annotation
            fig.add_annotation(
                text=f"Most Important Feature: {feature_to_display}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.15,
                showarrow=False,
                font=dict(size=12, color="black"),
                align="center",
                bgcolor="white",
                opacity=0.8,
            )

            figures.append(fig)

    return figures



# Run the application
if __name__ == "__main__":

    # ===========RUN SERVER==============
    app.run_server(debug=True, host="0.0.0.0")

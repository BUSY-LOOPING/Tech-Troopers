import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string, redirect
import pickle
from glob import glob
from drone_troopers.models.clustering_anomaly_rules_extractor import get_anomaly_n_rules
from drone_troopers.models.flight_classifier import predict
import tempfile

def get_resource_path(filename):
    """Get the absolute path of a resource file."""
    return os.path.join(os.path.dirname(__file__), filename)

app = Flask(__name__)


@app.route('/')
def index():
    return '''
    <!doctype html>
    <title>Upload File</title>
    <h1>Upload File</h1>
    <form action="/upload" method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        # Save the file temporarily to get the absolute path
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)

        # Get the prediction
        prediction = predict([file_path])
        print(prediction)

        label = "Normal Operation" if prediction[0] == 0 else "Non-Normal Operation"
        rules_html = ""

        if prediction[0] == 1:
            # Call the get_anomaly_n_rules function if prediction is 1
            anomaly_rules = get_anomaly_n_rules([file_path])
            for df in anomaly_rules:
                rules_html += df.to_html(classes='data', header="true", index=True)

        # Remove the temporary file after processing
        os.remove(file_path)

        return render_template_string(f'''
            <!doctype html>
            <title>Prediction Result</title>
            <h1>Prediction: {label}</h1>
            <h2>Anomaly Rules</h2>
            {rules_html}
            <a href="/">Upload another file</a>
        ''')

if __name__ == '__main__':
    app.run(debug=True)

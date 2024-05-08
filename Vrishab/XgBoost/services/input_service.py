#library to help read csv data and break it into the required format
import os
import csv

def read_data(file_name):
    function_name = 'read_data'
    try:
        print('got hit')
        current_dir = os.getcwd()
        file_path = os.path.abspath(os.path.join(current_dir, "..", "Data", file_name))
        print(file_path)

        columns = None
        data_rows = []

        with open(file_path, 'r') as file:
            # Create a CSV reader object
            csv_reader = csv.reader(file)
            columns = next(csv_reader) 
            
            for row in csv_reader:
                data_rows.append(row)
        
        return (columns, data_rows)
    except Exception as err:
        print('{fn}: err: {err}'.format(fn = function_name, err = err))
        raise err


def split_data(csv_data):
    function_name = 'split_data'
    try:
        print('work in progress')

    except Exception as err:
        print('{fn}: err: {err}'.format(fn = function_name, err = err))
        raise err

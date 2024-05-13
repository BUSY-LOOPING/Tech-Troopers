#library to help read csv data and break it into the required format
import os
import csv
import random

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


def split_data(csv_data, block_count = 10):
    function_name = 'split_data'
    try:
        len_data = len(csv_data)
        chunk_size = int(len_data/block_count)
        
        blocks = []
        i = 0
        
        for j in range(block_count - 1):
            k = i + chunk_size
        
            block = csv_data[i: k]
            blocks.append(block)

            i = k
        
        blocks.append(csv_data[i:])
        return blocks

    except Exception as err:
        print('{fn}: err: {err}'.format(fn = function_name, err = err))
        raise err

def create_train_test_split(csv_data_blocks, config):
    function_name = 'create_train_test_split'
    try:
        block_count = config['block_count']
        
        train_size = config['train_size']
        test_size = config['test_size']
        
        randomize = config['randomize']

        l = random.sample(range(block_count), block_count) if randomize else range(block_count)
        
        train_data = []
        for i in range(train_size):
            train_data.extend(csv_data_blocks[l[i]])
        
        test_data = []
        print(l)
        for i in range(test_size):
            test_data.extend(csv_data_blocks[l[block_count - 1 - i]])

        return (train_data, test_data)
    except Exception as err:
        print('{fn}: err: {err}'.format(fn = function_name, err = err))
        raise err


import os
import shutil


def create_path_from_params(*args):
    return os.path.join(*args)

def check_if_dir_exists(path):
    return os.path.isdir(path)

def list_files_in_dir(path):
    return os.listdir(path)

#function to drop and create folder
def drop_and_create_dir(path):
    if(os.path.isdir(path)):
        shutil.rmtree(path)
    os.mkdir(path)

#function to parse the list of file names in the folder
def read_and_parse_file_names_from_dir(path):
    file_dict = {}
    file_list = os.listdir(path)

    for file_name in file_list:
        name_list = file_name.split('.')
        if len(name_list) != 3:
            #not valid file name
            continue
        if name_list[2] != 'pkl':
            #not valid file extension
            continue
        
        file_name = name_list[0]
        index_number = int(name_list[1].split('_').pop())

        # print(name_list)

        # print(file_name)
        # print(index_number)

        if not file_dict.get(file_name):
            file_dict[file_name] = {
                'min': 0,
                'max': 0
            }
        max_ind = file_dict[file_name]['max']

        if index_number > max_ind:
            file_dict[file_name]['max'] = index_number
            continue

    return file_dict
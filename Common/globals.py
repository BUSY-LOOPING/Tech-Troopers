import os

cwd = os.getcwd()

learn_type_cat_dict = {
    'clf_cat': 0,
    'clf_num': 1,
    'reg_cat': 1
}

algo_type_cat_dict = {
    'gradient-boosting-results': {
        'isActive': 1,
        'name': 'gradient-boosting-results'
    },
    'random-forest-results': {
        'isActive': 1,
        'name': 'random-forest-results'
    },
    'xgboost-results': {
        'isActive': 1,
        'name': 'xgboost-results'
    }
}

results_folder_path = os.path.join(cwd, 'results')
results_folder_temp_path = os.path.join(cwd, 'results_temp')

results_file_mid_text = 'csv_shuffle_'
results_file_extension = 'pkl'
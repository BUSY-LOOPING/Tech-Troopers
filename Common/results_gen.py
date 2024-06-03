#import libs and dependencies
import globals
from helpers import file_helper
from workers import result_worker


if __name__ == '__main__':
    file_helper.drop_and_create_dir(globals.results_folder_path)
    file_helper.drop_and_create_dir(globals.results_folder_temp_path)

    for mode,v in globals.learn_type_cat_dict.items():
        print('processing for {k}'.format(k=mode))
        if not v:
            print('mode not enabled skipping...')
            continue

        mode_path = file_helper.create_path_from_params(globals.cwd, mode)
        if not file_helper.check_if_dir_exists(mode_path):
            print('mode folder not found skipping...')
            continue

        mode_results_path = file_helper.create_path_from_params(globals.results_folder_path, mode)
        mode_results_temp_path = file_helper.create_path_from_params(globals.results_folder_temp_path, mode)

        file_helper.drop_and_create_dir(mode_results_path)
        file_helper.drop_and_create_dir(mode_results_temp_path)

        for algo,v1 in globals.algo_type_cat_dict.items():
            print('processing {k1} algo'.format(k1=algo))
            if not globals.algo_type_cat_dict[algo].get('isActive', 0):
                print('algo is not active, skipping...')
                continue
            
            mod_algo_path = file_helper.create_path_from_params(mode_path, algo)
            file_dict = file_helper.read_and_parse_file_names_from_dir(mod_algo_path)

            file_helper.drop_and_create_dir(file_helper.create_path_from_params(mode_results_path, algo))
            file_helper.drop_and_create_dir(file_helper.create_path_from_params(mode_results_temp_path, algo))
            
            result_worker.result_worker(mode, algo, file_dict)
        print('tbd')
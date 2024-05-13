import config
from services.input_service import read_data, split_data,create_train_test_split
from services.xg_boost_service import xgBoostService

#main file
def main():
    function_name = 'main'
    try:
        #initialize config
        config.createConfig()
        conf = config.config
        
        conf_input = conf['input']
        (columns, data_rows) = read_data(conf_input['inputfilename'])

        print(columns)
        print(len(data_rows))

        blocks = split_data(data_rows, conf_input['numberofchunks'])
        print(len(blocks))

        split_conf = {
            'block_count': conf_input['numberofchunks'],
            'train_size': conf_input['trainchunksize'],
            'test_size': conf_input['testchunksize'],
            'randomize': conf_input['randomizechunks']
        }

        (train_data, test_data) = create_train_test_split(blocks, split_conf)

        xgboost_service = xgBoostService(train_data, test_data, columns)
        xgboost_service.build_pandas()
        xgboost_service.process()

        print('role 1 done')
    except Exception as err:
        print('{fn}: err: {err}'.format(fn = function_name, err = err))
        raise err

main()
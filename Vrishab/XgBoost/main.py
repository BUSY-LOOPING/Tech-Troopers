import config
from services.input_service import read_data

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
    except Exception as err:
        print('{fn}: err: {err}'.format(fn = function_name, err = err))
        raise err

main()
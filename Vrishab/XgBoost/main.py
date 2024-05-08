import config
import input_service

#main file
def main():
    function_name = 'main'
    try:
        #initialize config
        config.createConfig()
        conf = config.config

        

    except Exception as err:
        print('{fn}: err: {err}'.format(fn = function_name, err = err))
        raise err

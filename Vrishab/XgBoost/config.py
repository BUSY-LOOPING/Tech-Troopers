import configparser

from helpers.validation_helper import parseAndValidateStringToInt, parseAndValidateStringToIntRanges, parseAndValidateBooleanInt

config = None
def createConfig():
    function_name = 'createConfig'
    global config
    try:
        if config is not None:
            raise ValueError('Config value not none')

        conf = configparser.ConfigParser()
        conf.read('config.ini')
    
        config = {s:dict(conf.items(s)) for s in conf.sections()}

        #verify if needed sections are present
        if not isinstance(config.get('input'), dict):
           raise ValueError('input section missing')

        #verify input section
        config_input = config['input']
        print('{fn}: config input: {ci}'.format(fn = function_name, ci = config_input))
        if config_input.get('inputfilename') is None:
            raise ValueError('input file name missing')
        
        config_input['numberofchunks'] = parseAndValidateStringToIntRanges(config_input.get('numberofchunks'), (2, 50), 10)
        
        config_input['trainchunksize'] = parseAndValidateStringToInt(config_input.get('trainchunksize'), 1)
        config_input['testchunksize'] = parseAndValidateStringToInt(config_input.get('testchunksize'), 1)

        if config_input['numberofchunks'] < config_input['trainchunksize'] + config_input['testchunksize']:
            raise ValueError('chunks mismatch')

        config_input['randomizechunks'] = parseAndValidateBooleanInt(config_input.get('randomizechunks'), 0)
        config_input['writeparsedinputtofiles'] = parseAndValidateBooleanInt(config_input.get('writeparsedinputtofiles'), 0)

        #TODO: add validations when write to parsed input to files is true
    except Exception as err:
        print('{fn}: err: {err}'.format(fn = function_name, err = err))
        raise err

def clearConfig():
    function_name = 'clearConfig'
    global config
    try:
        if config is None:
            raise ValueError('Config value already none')
        
        config = None
    except Exception as err:
        print('{fn}: err: {err}'.format(fn = function_name, err = err))
        raise err


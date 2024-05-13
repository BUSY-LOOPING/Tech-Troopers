#helper function to convert string to int
def parseAndValidateStringToInt(val, defaultVal = None):
    function_name = 'parseAndValidateStringToInt'
    try:
        if val is None:
            if defaultVal is None:
                raise ValueError('value provided is none for a non noneable value')
            return defaultVal
        
        return int(val)
    except Exception as err:
        print('{fn}: err: {err}'.format(fn = function_name, err = err))
        raise err

def parseAndValidateStringToIntRanges(val, range, defaultVal):
    function_name = 'parseAndValidateStringToIntRanges'
    try:
        val_int = parseAndValidateStringToInt(val, defaultVal)
        (lbound, ubound) = range

        if not (val_int >= lbound and val_int <= ubound):
            raise ValueError('value out of range')
        
        return val_int
    except Exception as err:
        print('{fn}: err: {err}'.format(fn = function_name, err = err))
        raise err
    
def parseAndValidateBooleanInt(val, defaultVal):
    function_name = 'parseAndValidateBooleanInt'
    try:
        val_int = parseAndValidateStringToInt(val, defaultVal)
        if not (val_int == 0 or val_int == 1):
            raise TypeError('boolean value out of range')
        
        return val_int
    except Exception as err:
        print('{fn}: err: {err}'.format(fn = function_name, err = err))
        raise err
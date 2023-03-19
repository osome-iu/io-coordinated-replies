

def check_if_error_terminated(error_path):
    '''
    Checks if the job is terminated due to error
    :param error_path: path where error file is present
    
    :return Boolean
    '''
    error_string = file_hp.read_file(error_path)[-1]

    return 'Terminated' in error_string
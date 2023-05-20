from configparser import ConfigParser
import os
from pathlib import Path

def config(file = 'config.ini'):
    '''
    Reads the config file
    
    :param file: name of config file
    
    :return configparser
    '''
    abs_path = os.path.dirname(os.path.abspath(__file__))
    parent = Path(abs_path).parents[1]
    config_file = os.path.join(parent, file)
    configure = ConfigParser()
    configure.read(config_file)
    
    return configure
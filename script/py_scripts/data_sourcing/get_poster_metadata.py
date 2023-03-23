import pandas as pd
import numpy as np
import datetime
import warnings
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
from tqdm import tqdm
import sys
import os

import importlib

#### packages
import helper.strategy_helper as st
import helper.visualization as vz
import helper.helper as hp
import helper.file_helper as file_hp
import config.config as config
import helper.pandas_helper as pd_hp


def get_profile_info(profile_id_file, 
                     profile_json_file):
    '''
    Gets the user meta data
    :param profile_id_file: Twitter account userid file
    :param profile_json_file: File path with file name for
    json of profiles
    '''
    command = f'twarc2 users {profile_id_file} {profile_json_file}'

    os.system(command)  
        
    
if __name__ == "__main__":
    config = config.config()
    path = config['PATHS']

    poster_path = config['POSTER_PATH']
    poster_id_path = poster_path['poster_id_path']
    poster_json_path = poster_path['poster_json_path']

    get_profile_info(poster_id_path, poster_json_path)
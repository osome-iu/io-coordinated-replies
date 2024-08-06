#### **This script visualizes pairwise cosine**

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
import re

import importlib

#### packages
import helper.strategy_helper as st_hp
import helper.visualization as vz_hp
import config.config as config_hp

#### Load files
config = config_hp.config()

pos_cosine_path = config['EMBEDDINGS_PATH']['pos_cosine_with_replier_info']

df_pos_cosine = pd.read_pickle(pos_cosine_path)


#### Visualize
df_pos_cosine_1 = df_pos_cosine.loc[(df_pos_cosine['replier_label_x'] == 1) & (df_pos_cosine['replier_label_y'] == 1)]
df_pos_cosine_0 = df_pos_cosine.loc[(df_pos_cosine['replier_label_x'] == 0) & (df_pos_cosine['replier_label_y'] == 0)]
# df_pos_cosine_01 = df_pos_cosine.loc[(df_pos_cosine['replier_label_x'] == 0) & (df_pos_cosine['replier_label_y'] == 1)]

### For control and IO 
# neg_cosine_path = config['EMBEDDINGS_PATH']['neg_cosine_with_replier_info']
# df_neg_cosine = pd.read_pickle(neg_cosine_path)

print('*** Viz start ***')

fig_path = '/N/slate/potem/project/infoOps-strategy/script/notebooks/reply_coordination/cosine_viz'

parameters =   {
        'data': [df_pos_cosine_1, df_pos_cosine_0],
    
        'figsize': (10, 10),
        'fontsize': 36,
        'tick_size': 28,
        'legend_font': 26,
        'complementary': True,
    
        'columns': [
            {'column': 'cosine',
             'label': 'IO Replier',
            },
            {
            'column': 'cosine',
             'label': 'Normal Replier'
            },
            # {
            # 'column': 'cosine',
            #  'label': 'IO-Normal Replier',
            # 'color': 'black'
            # }
        ],
        'xlabel': 'Cosine Similarity',
        'ylabel': 'CCDF',
        'legend_location': 'upper right',
        'log_yscale': True,
        'log_xscale': False,
        'save': {
            'path': f'{fig_path}',
            'filename': 'exp_log_tweet_CCDF_IO_normal_cosine_new.png'
        },
        'random_color': False
    }

vz_hp.ccdf(parameters)



# parameters =   {
#         'data': [df_pos_cosine_1, df_pos_cosine_0, df_pos_cosine_01],
#         'fontsize': 20,
#         'complementary': True,
#         'columns': [
#             {'column': 'cosine',
#              'label': 'IO Replier',
#             },
#             {
#             'column': 'cosine',
#              'label': 'Normal Replier'
#             },
#             {
#             'column': 'cosine',
#              'label': 'IO-Normal Replier',
#             'color': 'black'
#             }
#         ],
#         'xlabel': 'Cosine Similarity',
#         'ylabel': 'Density',
#         'legend_location': 'upper right',
#         'log_yscale': False,
#         'log_xscale': False,
#         'save': {
#             'path': './plots',
#             'filename': 'exp_tweet_kde_IO_normal_cosine.png'
#         },
#         'random_color': False
#     }

# vz_hp.kde(parameters)

# print('*** Viz End ***')

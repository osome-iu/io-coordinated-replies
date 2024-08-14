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
import gc

import importlib

#### packages
import helper.strategy_helper as st_hp
import helper.visualization as vz_hp
import config.config as config_hp


config = config_hp.config()
pos_cosine_path = config['EMBEDDINGS_PATH']['pos_cosine_with_replier_info']
neg_cosine_path = config['EMBEDDINGS_PATH']['neg_cosine_with_replier_info']

df_pos_cosine = pd.read_pickle(pos_cosine_path)
df_pos_cosine_0 = df_pos_cosine.loc[
    (df_pos_cosine['replier_label_x'] == 0) & (df_pos_cosine['replier_label_y'] == 0)
]

del df_pos_cosine
gc.collect()

print('Finish loading cosine!')

config = config_hp.config()
stat = config['USER_FEATURES']
all_feature = stat['all_feature']
df_stat = pd.read_pickle(all_feature)

df_0 = df_stat.loc[df_stat['replier_label'] == 0]
df_1 = df_stat.loc[df_stat['replier_label'] == 1]

del df_stat
gc.collect()

print('Finish loading stat!')

all_df = [df_pos_cosine_0]
for i in range(0,10):
    df_sample = df_0.sample(len(df_1), random_state=i)
    
    df_0 = df_0.loc[~df_0['replier_userid'].isin(
        df_sample['replier_userid']
    )]
    
    df_all = pd.concat([df_sample,
                        df_1
                       ])

    df_sample_cosine = df_pos_cosine_0.loc[
        df_pos_cosine_0['replier_userid_x'].isin(
            df_all['replier_userid']
        ) | df_pos_cosine_0['replier_userid_y'].isin(
            df_all['replier_userid']
        )
    ]
    all_df.append(df_sample_cosine)

    del df_all
    del df_sample

    gc.collect()

print('Finish loading samples!')

def plot_histogram(parameters):
    '''
    Plots histogram
    :param parameters: parameters for the plot
    '''
       
    # parameters =  {
    #     'data': df,
    #     'fontsize': 14,
    #     'columns': [
    #         {'column': 'count',
    #          'label': 'Count of Replies',
    #         }
    #     ],
    #     'xlabel': '',
    #     'ylabel': '',
    #     'legend_location': '',
    #     'log_yscale': True,
    #     'log_xscale': True,
    #     'bins': 60,
    #     'save': {
    #         'path': '',
    #         'filename': ''
    #     },
    #     'title': ''
    # }
    
    
    num_bins = parameters['bins']
    column = parameters['columns'][0]['column']
    # df = parameters['data']
    
    if 'fontsize' in parameters:
        fontsize = parameters['fontsize']
    else:
        fontsize = 14
    
    if parameters['bins'] == None:
        num_bins = df[column].nunique()
        
    if parameters['log_xscale'] == True:
        num_bins=np.logspace(start=np.log10(min(df[column])), 
                             stop=np.log10(max(df[column])),
                             num=num_bins
                            )
    fig, ax = plt.subplots(figsize=(10,10))

    # colors = ['red', 'blue', 'green', 
    #           'orange', 'olive', 'pink', 'lime',
    #           'maroon'
    #          ]
    colors = ['blue', 
              'lightsteelblue', 
              'lightsteelblue', 
              'lightsteelblue', 
              'lightsteelblue', 
              'lightsteelblue', 
              'lightsteelblue', 
              'lightsteelblue', 
              'lightsteelblue'
              'lightsteelblue', 
              'lightsteelblue'
             ]
    for i, df in enumerate(parameters['data']):
        if i == 0:
            alpha = 0.3
        else:
            alpha = 0.1
            
        n = ax.hist(df[column],
                    bins=num_bins, 
                    color=colors[i],
                    alpha=alpha,
                    density=True,
                    # label=parameters['columns'][i]['label']
                   )
        # n = ax.plot(df[column],
        #             # bins=num_bins, 
        #             color=colors[i],
        #             alpha=0.3,
        #             # density=True,
        #             label=parameters['columns'][i]['label']
        #            )
    ax.set_xlabel(parameters['xlabel'],
                  fontsize=fontsize
                 )
    ax.set_ylabel(parameters['ylabel'],
                  fontsize=fontsize
                 )
    if 'tick_size' in parameters:
        tick_size = parameters['tick_size']
    else:
        tick_size = fontsize
        
        
    ax.tick_params(axis='both', 
                   which='both', 
                   labelsize=tick_size,
                   labelbottom=True
                  )

    if 'legend_location' in parameters:
        if 'legend_font' in parameters:
            legend_font = parameters['legend_font']
        else:
            legend_font = fontsize
            
    ax.legend(loc=parameters['legend_location'], 
              frameon=True, 
              fontsize=legend_font
             )
    
    if parameters['log_yscale'] == True:
        plt.yscale('log')
    if parameters['log_xscale'] == True:
        plt.xscale('log')
        
    plt.title(parameters['title'],
              fontsize=fontsize
             )
    fig.tight_layout()
    
    if 'save' in parameters:
        plot_path = parameters['save']['path']
        title = parameters['save']['filename']

        path = os.path.join(plot_path, title)
        fig.savefig(f'{path}', 
                  facecolor='white', 
                  transparent=False)
    plt.show()



import helper.visualization as vz_hp

bins = list(np.arange(0, 1.01, 0.1))

columns = []
columns= [{'column': 'cosine',
            'label': 'Normal repliers',
            }]
for i in range(1, 11):
    columns.append(
        {'column': 'cosine',
         'label': f'',
        }
    )
                   
parameters =  {
        'data': all_df, #[df_pos_cosine_1,df_pos_cosine_0] ,
        'fontsize': 36,
        'legend_font': 24,
        'columns':columns,
        'tick_size': 26,
        'xlabel': 'Cosine similarity',
        'ylabel': 'Density',
        'legend_location': 'upper left',
        'log_yscale': False,
        'log_xscale': False,
        'bins': bins,
        'save': {
            'path': '/N/slate/potem/project/infoOps-strategy/script/notebooks/reply_coordination/cosine_viz/plots',
            'filename': 'sample_histogram_io_normal_replier.png'
        },
        'title': ''
    }

plot_histogram(parameters)

print('Finish plotting samples!')
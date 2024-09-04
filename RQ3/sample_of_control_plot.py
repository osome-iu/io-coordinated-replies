import pandas as pd
import numpy as np

import importlib

#### packages
import helper.strategy_helper as st_hp
import helper.visualization as vz_hp
import config.config as config_hp

#Data path
pos_cosine_path = './../data/pos_cosine_with_replier_info.pkl.gz'

df_pos_cosine = pd.read_pickle(pos_cosine_path)
df_pos_cosine_0 = df_pos_cosine.loc[(df_pos_cosine['replier_label_x'] == 0)]

import gc

del df_pos_cosine
gc.collect()

#Sample the normal
all_feature = './../data/replier_classifier_features.pkl.gz'

df_stat = pd.read_pickle(all_feature)
df_0 = df_stat.loc[df_stat['replier_label'] == 0]
df_1 = df_stat.loc[df_stat['replier_label'] == 1]

import gc

del df_stat
gc.collect()


#samples
all_df = []
for i in range(0,10):
    #get the random sample from control
    df_sample = df_0.sample(len(df_1), random_state=i)

    #prepare data for next sample without replacement
    df_0 = df_0.loc[~df_0['replier_userid'].isin(
        df_sample['replier_userid']
    )]

    df_sample_cosine = df_pos_cosine_0.loc[
        df_pos_cosine_0['replier_userid_x'].isin(
            df_sample['replier_userid']
        )
    ]
    all_df.append(df_sample_cosine)
    
    del df_all
    del df_sample

    gc.collect()


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

    colors = ['red', 'blue', 'green', 
              'orange', 'olive', 'pink', 'lime',
              'maroon'
             ]
    colors = ['blue', 
              'pink', 
              'pink', 
              'pink', 
              'pink', 
              'pink', 
              'pink', 
              'pink', 
              'pink'
              'pink', 
              'pink',
              'pink'
             ]
    for i, df in enumerate(parameters['data']):
        if i == 0:
            color = 'blue'
            alpha = 0.5
            histtype='bar'
            linewidth =4
        else:
            color = None
            alpha = 0.5
            histtype='step'
            linewidth=2
            style = {'facecolor': 'none', 'edgecolor': 'C0', 'linewidth': 3}
            
        n = ax.hist(df[column],
                    bins=num_bins, 
                    color=color,
                    alpha=alpha,
                    density=True,
                    histtype=histtype,
                    linewidth=linewidth
                    # label=parameters['columns'][i]['label']
                   )

    ax.set_ylim(0, 2.74)
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



#Plot
import helper.visualization as vz_hp

bins = list(np.arange(0, 1.01, 0.1))

columns = []
columns= [{'column': 'cosine',
             'label': 'Normal repliers',
            }]
for i in range(1, 11):
    columns.append(
        {'column': 'cosine',
         'label': f'Sample {i}',
        }
    )
                   
parameters =  {
        'data': all_df,
        'fontsize': 36,
        'legend_font': 24,
        'columns':columns,
        'tick_size': 26,
        'xlabel': 'Cosine similarity',
        'ylabel': 'Density',
        'log_yscale': False,
        'log_xscale': False,
        'bins': bins,
        'save': {
            'path': './../plots',
            'filename': 'sample_histogram_io_normal_replier_any.png'
        },
        'title': ''
    }

plot_histogram(parameters)





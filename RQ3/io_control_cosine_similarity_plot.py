import pandas as pd
import numpy as np

import importlib

#### packages
import helper.strategy_helper as st_hp
import helper.visualization as vz_hp

pos_cosine_path = './../data/pos_cosine_with_replier_info.pkl.gz'

df_pos_cosine = pd.read_pickle(pos_cosine_path)

print('Data loaded!')

df_pos_cosine_0 = df_pos_cosine.loc[(df_pos_cosine['replier_label_x'] == 0)]
df_pos_cosine_1 = df_pos_cosine.loc[(df_pos_cosine['replier_label_x'] == 1)]

print('Data Separated!')

import gc
del df_pos_cosine
gc.collect()


def plot_single_histogram(parameters):
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
    
    import numpy as np
    
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
    for i, df in enumerate(parameters['data']):
        counts, bin_edges, patches  = ax.hist(df[column],
                    bins=num_bins, 
                    color=colors[i],
                    alpha=0.5,
                    density=True,
                    label=parameters['columns'][i]['label']
                   )
        max_density = np.max(counts)
        
        print(max_density)

    y_min, y_max = ax.get_ylim()
    
    # Print the y-axis range
    print(f"Y-axis range: min={y_min}, max={y_max}")

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



bins = list(np.arange(0, 1.01, 0.1))
parameters =  {
        'data': [df_pos_cosine_1,df_pos_cosine_0] ,
        'fontsize': 36,
        'legend_font': 24,
        'columns': [
            {'column': 'cosine',
             'label': 'IO repliers',
            },
            {'column': 'cosine',
             'label': 'Normal repliers',
            }
           ],
        'tick_size': 26,
        'xlabel': 'Cosine similarity',
        'ylabel': 'Density',
        'log_yscale': False,
        'log_xscale': False,
        'bins': bins,
        'save': {
            'path': './../plots',
            'filename': 'io_normal_to_any_replier.png'
        },
        'title': ''
    }

print('Start plotting!')
plot_single_histogram(parameters)

print('End plotting!')

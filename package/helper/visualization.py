import pandas as pd
import numpy as np
import warnings
import datetime
import gzip
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
from tqdm import tqdm
from random import sample
import itertools
import difflib
import json
import sys
import os
import math
import matplotlib.ticker as tck

def ccdf(parameters):
    '''
    Plots ccdf for data
    
    :param parameters: parameters to set for the plot
    '''
    
    # {
    #     'data': df,
    #     'fontsize': 14,
    #     'complementary': True,
    #     'columns': [
    #         {'column': ''
    #          'label': '',
    #         },{
    #         'column': '',
    #          'label': ''
    #         }
    #     ],
    #     'xlabel': '',
    #     'ylabel': '',
    #     'legend_location': '',
    #     'log_yscale': True,
    #     'log_xscale': True,
    #     'save': {
    #         'path': '',
    #         'filename': ''
    #     },
        # 'random_color': False
    # }
    
    keys = parameters.keys()
    if 'figsize' in keys:
        size = parameters['figsize']
    else:
        size = (8,8)
        
    fig, ax = plt.subplots(figsize=size)
    # fig = plt.figure(figsize=size)

    # Add an axes at position [left, bottom, width, height]
    # where each value is between 0 and 1

    # ax = fig.add_axes([0.2, 0.2, 0.9, 0.9])
    
    fontsize = parameters['fontsize']
    colors = ['red', 'blue', 'green', 'orange', 'olive', 'pink', 'lime', 'maroon']
    total_columns = len(parameters['columns'])
    
    if parameters['random_color'] == True:
        all_colors =  [k for k,v in pltc.cnames.items()]
        colors = sample(all_colors, total_columns)
    
    symbols = ['.', 'o', '+', 'x', '*', 'v', '^', '>']
    
    i = 0
    cmap = plt.cm.get_cmap('hsv', total_columns)
    max_n = 0
    for data in parameters['data']:
        column = parameters['columns'][i]['column']
        data = parameters['data'][i][column]
        label = parameters['columns'][i]['label']
        
        if 'color' in parameters['columns'][i].keys():
            assigned_color = parameters['columns'][i]['color']
        else:
            assigned_color = colors[i]
        
        if max_n < max(data):
            max_n = max(data)
            
        sns.ecdfplot(data, 
                     complementary=parameters['complementary'],
                     label=label,
                     # marker=symbols[i],
                     color=assigned_color,
                     ax=ax,
                     linewidth=2,)

        i = i + 1
        
    if 'log_yscale' in keys and parameters['log_yscale'] == True:
        ax.set_yscale('log')
        # ax.yaxis.set_minor_locator(AutoMinorLocator())
       
        
    if 'log_xscale' in keys and parameters['log_xscale'] == True:
        ax.set_xscale('log')
        
        n = int(math.log10(max_n) + 1)
        all_ticks = []
        for x in range(0, int(n)):
            for i in range(1, 10):
                all_ticks.append(i * (10**x))
        
        ax.xaxis.set_minor_locator(tck.FixedLocator(all_ticks))


    if parameters['complementary'] == True:
        parameters['ylabel'] = 'CCDF'
        
    ax.set_xlabel(parameters['xlabel'], 
                  fontsize=fontsize)
    ax.set_ylabel(parameters['ylabel'], 
                  fontsize=fontsize)
    
    if 'tick_size' in keys:
        tick_size = parameters['tick_size']
    else:
        tick_size = fontsize
        
        
    ax.tick_params(axis='both', 
                   which='both', 
                   labelsize=tick_size,
                   labelbottom=True
                  )

    # ax.xaxis.set_minor_locator(tck.AutoMinorLocator())

    if 'legend_location' in keys:
        if 'legend_font' in keys:
            legend_font = parameters['legend_font']
        else:
            legend_font = fontsize
            
        ax.legend(loc=parameters['legend_location'], 
                  frameon=True, 
                  fontsize=legend_font
                 )
        
    if 'legend_lower' in keys:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', 
                  bbox_to_anchor=(1, -0.06),
                  fancybox=True, 
                  shadow=True, ncol=3)
    
    
        # ax.xaxis.set_minor_locator(AutoMinorLocator())

    if 'title' in keys:
        plt.title(parameters['title'])
        
        
    if 'figure_text' in keys:
        plt.text(parameters['figure_text_x'], 
                 parameters['figure_text_y'], 
                 parameters['figure_text'], 
                 fontsize=parameters['figure_font'],
                 ha="center", 
                 va="center", 
                )
        
    if 'subplot_adjust' in keys:
        plt.subplots_adjust(
            bottom=parameters['subplot_adjust']
        )
   
    fig.tight_layout()
    
    if 'save' in keys:
        path = parameters['save']['path']
        filename = parameters['save']['filename']
        fig_path = os.path.join(path, filename)
        
        print(fig_path)
        
        fig.savefig(fig_path, 
              facecolor='white', 
              transparent=False)
        
    plt.show()
    
    
    
    
def bar_single_plot(parameters):
    keys = parameters.keys()
    
    if 'size' in keys:
        size = parameters['size']
    else:
        size = (8, 8)
        
    
    fig, ax = plt.subplots(figsize=size, 
                           sharey=parameters['sharey'], 
                           sharex=parameters['sharex'])
        
    fontsize = parameters['fontsize']
    colors = ['red', 'blue', 'orange', 'red', 'olive', 
              'pink', 'lime', 'maroon']
    
    data = parameters['data']
    x_index = np.arange(len(data))
    x = parameters['x']
    # data[x] = data[x].astype(str)
    
    for i, columns in enumerate(parameters['columns']):
        temp_column = parameters['columns'][i]['column']
        ax.bar(x_index, data[temp_column], 
                 # align='center', 
                 alpha=0.5,
                 label=parameters['columns'][i]['label'],
                 color=colors[i],
                 # ax=ax,
                 # linewidth=2
                )
        
        if 'mean' in parameters['columns'][i]:
            if (parameters['columns'][i]['mean'] == True):
                ax.axhline(round(data[temp_column].mean(), 2),
                           color=colors[i],
                           linewidth=2
                          )
                ax.axhline(round(data[temp_column].median(), 2),
                           color=colors[i],
                           marker='+',
                           linewidth=5
                          )
        
    ax.set_ylabel(parameters['ylabel'], fontsize=fontsize)
    ax.set_xlabel(parameters['xlabel'], fontsize=fontsize)

    ax.set_xticks(x_index, data[x])

    # ax.set_xlabel('Time (Year-month)', fontsize=fontsize)
    ax.tick_params(axis='both', which='both', 
                   labelsize=10, labelbottom=True)
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.yaxis.set_tick_params(labelbottom=True)

    if parameters['log_yscale'] == True:
        ax.set_yscale('log')
    if parameters['log_xscale'] == True:
        ax.set_xscale('log')

    ax.legend(loc=parameters['legend_location'], frameon=True, 
              fontsize=fontsize)
    title = parameters['title']
    ax.set_title(f'{title}')

    ax.tick_params(axis='x', labelrotation=90)
    
    # plt.xticks(rotation = 90)
    # plt.title(title)
        
    if parameters['show'] == False:
        plt.close()
    else:
        plt.show()
        
    fig.tight_layout()

    if 'save' in parameters:
        save_path = parameters['save']['path']
        filename = parameters['save']['filename']

        fig.savefig(f'{save_path}/{filename}.png', 
              facecolor='white', 
              transparent=False)


    
def boxplot(parameters):
    if 'size' in parameters:
        size = parameters['size']
    else:
        size=(8,8)
        
    keys = parameters.keys()
    fig, ax = plt.subplots(figsize=size)
    fontsize = parameters['fontsize']
    data = parameters['data']
    i = 0
    columns = []
    ticks = []
    labels = []
    
    for df in data:
        column = parameters['columns'][i]['column']
        label = parameters['columns'][i]['label']
        
        columns.append(df[column])
        labels.append(label)
        
        i = i + 1
        ticks.append(i)
        
    ax.boxplot(columns)
    
    if 'yscale' in keys and parameters['yscale'] == True:
        ax.set_yscale('log')
        
    ax.tick_params(axis='x', which='major', labelsize=fontsize)

    ax.set_xlabel(parameters['xlabel'], 
                  fontsize=fontsize)
    ax.set_ylabel(parameters['ylabel'], 
                  fontsize=fontsize)
    
    if 'title' in keys:
        title = parameters['title']
        ax.set_title(f'{title}')
    
    plt.xticks(ticks, labels)
    
    fig.tight_layout()

    if 'save' in keys:
        path = parameters['save']['path']
        filename = parameters['save']['filename']
        
        fig.savefig(f'{path}/{filename}', 
              facecolor='white', 
              transparent=False)
        
    plt.show()
    
    
    
def multiple_bar_graph(parameters):
    x = np.arange(len(parameters['labels']))  
    keys = parameters.keys()
    width = 0.32
    
    if 'size' in keys:
        figsize=parameters['size']
    else:
        figsize=(8,8)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    rects1 = ax.bar(x - width/2, parameters['x'][0], width, 
                    label=parameters['label_name'][0], color='red',
                    alpha=0.5
                   )
    rects2 = ax.bar(x + width/2, parameters['x'][1], width, 
                    label=parameters['label_name'][1], color='blue',
                    alpha=0.5
                   )

    ax.set_ylabel(parameters['ylabel'], fontsize=parameters['fontsize'])
    ax.set_xlabel(parameters['xlabel'], fontsize=parameters['fontsize'])
    ax.set_title(parameters['title'], fontsize=parameters['fontsize'])
    ax.set_xticks(x, parameters['labels'], fontsize=parameters['fontsize'])
    ax.tick_params(axis='x', labelrotation=90)
    
    
    if ('reports' in keys) and (len(parameters['reports']) != 0):
        for i, text in enumerate(parameters['reports']):
            x_ticks = ax.get_xticklabels()
            
            for i, x_t in enumerate(x_ticks):
                if x_t.get_text() == text:
                    ax.get_xticklabels()[i].set_color("purple")
                    
                    
    if ('diff_color' in keys) and (len(parameters['diff_color']) != 0):
        for i, text in enumerate(parameters['diff_color']):
            x_ticks = ax.get_xticklabels()
            
            for i, x_t in enumerate(x_ticks):
                if x_t.get_text() == text:
                    ax.get_xticklabels()[i].set_color("red")
                
    
    ax.legend(loc="upper right", frameon=True, fontsize=parameters['fontsize'])
    
    if 'value' in keys:
        value = parameters['value']
        
        ax.bar_label(value[0], padding=3)
        ax.bar_label(value[1], padding=3)
    else:
        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    
    if 'save' in keys:
        path = parameters['save']['path']
        filename = parameters['save']['filename']
        
        fig.savefig(f'{path}/{filename}', 
              facecolor='white', 
              transparent=False)
        
    plt.show()
    
    
def ccdf_symbol(parameters):
    '''
    Plots ccdf for data with symbols
    
    :param parameters: parameters to set for the plot
    '''
    
    # {
    #     'data': df,
    #     'fontsize': 14,
    #     'complementary': True,
    #     'columns': [
    #         {'column': ''
    #          'label': '',
    #         },{
    #         'column': '',
    #          'label': ''
    #         }
    #     ],
    #     'xlabel': '',
    #     'ylabel': '',
    #     'legend_location': '',
    #     'log_yscale': True,
    #     'log_xscale': True,
    #     'save': {
    #         'path': '',
    #         'filename': ''
    #     },
        # 'random_color': False
    # }
    
    keys = parameters.keys()
    fig, ax = plt.subplots(figsize=(8, 8))
    fontsize = parameters['fontsize']
    colors = ['red', 'blue', 'green', 'orange', 'olive', 'pink', 'lime', 'maroon']
    total_columns = len(parameters['columns'])
    
    if parameters['random_color'] == True:
        all_colors =  [k for k,v in pltc.cnames.items()]
        colors = get_cmap(total_columns)
        # colors = sample(all_colors, total_columns)
        
    
    symbols = ['.', 'o', '+', 'x', '*', 'v', '^', '>']
    
    marker = itertools.cycle(symbols) 
    
    # for n in y:
    #     plt.plot(x,n, marker = marker.next(), linestyle='')

    
    i = 0
    cmap = plt.cm.get_cmap('hsv', total_columns)
    
    for data in parameters['data']:
        column = parameters['columns'][i]['column']
        data = parameters['data'][i][column]
        label = parameters['columns'][i]['label']
        
        if i == 15:
            sns.ecdfplot(data, 
                     complementary=parameters['complementary'],
                     label=label,
                     marker=symbols[3],
                     color='red',
                     ax=ax,
                     # linewidth=1,
                        )
        elif i == 11:
            sns.ecdfplot(data, 
                     complementary=parameters['complementary'],
                     label=label,
                     marker=symbols[1],
                     color='blue',
                     ax=ax,
                     # linewidth=1,
                        )
        else:
            sns.ecdfplot(data, 
                         complementary=parameters['complementary'],
                         label=label,
                         marker=next(marker),
                         color=colors(i),
                         ax=ax,
                         # linewidth=1,
                        )

        i = i + 1
        
    ax.set_xlabel(parameters['xlabel'], 
                  fontsize=fontsize)
    ax.set_ylabel(parameters['ylabel'], 
                  fontsize=fontsize)

    ax.tick_params(axis='both', labelsize=fontsize) 
    
    if 'legend_location' in keys:
        ax.legend(loc=parameters['legend_location'], 
                  frameon=True, fontsize=fontsize)
        
    if 'legend_lower' in keys:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.2,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', 
                  bbox_to_anchor=(0.5, -0.1),
                  fancybox=True, 
                  shadow=True, ncol=3)
    
    if 'log_yscale' in keys:
        ax.set_yscale('log')
    if 'log_xscale' in keys:
        ax.set_xscale('log')
        
    if 'title' in keys:
        plt.title(parameters['title'])

    if 'save' in keys:
        path = parameters['save']['path']
        filename = parameters['save']['filename']
        
        fig.savefig(f'{path}/{filename}', 
              facecolor='white', 
              transparent=False)
        
    plt.show()
    

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)



# timeseries_parameters = {
#         'data': replies_per_month,
#         'size': (8, 8),
#         'fontsize': 14,
#         'title': iran_202012_campaign,
#         'sharex': True,
#         'sharey': True,
#         'columns': [
#             {
#                 'x': 'tweet_time_month',
#                 'y': 'ratio_reply',
#                 'label': 'Information Operation',
#             },
#             {
#                 'x': 'tweet_time_month',
#                 'y': 'ratio_reply',
#                 'label': 'Control',
#             },
#         ],
#         'x': 'Percentage of replies to external account \n to total tweets per month',
#         'two_column': False,
#         'ylabel': 'Ratio of replies to external account \n to total tweets per month',
#         'xlabel': 'Time (Year-month)',
#         'legend_location': 'upper right',
#         'random_color': False,
#         'save': {
#             'path': f'{plot_path}',
#             'filename': f'{iran_202012_campaign}_replies_to_tweet_percentage_timeseries.png'
#         }
#     }

def time_series_plot(parameters):
    '''
    Creates time series plot
    
    :param parameters: dictionary of parameters to be used in plots
    '''
    
    # timeseries_parameters = {
    #     'data': replies_per_month,
    #     'size': (8, 8),
    #     'fontsize': 14,
    #     'title': iran_202012_campaign,
    #     'sharex': True,
    #     'sharey': True,
    #     'columns': [
    #         {
    #             'x': 'tweet_time_month',
    #             'y': 'ratio_reply',
    #             'label': 'Information Operation',
    #         },
    #         {
    #             'x': 'tweet_time_month',
    #             'y': 'ratio_reply',
    #             'label': 'Control',
    #         },
    #     ],
    #     'x': 'Percentage of replies \n to total tweets per month',
    #     'two_column': False,
    #     'ylabel': 'Ratio of replies \n to total tweets per month',
    #     'xlabel': 'Time (Year-month)',
    #     'legend_location': 'upper right',
    #     'random_color': False,
    #     'save': {
    #         'path': f'{plot_path}',
    #         'filename': f'*.png'
    #     }
    # }
    
    if 'size' in parameters:
        size = parameters['size']
    else:
        size = (10, 10)
        
    data = parameters['data']
    
    fig, ax = plt.subplots(figsize=size)
        
    fontsize = 14
    colors = ['red', 'blue', 'orange', 'red', 'olive', 
              'pink', 'lime', 'maroon']
    x_index = np.arange(len(data))
    
    x_label = parameters['xlabel']
    y_label = parameters['ylabel']
    title = parameters['title']
    
    if 'save' in parameters:
        plot_path = parameters['save']['path']
        filename = parameters['save']['filename']
    
    for index, df in enumerate(parameters['data']):
        x = parameters['columns'][index]['x']
        y = parameters['columns'][index]['y']
        label = parameters['columns'][index]['label']
        
        ax.plot(df[x], 
                df[y],
                color=colors[index],
                label=label,
                linewidth=2,
               )
        
       
        # ax.axhline(round(data[key].mean(), 2),
        #            color=colors[i],
        #            linewidth=2
        #           )
        # ax.axhline(round(data[key].median(), 2),
        #            color=colors[i],
        #            marker='+',
        #            linewidth=5
        #           )
        
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
        
        # new_axis.set_xticks(x_index, data[x])
        # new_axis.set_xlabel('Time (Year-month)', 
        # fontsize=fontsize)
        
    ax.tick_params(axis='both', which='both', 
                       labelsize=8, labelbottom=True)
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.yaxis.set_tick_params(labelbottom=True)

    frameon = False
    if 'frameon' in parameters:
        frameon = parameters['frameon']
        
    ax.legend(loc="upper right", 
              frameon=frameon, 
              fontsize=fontsize)
    ax.set_title(f'{title}')
    ax.tick_params(axis='x', labelrotation=90)
    
    plt.xticks(rotation = 90)
    plt.title(title)
        
    plt.show()
    
    if 'save' in parameters:
        fig.savefig(f'{plot_path}/{filename}.png', 
              facecolor='white', 
              transparent=False)


def multiple_time_series_plot(parameters):
    '''
    Plots the time series of multiple data
    :param parameters: parameters for visualization
    '''
    
#     columns = []
#     for year in campaigns:
#         for new_campaign in campaigns[year]:
#             columns.append({
#                 'x': 'counter',
#                 'label': new_campaign,
#                 'y': 'ratio_reply'
#             })


#     timeseries_parameters = {
#             'data': replies[10:14],
#             'size': (8, 8),
#             'fontsize': 14,
#             'title': iran_202012_campaign,
#             'sharex': True,
#             'sharey': True,
#             'columns': columns[10:14],
#             'x': 'Percentage of replies \n to total tweets per month',
#             'two_column': False,
#             'ylabel': 'Ratio of replies \n to tweets per month',
#             'xlabel': 'Time (Year-month)',
#             'legend_location': 'upper right',
#             'random_color': False,
#             'save': {
#                 'path': f'{plot_path}',
#                 'filename': f'all_replies_to_tweet_timeseries.png'
#             }
#         }
    
    if 'size' in parameters:
        size = parameters['size']
    else:
        size = (10, 10)
        
    fig, ax = plt.subplots(figsize=size)
        
    fontsize = 14
    colors = ['red', 'blue', 'orange', 'red', 'olive', 
              'pink', 'lime', 'maroon']
    x_index = np.arange(len(data))
    
    x_label = parameters['xlabel']
    y_label = parameters['ylabel']
    title = parameters['title']
    plot_path = parameters['save']['path']
    filename = parameters['save']['filename']
    
    total_columns = len(parameters['columns'])
    all_colors =  [k for k,v in pltc.cnames.items()]
    colors = get_cmap(total_columns)
    symbols = ['.', 'o', '+', 'x', '*', 'v', '^', '>']
    marker = itertools.cycle(symbols) 
    colors = ['red', 'blue', 'green', 'orange', 'olive', 'pink', 'lime', 'maroon']

    for index, df in enumerate(parameters['data']):
        x = parameters['columns'][index]['x']
        y = parameters['columns'][index]['y']
        label = parameters['columns'][index]['label']
        
        ax.plot(df[x], 
                df[y],
                color=colors[index],
                marker=next(marker),
                label=label,
                linewidth=2,
               )
        
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.tick_params(axis='both', which='both', 
                       labelsize=8, labelbottom=True)
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.yaxis.set_tick_params(labelbottom=True)

    ax.legend(loc="upper right", 
              frameon=True, 
              fontsize=fontsize)
    ax.set_title(f'{title}')
    ax.tick_params(axis='x', labelrotation=90)
    
    plt.xticks(rotation = 90)
    plt.title(title)
        
    plt.show()
    
    fig.savefig(f'{plot_path}/{filename}.png', 
          facecolor='white', 
          transparent=False)
    
    
    
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
    df = parameters['data']
    
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
    fig, ax = plt.subplots()
    
    n = ax.hist(df[column],
                bins=num_bins, 
               )
    ax.set_xlabel(parameters['xlabel'],
                  fontsize=fontsize
                 )
    ax.set_ylabel(parameters['ylabel'],
                  fontsize=fontsize
                 )
    
    if parameters['log_yscale'] == True:
        plt.yscale('log')
    if parameters['log_xscale'] == True:
        plt.xscale('log')
        
    plt.title(parameters['title'],
              fontsize=fontsize
             )
    
    if 'save' in parameters:
        plot_path = parameters['save']['path']
        title = parameters['save']['filename']

        path = os.path.join(plot_path, title)
        fig.savefig(f'{path}', 
                  facecolor='white', 
                  transparent=False)
    plt.show()
    
    
    
def scatter_plot(parameters):
    '''
    Plots the scatterplot
    
    :param parameters: parameters for the plot
    '''
    
    # parameters =  {
    #     'data': df_jaccard,
    #     'fontsize': 14,
    #     'columns': {
    #         'x': 'ratio',
    #         'y': 'count_total_replies',
    #     },
    #     'alpha': 0.5,
    #     'marker_size': 5,
    #     'marker': None,
    #     'xlabel': 'Jaccard coefficent \n (for each IO account, each poster)',
    #     'ylabel': 'Number of daily tweets from poster  ',
    #     'legend_location': '',
    #     'log_yscale': False,
    #     'log_xscale': False,
    #     'bins': None,
    #     'title': f'{year}_{campaign}_per_poster_per_tweet_1day',
    #     'save': {
    #         'path': f'{time_plot_path}',
    #         'filename': f'{year}_{campaign}_jaccard_1day.png'
    #     },
    # }
    
    keys = parameters.keys()
    fig, ax = plt.subplots(figsize=(8, 8))
    fontsize = parameters['fontsize']
    
    colors = ['blue', 'red', 'green', 'orange', 'olive', 'pink', 'lime', 'maroon']
    symbols = ['.', 'o', '+', 'x', '*', 'v', '^', '>']
    
    x_column = parameters['columns']['x']
    y_column = parameters['columns']['y']
    data = parameters['data']
    color = colors[0]
    
    alpha = parameters['alpha'] if 'alpha' in keys else 0.5
    marker_size = parameters['marker_size'] if 'marker_size' in keys else 3
    
    if 'marker' in keys:
        marker = symbols[1]
    else:
        marker = symbols[0]
        
    ax.scatter(data[x_column], 
               data[y_column], 
               marker_size, 
               c=color, 
               alpha=alpha, 
               marker=marker,
           label="Luck")

    ax.set_xlabel(parameters['xlabel'], 
                  fontsize=fontsize)
    ax.set_ylabel(parameters['ylabel'], 
                  fontsize=fontsize)

    ax.tick_params(axis='both', labelsize=fontsize) 
    
    if 'log_yscale' in keys and parameters['log_yscale'] == True:
        ax.set_yscale('log')
    if 'log_xscale' in keys and parameters['log_xscale'] == True:
        ax.set_xscale('log')
        
    if 'title' in keys:
        plt.title(parameters['title'])

    if 'save' in keys:
        path = parameters['save']['path']
        filename = parameters['save']['filename']
        
        file_path = os.path.join(path, filename)
        
        fig.savefig(f'{file_path}', 
              facecolor='white', 
              transparent=False)
        
    plt.show()
    
    
    
    

def time_series_bar_plot(parameters):
    '''
    Creates time series plot
    
    :param parameters: dictionary of parameters to be used in plots
    '''
    
    # timeseries_parameters = {
    #     'data': replies_per_month,
    #     'size': (8, 8),
    #     'fontsize': 14,
    #     'title': iran_202012_campaign,
    #     'sharex': True,
    #     'sharey': True,
    #     'columns': [
    #         {
    #             'x': 'tweet_time_month',
    #             'y': 'ratio_reply',
    #             'label': 'Information Operation',
    #         },
    #         {
    #             'x': 'tweet_time_month',
    #             'y': 'ratio_reply',
    #             'label': 'Control',
    #         },
    #     ],
    #     'x': 'Percentage of replies \n to total tweets per month',
    #     'two_column': False,
    #     'ylabel': 'Ratio of replies \n to total tweets per month',
    #     'xlabel': 'Time (Year-month)',
    #     'legend_location': 'upper right',
    #     'labels_location': {
            # 'x_index': list(),
            # 'labels': label_list()
    # },
    #     'random_color': False,
    #     'save': {
    #         'path': f'{plot_path}',
    #         'filename': f'*.png'
    #     },
    # }
    
    if 'size' in parameters:
        size = parameters['size']
    else:
        size = (10, 10)
        
    data = parameters['data']
    
    fig, ax = plt.subplots(figsize=size)
        
    fontsize = 14
    colors = ['red', 'blue', 'orange', 'red', 'olive', 
              'pink', 'lime', 'maroon']
    x_index = np.arange(len(data))
    
    x_label = parameters['xlabel']
    y_label = parameters['ylabel']
    title = parameters['title']
    
    for index, df in enumerate(parameters['data']):
        x = parameters['columns'][index]['x']
        y = parameters['columns'][index]['y']
        label = parameters['columns'][index]['label']
        
        ax.bar(df[x], 
                df[y],
                color=colors[index],
                label=label,
                linewidth=2,
               )
       
    if 'yscale' in parameters and parameters['yscale'] == True:
        ax.set_yscale('log')
    
    ax.set_ylabel(y_label, fontsize=fontsize, labelpad=1)
    ax.set_xlabel(x_label, fontsize=fontsize)
    
    if 'labels_location' in parameters:
        if 'x_index' in parameters['labels_location']:
            x_index = parameters['labels_location']['x_index']
            label_list = parameters['labels_location']['label_list']
            plt.xticks(df[x][x_index], rotation=30, ha='right')

    ax.tick_params(axis='both', which='both', 
                   labelsize=8, labelbottom=True)
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.yaxis.set_tick_params(labelbottom=True)

    frameon = False
    if 'frameon' in parameters:
        frameon = parameters['frameon']
        
    ax.legend(loc="upper right", 
              frameon=frameon, 
              fontsize=fontsize)
    
    ax.set_title(f'{title}')
    
    plt.title(title)
        
    plt.show()
    
    if 'save' in parameters:
        plot_path = parameters['save']['path']
        filename = parameters['save']['filename']
    
        path_with_filename = f'{plot_path}' + os.sep + f'{filename}.png'
        
        fig.savefig(path_with_filename, 
                    facecolor='white', 
                    transparent=False)

        

def kde(parameters):
    '''
    Plots ccdf for data
    
    :param parameters: parameters to set for the plot
    '''
    
    # {
    #     'data': df,
    #     'fontsize': 14,
    #     'complementary': True,
    #     'columns': [
    #         {'column': ''
    #          'label': '',
    #         },{
    #         'column': '',
    #          'label': ''
    #         }
    #     ],
    #     'xlabel': '',
    #     'ylabel': '',
    #     'legend_location': '',
    #     'log_yscale': True,
    #     'log_xscale': True,
    #     'save': {
    #         'path': '',
    #         'filename': ''
    #     },
        # 'random_color': False
    # }
    
    keys = parameters.keys()
    fig, ax = plt.subplots(figsize=(8, 8))
    fontsize = parameters['fontsize']
    colors = ['red', 'blue', 'orange', 'orange', 'olive', 'pink', 'lime', 'maroon']
    total_columns = len(parameters['columns'])
    
    if parameters['random_color'] == True:
        all_colors =  [k for k,v in pltc.cnames.items()]
        colors = sample(all_colors, total_columns)
    
    symbols = ['.', 'o', '+', 'x', '*', 'v', '^', '>']
    
    i = 0
    cmap = plt.cm.get_cmap('hsv', total_columns)
    
    for data in parameters['data']:
        column = parameters['columns'][i]['column']
        data = parameters['data'][i][column]
        label = parameters['columns'][i]['label']
            
        sns.kdeplot(data, 
                     label=label,
                     color=colors[i],
                     ax=ax,
                     linewidth=2,)

        i = i + 1
        
    ax.set_xlabel(parameters['xlabel'], 
                  fontsize=fontsize)
    ax.set_ylabel(parameters['ylabel'], 
                  fontsize=fontsize)

    ax.tick_params(axis='both', labelsize=fontsize) 
    
    if 'legend_location' in keys:
        ax.legend(loc=parameters['legend_location'], 
                  frameon=True, fontsize=fontsize)
        
    if 'legend_lower' in keys:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', 
                  bbox_to_anchor=(1, -0.06),
                  fancybox=True, 
                  shadow=True, ncol=3)
    
    if 'log_yscale' in keys and parameters['log_yscale'] == True:
        ax.set_yscale('log')
    if 'log_xscale' in keys and parameters['log_xscale'] == True:
        ax.set_xscale('log')
        
    if 'title' in keys:
        plt.title(parameters['title'])

    if 'save' in keys:
        path = parameters['save']['path']
        filename = parameters['save']['filename']
        print(f'{path}/{filename}')
        fig_path = os.path.join(path, filename)
        print(fig_path)
        fig.savefig(fig_path, 
              facecolor='white', 
              transparent=False)
        
    plt.show()
    
    
    

def bar_graph(parameters):
    '''
    Plots the bar graph
    '''
    
    # parameters =  {
    #     'fontsize': 14,
    #     'size': (10, 12),
    #     'bar': 'h',
    #     'x': x,
    #     'y': y,
    #     'xlabel': 'Number of targets (Egypt_022020 campaign)', 
    #     'ylabel': 'Countries',
    #     'legend_location': '',
    #     # 'labelrotation': 90,
    #     # 'title': 'Egypt_022020',
    #     'save': {
    #         'path': reply_plot_path,
    #         'filename': 'egypt_022020_target_country.png'
    #     },
    #     'random_color': False
    # }
    
    keys = parameters.keys()
    
    if 'size' in keys:
        size = parameters['size']
    else:
        size = (8, 8)
        
    
    fig, ax = plt.subplots(figsize=size)
        
    fontsize = parameters['fontsize']
    colors = ['red', 'blue', 'orange', 'red', 'olive', 
              'pink', 'lime', 'maroon']
    
    x = parameters['x']
    y = parameters['y']
    
    if ('bar' in keys) and parameters['bar'] == 'h':
        ax.barh(x,
               y, 
               alpha=0.5,
               color=colors[1],
                )
    else:
        ax.bar(x,
           y, 
           alpha=0.5,
           color=colors[1],
            )
        
    ax.set_ylabel(parameters['ylabel'], fontsize=fontsize)
    ax.set_xlabel(parameters['xlabel'], fontsize=fontsize)

    if 'tick_size' in keys:
        tick_size = parameters['tick_size']
    else:
        tick_size = fontsize
        
    ax.tick_params(axis='both', 
                   which='both', 
                   labelsize=tick_size, 
                   labelbottom=True)
    
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.yaxis.set_tick_params(labelbottom=True)
    
    #######################

    if 'legend_location' in keys:
        if 'legend_font' in keys:
            legend_font = parameters['legend_font']
        else:
            legend_font = fontsize
            
        ax.legend(loc=parameters['legend_location'], 
                  frameon=True, 
                  fontsize=legend_font
                 )
        
    if 'legend_lower' in keys:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', 
                  bbox_to_anchor=(1, -0.06),
                  fancybox=True, 
                  shadow=True, ncol=3)
    
    
        # ax.xaxis.set_minor_locator(AutoMinorLocator())

    if 'figure_text' in keys:
        plt.text(parameters['figure_text_x'], 
                 parameters['figure_text_y'], 
                 parameters['figure_text'], 
                 fontsize=parameters['figure_font'],
                 ha="center", 
                 va="center", 
                )
        
    
    #######################

    if 'title' in keys:
        title = parameters['title']
        ax.set_title(f'{title}')

    if ('labelrotation' in keys) and parameters['labelrotation'] != None:
        labelrotation = parameters['labelrotation']
    else:
        labelrotation = None
        
    ax.tick_params(axis='x', labelrotation=labelrotation)
    
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
#     ax.spines['bottom'].set_color('none')
#     ax.spines['left'].set_color('none')
    fig.tight_layout()

    if 'show' in keys and parameters['show'] == False:
        plt.close()
    else:
        plt.show()
    
    if 'save' in keys:
        save_path = parameters['save']['path']
        filename = parameters['save']['filename']

        fig.savefig(f'{save_path}/{filename}', 
              facecolor='white', 
              transparent=False)
        
        
        
        
def line_plot(parameters):
    '''
    Plots ccdf for data
    
    :param parameters: parameters to set for the plot
    '''
    
    # {
    #     'data': df,
    #     'fontsize': 14,
    #     'complementary': True,
    #     'columns': [
    #         {'column': ''
    #          'label': '',
    #         },{
    #         'column': '',
    #          'label': ''
    #         }
    #     ],
        # 'x': '',
    #     'xlabel': '',
    #     'ylabel': '',
    #     'legend_location': '',
    #     'log_yscale': True,
    #     'log_xscale': True,
    #     'save': {
    #         'path': '',
    #         'filename': ''
    #     },
        # 'random_color': False
    # }
    
    keys = parameters.keys()
    
    if 'size' in keys:
        size = parameters['size']
    else:
        size = (8,8)
        
    fig, ax = plt.subplots(figsize=size)
    
    fontsize = parameters['fontsize']
    colors = ['red', 'blue', 'black', 'orange', 'olive', 'pink', 'lime', 'maroon']
    total_columns = len(parameters['columns'])
    
    if parameters['random_color'] == True:
        all_colors =  [k for k,v in pltc.cnames.items()]
        colors = sample(all_colors, total_columns)
    
    symbols = ['x', 'o', '+', '*', '.', 'v', '^', '>']
    
    i = 0
    cmap = plt.cm.get_cmap('hsv', total_columns)
    x = parameters['data'][parameters['x']]
    for i, column in enumerate(parameters['columns']):
        data = parameters['data'][column['column']]
        label = column['label']
            
        ax.plot(x,
                data, 
                     # complementary=parameters['complementary'],
                 label=label,
                 marker=symbols[i],
                 color=colors[i],
                 # ax=ax,
                 linewidth=2,
                markersize=14
               )

    if 'x_ticks' in keys:
        labels = parameters['data'][parameters['x_ticks']].tolist()
        plt.xticks(x)
        
        ax.set_xticklabels(labels, fontsize=fontsize + 2)
    
    ax.set_xlabel(parameters['xlabel'], 
                  fontsize=fontsize + 2)
    ax.set_ylabel(parameters['ylabel'], 
                  fontsize=fontsize + 2)

    ax.tick_params(axis='both', labelsize=fontsize) 
    
    if 'legend_location' in keys:
        legend_size = parameters['legend_size']
        ax.legend(loc=parameters['legend_location'], 
                  frameon=True, fontsize=legend_size)
        
    if 'legend_lower' in keys:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', 
                  bbox_to_anchor=(1, -0.06),
                  fancybox=True, 
                  shadow=True, ncol=3)
    
    if 'log_yscale' in keys and parameters['log_yscale'] == True:
        ax.set_yscale('log')
    if 'log_xscale' in keys and parameters['log_xscale'] == True:
        ax.set_xscale('log')
    
    ax.set_aspect('auto')

    if 'title' in keys:
        plt.title(parameters['title'])
        
    plt.ylim([0.5, 1])
    fig.tight_layout()

    if 'save' in keys:
        path = parameters['save']['path']
        filename = parameters['save']['filename']
        print(f'{path}/{filename}')
        fig_path = os.path.join(path, filename)
        print(fig_path)
        fig.savefig(fig_path, 
              facecolor='white', 
              transparent=False)
        
    plt.show()
    
    
    
def word_cloud(word_list,
               parameters=None, 
               filename=None):
    from wordcloud import WordCloud

    text = ' '.join(word_list)

    wordcloud = WordCloud(
        width=1000,
        height=1000,
        background_color='white',  
        colormap='viridis',
        collocations=False
    ).generate(text)

    if parameters != None and 'size' in parameters:
        size=parameters['size']
    else:
        size = (10,10)
        
    fig, ax = plt.subplots(figsize=size)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    if filename != None:
        # wordcloud.to_file(filename)
        fig.tight_layout()

        
        fig.savefig(filename, 
              facecolor='white', 
              transparent=False)

    plt.show()
        
        
        
def grouped_boxplot(parameters):
    '''
    Plots the boxplot by grouping
    '''
    # parameters = {
    #     'data': [df_all_pos_men, df_all_neg_men],
    #     'fontsize': 14,
    #     'complementary': True,
    #     'columns': [
    #         {'column': 'followers_count',
    #          'label': 'Follower',
    #         },{
    #         'column': 'followers_count',
    #          'label': 'Following'
    #         }
    #     ],
    #     'group': ['IO-replier', 'Normal-replier'],
    #     'xlabel': 'Replier',
    #     'ylabel': 'Log(x)',
    #     'legend_location': '',
    #     'yscale': True,
    #     'save': {
    #         'path': './plots',
    #         'filename': 'boxplot_follower_following_IO_normal.png'
    #     },
    #     'random_color': False
    # }
    
    if 'size' in parameters:
        size = parameters['size']
    else:
        size=(8,8)
        
    keys = parameters.keys()
    fig, ax = plt.subplots(figsize=size)
    fontsize = parameters['fontsize']
    data = parameters['data']
    
    columns = {}
    labels = {}
    for i, df in enumerate(data):
        columns[i] = {
            'column': [],
            'label': []
        }
        
        for j, row in enumerate(parameters['columns']):
            column = row['column']
            label = row['label']
            
            if 'log_yscale' in keys and parameters['log_yscale'] == True:
                add = 1
            else:
                add = 0
                
            df[column] = df[column] + add
            
            columns[i]['column'].append(df[column])
            columns[i]['label'].append(label)
        
    ticks = range(len(parameters['columns']))
        
    def define_box_properties(plot_name, color_code, label):
        for k, v in plot_name.items():
            plt.setp(plot_name.get(k), color=color_code)

        # use plot function to draw a small line to name the legend.
        plt.plot([], c=color_code, label=label)
        plt.legend()
        
        
    plots = []
    for i in range(len(columns)):
        data1 = columns[i]['column']
               
        position = np.array(
                   np.arange(len(data1)))*2.0
        
        if i == 0:
            position = position - 0.35
        else:
            position = position + 0.35
            
        plot1 = ax.boxplot(data1,
                       positions=position, 
                       widths=0.6,
                       # showfliers=False
                          )
        plots.append(plot1)
        
        
    colors  = ['red', 'blue', 'black']
    if 'group' in keys:
        for i, plot in enumerate(plots):
            define_box_properties(plot, 
                                  colors[i], 
                                  parameters['group'][i])
        # define_box_properties(plot2, 'blue', 'Winter')

    
    if 'yscale' in keys and parameters['yscale'] == True:
        if 'ybase' in keys:
            ybase = parameters['ybase']
        else:
            ybase = 10
        ax.set_yscale('log', base=ybase)
        
    ax.tick_params(axis='x', which='major', labelsize=fontsize)

    ax.set_xlabel(parameters['xlabel'], 
                  fontsize=fontsize)
    ax.set_ylabel(parameters['ylabel'], 
                  fontsize=fontsize)
    
    if 'title' in keys:
        title = parameters['title']
        ax.set_title(f'{title}')
    
    plt.xticks(np.arange(0, len(ticks) * 2, 2), columns[0]['label'])
 
    # set the limit for x axis
    plt.xlim(-2, len(ticks)*2)
    
    fig.tight_layout()

    if 'save' in keys:
        path = parameters['save']['path']
        filename = parameters['save']['filename']
        
        fig.savefig(f'{path}/{filename}', 
              facecolor='white', 
              transparent=False)
        
    plt.show()
    
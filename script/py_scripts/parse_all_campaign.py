import os
import sys

import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/N/u/potem/Quartz/project/infoOps-strategy/script/helper')
from strategy_helper import *
from helper import *

ouput_path = '/N/slate/potem/data/derived/all_tweets/'
script_path = '/N/u/potem/Quartz/project/infoOps-strategy/script/py_scripts'
script_file = 'parse-zip-data.py'
control_folder = 'DriversControl'
control_filename = 'control_driver_tweets.jsonl.gz'

campaigns = {
    '2021_12': [
                'CNHU_0621', 'CNCC_0621', 'MX_0621', 
                'REA_0621', 'RNA_0621', 'Tanzania_0621', 
                'uganda_0621', 
                'Venezuela_0621'
               ],
    '2020_12': ['armenia_202012', 'GRU_202012', 'IRA_202012', 
                'iran_202012'],
    '2020_09': ['ira_092020', 'iran_092020', 'thailand_092020'],
    '2020_08': ['cuba_082020', 'qatar_082020'],
}

reports = ['thailand_092020', 'qatar_082020', 
           'iran_092020',
           'cuba_082020', 'ira_092020', 
           'Venezuela_0621', 
           'iran_202012',
           'Tanzania_0621', 'MX_0621', 
           'CNHU_0621', 'CNCC_0621',
          ]

image_campaigns = {
    '2019_06': ['catalonia_201906_1', 
                'russia_201906_1',
                'iran_201906_1',
                'iran_201906_2',
                'iran_201906_3',
                'venezuela_201906_1'
               ],
    '2019_01': ['iran_201901_1',
                'russia_201901_1',
                'bangladesh_201901_1',
                'venezuela_201901_1',
                'venezuela_201901_2'
               ],
    '2018_10': ['ira',
                'iranian'],
}


remaining = {
    '2020_05': {
        'china_052020': ['china_052020_tweets_csv_unhashed.csv.gz'],
        'turkey_052020': [
            'turkey_052020_tweets_csv_unhashed_2009.csv.gz',
            'turkey_052020_tweets_csv_unhashed_2010.csv.gz',
            'turkey_052020_tweets_csv_unhashed_2011.csv.gz',
            'turkey_052020_tweets_csv_unhashed_2012.csv.gz',
            'turkey_052020_tweets_csv_unhashed_2013.csv.gz',
            'turkey_052020_tweets_csv_unhashed_2014.csv.gz',
            'turkey_052020_tweets_csv_unhashed_2015.csv.gz',
            'turkey_052020_tweets_csv_unhashed_2016_01.csv.gz',
            'turkey_052020_tweets_csv_unhashed_2016_06.csv.gz',
            'turkey_052020_tweets_csv_unhashed_2017_01.csv.gz',
            'turkey_052020_tweets_csv_unhashed_2017_06.csv.gz',
            'turkey_052020_tweets_csv_unhashed_2018_01.csv.gz',
            'turkey_052020_tweets_csv_unhashed_2018_06.csv.gz',
            'turkey_052020_tweets_csv_unhashed_2019_01.csv.gz',
            'turkey_052020_tweets_csv_unhashed_2019_06.csv.gz',
            'turkey_052020_tweets_csv_unhashed_2020_01.csv.gz'
            ],
        'russia_052020': ['russia_052020_tweets_csv_unhashed_1.csv.gz',
                          'russia_052020_tweets_csv_unhashed_2.csv.gz'
                         ]
               },
    '2020_04': {
        'egypt_022020': ['egypt_022020_tweets_csv_unhashed.csv.gz'],
        'honduras_022020': ['honduras_022020_tweets_csv_unhashed.csv.gz'],
        'indonesia_022020': ['indonesia_022020_tweets_csv_unhashed.csv.gz'],
        'sa_eg_ae_022020': ['sa_eg_ae_022020_tweets_csv_unhashed_01.csv.gz',
                            'sa_eg_ae_022020_tweets_csv_unhashed_01.csv.gz',
                            'sa_eg_ae_022020_tweets_csv_unhashed_02.csv.gz',
                            'sa_eg_ae_022020_tweets_csv_unhashed_03.csv.gz',
                            'sa_eg_ae_022020_tweets_csv_unhashed_04.csv.gz',
                            'sa_eg_ae_022020_tweets_csv_unhashed_05.csv.gz',
                            'sa_eg_ae_022020_tweets_csv_unhashed_06.csv.gz',
                            'sa_eg_ae_022020_tweets_csv_unhashed_07.csv.gz',
                            'sa_eg_ae_022020_tweets_csv_unhashed_08.csv.gz',
                            'sa_eg_ae_022020_tweets_csv_unhashed_09.csv.gz',
                            'sa_eg_ae_022020_tweets_csv_unhashed_10.csv.gz'
                            
                           ],
        'serbia_022020': ['serbia_022020_tweets_csv_unhashed_01.csv.gz',
                          'serbia_022020_tweets_csv_unhashed_02.csv.gz',
                          'serbia_022020_tweets_csv_unhashed_03.csv.gz',
                          'serbia_022020_tweets_csv_unhashed_04.csv.gz',
                          'serbia_022020_tweets_csv_unhashed_05.csv.gz',
                          'serbia_022020_tweets_csv_unhashed_06.csv.gz',
                          'serbia_022020_tweets_csv_unhashed_07.csv.gz',
                          'serbia_022020_tweets_csv_unhashed_08.csv.gz',
                          'serbia_022020_tweets_csv_unhashed_09.csv.gz',
                          'serbia_022020_tweets_csv_unhashed_10.csv.gz',
                          'serbia_022020_tweets_csv_unhashed_11.csv.gz',
                          'serbia_022020_tweets_csv_unhashed_12.csv.gz',
                          'serbia_022020_tweets_csv_unhashed_13.csv.gz']
        
    } ,
    '2020_03': {
        'ghana_nigeria_032020': ['unhashed_2020_03_ghana_nigeria_032020_tweets_csv_unhashed.csv.gz'],
    },
    '2019_11': {
        'saudi_arabia_112019': [
            'saudi_arabia_112019_tweets_csv_unhashed_1.csv.gz',
            'saudi_arabia_112019_tweets_csv_unhashed_2.csv.gz',
            'saudi_arabia_112019_tweets_csv_unhashed_3.csv.gz',
            'saudi_arabia_112019_tweets_csv_unhashed_4.csv.gz',
            'saudi_arabia_112019_tweets_csv_unhashed_5.csv.gz',
            'saudi_arabia_112019_tweets_csv_unhashed_6.csv.gz',
            'saudi_arabia_112019_tweets_csv_unhashed_7.csv.gz',
            'saudi_arabia_112019_tweets_csv_unhashed_8.csv.gz',
            'saudi_arabia_112019_tweets_csv_unhashed_9.csv.gz'
                               ]
    },
    
    '2019_08': {
        'china_082019_1': ['china_082019_1_tweets_csv_unhashed.csv.gz'],
        'china_082019_2': ['china_082019_2_tweets_csv_unhashed.csv.gz'],
        'china_082019_3': ['china_082019_3_tweets_csv_unhashed_part1.csv.gz',
                           'china_082019_3_tweets_csv_unhashed_part2.csv.gz',
                           'china_082019_3_tweets_csv_unhashed_part3.csv.gz'
                          ],
        'ecuador_082019_1': ['ecuador_082019_tweets_csv_unhashed.csv.gz'],
        'egypt_uae_082019_1': ['egypt_uae_082019_tweets_csv_unhashed.csv.gz'],
        'saudi_arabia_082019_1': ['unhashed_2019_08_saudi_arabia_082019_1_saudi_arabia_082019_1_tweets_csv_unhashed.zip'],
        'spain_082019_1': ['spain_082019_tweets_csv_unhashed.csv.gz'],
        'uae_082019_1': ['uae_082019_tweets_csv_unhashed.csv.gz']
    },
    
    '2018_10': {
        'ira': ['ira_tweets_csv_unhashed.csv.gz'],
        'iranian': ['iranian_tweets_csv_unhashed.csv.gz']
    },
}

test = {
    '2020_05': {
        'china_052020': ['china_052020_tweets_csv_unhashed.csv.gz'],
        'turkey_052020': ['all.csv.gz'],
        'russia_052020': ['all.csv.gz']
               },
    '2020_04': {
        'egypt_022020': ['egypt_022020_tweets_csv_unhashed.csv.gz'],
        'honduras_022020': ['honduras_022020_tweets_csv_unhashed.csv.gz'],
        'indonesia_022020': ['indonesia_022020_tweets_csv_unhashed.csv.gz'],
        'sa_eg_ae_022020': ['all.csv.gz'],
        'serbia_022020': ['all.csv.gz']
        
    } ,
    '2020_03': {
        'ghana_nigeria_032020': ['unhashed_2020_03_ghana_nigeria_032020_tweets_csv_unhashed.csv.gz'],
    },
    '2019_11': {
        'saudi_arabia_112019': ['all.csv.gz']
    },
    '2019_08': {
        'china_082019_1': ['china_082019_1_tweets_csv_unhashed.csv.gz'],
        'china_082019_2': ['china_082019_2_tweets_csv_unhashed.csv.gz'],
        'china_082019_3': ['all.csv.gz'],
        'ecuador_082019_1': ['ecuador_082019_tweets_csv_unhashed.csv.gz'],
        'egypt_uae_082019_1': ['egypt_uae_082019_tweets_csv_unhashed.csv.gz'],
        'saudi_arabia_082019_1': ['unhashed_2019_08_saudi_arabia_082019_1_saudi_arabia_082019_1_tweets_csv_unhashed.zip'],
        'spain_082019_1': ['spain_082019_tweets_csv_unhashed.csv.gz'],
        'uae_082019_1': ['uae_082019_tweets_csv_unhashed.csv.gz']
    },
    
    '2019_01': {
        'iran_201901_1' : ['all.csv.gz'], 
        'russia_201901_1' : ['russian_linked_tweets_csv_unhashed.csv.gz'],
        'bangladesh_201901_1' : ['bangladesh_linked_tweets_csv_unhashed.csv.gz'],
        'venezuela_201901_1' : ['all.csv.gz'],
        'venezuela_201901_2' : ['venezuela_linked_tweets_csv_unhashed.csv.gz']
   },
    '2018_10': {
        'ira': ['ira_tweets_csv_unhashed.csv.gz'], 
        'iranian': ['iranian_tweets_csv_unhashed.csv.gz']
    },
    '2019_06': {
        'catalonia_201906_1': ['catalonia_201906_1_tweets_csv_unhashed.csv.gz'],
        'russia_201906_1': ['russia_201906_1_tweets_csv_unhashed.csv.gz'], #need checking
        'iran_201906_1':['russia_201906_1_tweets_csv_unhashed.csv.gz'],
        'iran_201906_2':['iran_201906_2_tweets_csv_unhashed.csv.gz'],
        'iran_201906_3':['iran_201906_3_tweets_csv_unhashed.csv.gz'],
        'venezuela_201906_1': ['venezuela_201906_1_tweets_csv_unhashed.csv.gz']
    },
}

all_campaigns = {
    '2021_12': [
                'CNHU_0621', 'CNCC_0621', 'MX_0621', 
                'REA_0621', 'RNA_0621', 'Tanzania_0621', 
                'uganda_0621', 
                'Venezuela_0621'
               ], #done
    '2020_12': ['armenia_202012', 'GRU_202012', 'IRA_202012', 
                'iran_202012'], #done
    '2020_09': ['ira_092020', 'iran_092020', 'thailand_092020'], #done
    '2020_08': ['cuba_082020', 'qatar_082020'], #done
    '2020_05': [ #done
        'china_052020', 'turkey_052020', 
        'russia_052020'],
    '2020_04': [ #done
        'egypt_022020', 'honduras_022020', 
        'indonesia_022020', 'sa_eg_ae_022020', 
        'serbia_022020'],
    '2020_03': ['ghana_nigeria_032020'], #done
    '2019_11': ['saudi_arabia_112019'], #done
    '2019_08': [ #done
        'china_082019_1', 'china_082019_2', 
        'china_082019_3', 'ecuador_082019_1', 
        'egypt_uae_082019_1', 'saudi_arabia_082019_1', 
        'spain_082019_1', 'uae_082019_1'
    ],
    '2019_01': [ #done
        'iran_201901_1' , 
        'russia_201901_1',
        'bangladesh_201901_1',
        'venezuela_201901_1',
        'venezuela_201901_2'],
    '2018_10': ['ira', 'iranian']
}

merge_list = {
    '2019_01': ['iran_201901_1', 'venezuela_201901'],
    '2019_08' : ['china_082019'],
    '2019_06': ['iran_201906']
    
}

merge_paths = '/N/slate/potem/data/derived/all_tweets/'


path = '/N/slate/potem/YYYY_MM/'
script = os.path.join(script_path, script_file)

def run_python_script(test=test, script=script, 
                      path=path, ouput_path=output_path):
    '''
    Runs the python script from the python
    
    :param test: dictionary of campaign year and campaign files
    :param script: full path to python script file with file name
    :param path: path of raw campaign files
    :param output_path: path where parsed file is to be saved
    '''
    for year in test:
        # create_folder(ouput_path, year)

        # new_ouput_path = os.path.join(ouput_path, year)

        for new_campaign in test[year]:
            for campaign_file in test[year][new_campaign]:

                input_folder = os.path.join(path, year, new_campaign)

                print(f'\n\n ---- Starting for campaign : {new_campaign} -- \n')

                #Example format, use allcsv.gz if there are multiple files
                #within the one folder
                # python parse-zip-data.py
                # --input=/N/slate/potem/YYYY_MM/2019_01/venezuela_201901_1 
                # --output=/N/slate/potem/data/derived/all_tweets 
                # --campaign-name=venezuela_201901_1 
                # --filename=venezuela_201901_1_tweets_csv_unhashed_3.csv.gz

                command = f"python {script} --input={input_folder} --output={ouput_path} --campaign-name={new_campaign} --filename={campaign_file}"

                os.system(command)  

                print(f'\n\n ---- Ending for campaign : {new_campaign} -- \n')


def append_all_files(merge_paths, merge_list):
    '''
    Merge multiple files within a folder (2 level of folder, folder_1->folder_2->files)
    
    :param merge_paths: path of folder in 1st level
    :param merge_list: dictionary of folder_1 as keys and folder_2 as dictionary 
    '''
    
    for year in merge_list:
        for campaign in merge_list[year]:
            destination_path = os.path.join(merge_paths, year, campaign)
            source_path = os.path.join(merge_paths, year, campaign, f'*.pkl.gz')
            df = pd.DataFrame()
            for ops_file in glob.glob(source_path):
                print(ops_file)

                df_new = pd.read_pickle(ops_file)
                print(len(df_new))
                df = df.append(df_new, 
                          ignore_index=True)

                print(len(df))

            df.to_pickle(f'{destination_path}/{campaign}_tweets.pkl.gz')

def parse_all_control_data(all_campaigns, input_path,
                           control_folder, filename, output_path
                          ):
    '''
    Parse the control jsonl.gz file and converts to pkl.gz file
    
    :param all_campaigns: dictionary of year and campaign list
    :param input_path: location to the input control file
    :param control_folder: name of new directory for control file in ouput path
    :param filename: name of control file in input path
    :param output_path: location where output file is to be saved
    '''
    
    for year in all_campaigns:
        for campaign in all_campaigns[year]:
            input_folder_path = os.path.join(input_path, 
                                             year, campaign, 
                                             control_folder, filename
                                            )
            new_output_path = os.path.join(output_path, year, campaign)

            create_folder(new_output_path, control_folder)

            new_output_path = os.path.join(new_output_path, control_folder)

            parse_control_data(campaign, input_folder_path,
                               new_output_path
                              )
            
            
def main():
    try:
        print('\n ------- Start: Parsing raw InfoOps files ------ \n')
        run_python_script()
        print('\n ------- End: Parsing raw InfoOps files ------ \n')
        
        print('\n ------- Start: Merging multiple InfoOps files ------ \n')
        append_all_files(merge_paths, merge_list)
        print('\n ------- End: Merging multiple InfoOps files ------ \n')
        
        print('\n ------- Start: Parsing control InfoOps files ------ \n')
        parse_all_control_data(all_campaigns, path,
                               control_folder, control_filename, 
                               output_path
                              )
        print('\n ------- End: Parsing control InfoOps files ------ \n')
        
    except Exception as e:
        print(e)
        return e
    

if __name__ == "__main__":
    main()
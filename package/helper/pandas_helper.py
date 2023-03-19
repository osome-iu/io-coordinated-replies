import pandas as pd
import numpy as np

def read_first_last_row_of_grp(df, group_by, sort_by):
    '''
    Gets the first and last row within the group after groupby
    
    :param df: Dataframe
    :param group_by: column to groupby
    :param sort_by: column to sort the dataframe
    
    :return Dataframe
    '''
    
    grps = df.groupby(group_by)
    
    return (pd.concat([grps.head(1),
                            grps.tail(1)])
                   .drop_duplicates()
                   .sort_values(sort_by)
                   .reset_index(drop=True)
                 )


def add_YYYY_MM_DD(df, column):
    '''
    Extracts year, month and day from timestamp column and adds 
    the column to dataframe
    
    :param df: Dataframe
    :param column: Column name for timestamp
    
    :return Dataframe
    '''
    
    df[column] = pd.to_datetime(df[column])
    df[f'{column}_year'] = df[column].map(
                lambda x: x.strftime('%Y-%m-%d'))
    
    return df
    
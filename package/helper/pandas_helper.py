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
    
    
def time_difference(df,
                    column_to_group,
                    column_time,
                    new_column_name,
                    unit_for_diff='hour'
                ):
    '''
    :param df: Dataframe
    :param column_to_group: column name to groupby
    :param column_time: column name that has time
    :param new_column_name: new column name to 
    save time difference value
    :param unit_for_diff: unit to convert the time difference
    
    :return dataframe
    '''
    
    df[new_column_name] = df.groupby(column_to_group)[column_time].diff()
    df = df.dropna(subset=[new_column_name])
    df[new_column_name] = df[new_column_name] + datetime.timedelta(seconds=1)

    diff_column = new_column_name+ f'_{unit_for_diff}'
    
    if unit_for_diff == 'hour':
        df[diff_column] = df['diff'].apply(
            lambda x: np.ceil(x.total_seconds() / 3600))
    if unit_for_diff == 'min':
        df[diff_column] = df['diff'].apply(
            lambda x: np.ceil(x.total_seconds() / 60))
        
        
    return df
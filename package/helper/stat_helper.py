import pandas as pd
import numpy as np
from scipy import stats

import pandas as pd
import numpy as np
import datetime
import warnings
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import sys
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

from scipy.stats import ks_2samp
from scipy.stats import mannwhitneyu
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from sklearn.model_selection import cross_validate



def mann_whitneyu_test(data1, data2):
    '''
    Perform the Mann Whitneyu Test
    :param data1: data distribution
    :param data2: data distribution
    '''
    s, p_value = mannwhitneyu(data1, data2)
    
    # Print the results
    print("Mann-Whitney U statistic:", s)
    print("p-value:", p_value)
    

def read_data(input_path, filename):
    '''
    Read the data file from the given location
    
    :param input_path: Location to input file
    :param filename: Name of the file
    :return pandas dataframe
    '''
                        
    parts = filename.split('.')
                        
    if parts[-1] == 'csv':
        df = pd.read_csv(f'{input_path}/{filename}')
        
        print(df.info())
        
        return df
                        
    if parts[-1] == 'gz' and parts[-2] == 'pkl':
        df = pd.read_pickle(f'{input_path}/{filename}')
        
        if len(df) <= 1:
            print('The dataframe has just one column')
            
            raise Exception('Data insufficient')
        else:
            print(df.info())
            
            return df
    else:
    
        raise Exception(
            '--filename : Invalid file format. \n Only pkl.gz and csv accepted')
        

def ratios(df):
    n_users = df['userid'].nunique()
    retweets = df.loc[df['is_retweet'] == True]
    n_retweets = len(retweets)
    n_tweets = len(df.loc[(df['is_retweet'] == False) & \
                (~df['quoted_tweet_tweetid'].notnull()) & \
                (df['in_reply_to_tweetid'].isnull())])
    n_total = len(df)
    n_replies = len(df_mexico.loc[~df_mexico['in_reply_to_tweetid'].isnull()])
    n_quoted = len(df_mexico.loc[~df_mexico['quoted_tweet_tweetid'].isnull()])

    
    print('\n ------- Retweet --------------\n')
    df_retweet = df.loc[df['is_retweet'] == True]
    df_retweet_grp = df_retweet.groupby(
        ['retweet_tweetid']).size().to_frame(
        'retweet_count').reset_index()
    
    print('\n Retweet to user ratio :', round(n_retweets/n_users, 2))
    print('\n Retweet to tweet ratio :', round(n_retweets/n_tweets, 2))
    
    
    print('\n ------- Replies -------------\n')
    print('\n Replies to user ratio :', round(n_replies/n_users))
    print('\n Replies to tweet ratio : ', round(n_replies/n_tweets))
    print('\n Maximum replies a tweet got :',
          max(df_mexico_tweet_grp['retweet_count']))
    

def statistics(df, 
               column_to_groupby='poster_tweetid',
               column_to_take='age'
              ):
    '''
    Get the statistics of column
    '''
    df_stat = (df
              .groupby([column_to_groupby])[column_to_take]
              .describe()
               # .to_frame()
              .reset_index()
             )
    
    df_stat = df_stat.rename(columns={
        'count': f'count_{column_to_take}',
        'min': f'min_{column_to_take}',
        'max': f'max_{column_to_take}',
        'mean': f'mean_{column_to_take}',
        'median': f'median_{column_to_take}',
        'std': f'std_{column_to_take}',
        '50%': f'50%_{column_to_take}',
        '25%': f'25%_{column_to_take}',
        '75%': f'75%_{column_to_take}'
    })
    
    df_skew = (df
              .groupby([column_to_groupby])[column_to_take]
              .skew(skipna=False)
              .to_frame(f'skew_{column_to_take}')
              .reset_index()
             )
    df_kurtosis = (df
          .groupby([column_to_groupby])[column_to_take]
          .apply(pd.DataFrame.kurt)
          .to_frame(f'kurtosis_{column_to_take}')
          .reset_index()
         )
    
    df_group = df_stat.merge(df_skew, 
                             on=column_to_groupby)
    df_group = df_group.merge(df_kurtosis,
                              on=column_to_groupby)
    
    return df_group

def entropy(data):
    '''
    Calculates the entropy of distribution
    '''
    a, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy


def all_stat(df, 
             column_to_groupby='poster_tweetid',
             column_to_take='age',
             label='tweet_label',
             list_data=False
            ):
    '''
    Calculates the summary statistics of dataframe
    '''
    if list_data == False:
        list_column = f'list_{column_to_take}'

        df_stat = (df
                   .groupby([column_to_groupby])[column_to_take]
                   .apply(list)
                   .to_frame(list_column)
                   .reset_index()
                  )
    else:
        list_column = column_to_take
        df_stat = df
        df = df.explode(column_to_groupby)
        
    df_des = (df
          .groupby([column_to_groupby])[column_to_take]
          .describe()
           # .to_frame()
          .reset_index()
         )
    print(df_des.head())
    
    df_des = df_des.rename(columns={
        'count': f'count_{column_to_take}',
        'min': f'min_{column_to_take}',
        'max': f'max_{column_to_take}',
        'mean': f'mean_{column_to_take}',
        'median': f'median_{column_to_take}',
        'std': f'std_{column_to_take}',
        '50%': f'50%_{column_to_take}',
        '25%': f'25%_{column_to_take}',
        '75%': f'75%_{column_to_take}'
    })
    
    if label == 'replier_label':
        df_des[f'std_{column_to_take}'] = df_des[f'std_{column_to_take}'] + 1
        
    df_des = df_des.drop([f'count_{column_to_take}'], 
                         axis=1)
    
    df_skew = (df
              .groupby([column_to_groupby])[column_to_take]
              .skew(skipna=False)
              .to_frame(f'skew_{column_to_take}')
              .reset_index()
             )
    df_kurtosis = (df
          .groupby([column_to_groupby])[column_to_take]
          .apply(pd.DataFrame.kurt)
          .to_frame(f'kurtosis_{column_to_take}')
          .reset_index()
         )

    df_group = df_des.merge(df_skew, 
                             on=column_to_groupby)
    df_group = df_group.merge(df_kurtosis,
                              on=column_to_groupby)
    
    df_stat['min'] = df_stat[list_column].apply(
        lambda x: min(x)
    )
    df_stat['max'] = df_stat[list_column].apply(
        lambda x: max(x)
    )
    
    #Range
    df_stat[f'range_{column_to_take}'] = df_stat['max'] - df_stat['min']
    
    df_stat = df_stat.drop(columns=['min', 'max'])
    
    #IQR
    df_stat[f'iqr_{column_to_take}'] = df_stat[list_column].apply(
        lambda x: np.quantile(np.array(x), [0.75])[0] - np.quantile(np.array(x), [0.25])[0]
    
    ) 
   
    # Entropy
    df_stat[f'entropy_{column_to_take}'] = df_stat[list_column].apply(entropy)
    
    df_group = df_group.merge(df_stat,
                              on=column_to_groupby)
    
    grps = df.groupby([column_to_groupby, 
                       label]).groups.keys()
    df_grps = pd.DataFrame(data=grps, columns=[column_to_groupby, 
                                               label])
    
    df_group = df_group.merge(df_grps[[column_to_groupby, 
                                       label]],
                              on=column_to_groupby)
    
    df_group = df_group.drop([list_column], 
                             axis=1)
    
    return df_group



def KS_test(data1, data2):
    '''
    Tes the KS test for data1 and data2
    '''
    statistic, pvalue = ks_2samp(data1, 
                                 data2)


    print('KS test statistic:', statistic)
    print('p-value:', pvalue)
    
    
def limit_statistics(df, 
             column_to_groupby='replier_userid',
             column_to_take='diff_min',
             label='replier_label'
            ):
    '''
    Calculates the summary statistics of dataframe
    '''
    
    list_column = f'list_{column_to_take}'
    df_stat = (df
               .groupby([column_to_groupby])[column_to_take]
               .apply(list)
               .to_frame(list_column)
               .reset_index()
              )
    
    df_des = (df
          .groupby([column_to_groupby])[column_to_take]
          .describe()
           # .to_frame()
          .reset_index()
         )
    
    print(df_des.columns)
    
    df_des = df_des.rename(columns={
        'count': f'count_{column_to_take}',
        'min': f'min_{column_to_take}',
        'max': f'max_{column_to_take}',
        'mean': f'mean_{column_to_take}',
        # 'median': f'median_{column_to_take}',
        'std': f'std_{column_to_take}',
        '50%': f'median_{column_to_take}',
        '25%': f'25%_{column_to_take}',
        '75%': f'75%_{column_to_take}'
    })
    
    df_des = df_des.drop(columns=[f'count_{column_to_take}',
                                  # f'50%_{column_to_take}',
                                  f'25%_{column_to_take}',
                                  f'75%_{column_to_take}',
                                  f'std_{column_to_take}'
                                 ], 
                         axis=1)
    
    df_stat['min'] = df_stat[list_column].apply(
        lambda x: min(x)
    )
    df_stat['max'] = df_stat[list_column].apply(
        lambda x: max(x)
    )
    
    #Range
    df_stat[f'range_{column_to_take}'] = df_stat['max'] - df_stat['min']
    
    df_stat = df_stat.drop(columns=['min', 'max'])
    
    #IQR
    df_stat[f'iqr_{column_to_take}'] = df_stat[list_column].apply(
        lambda x: np.quantile(np.array(x), [0.75])[0] - np.quantile(np.array(x), [0.25])[0]
    
    ) 
    
    # Entropy
    df_stat[f'entropy_{column_to_take}'] = df_stat[list_column].apply(entropy)
    
    df_group = df_des.merge(df_stat,
                              on=column_to_groupby
                           )

    #Adding replier label info
    grps = df.groupby([column_to_groupby, 
                       label]).groups.keys()
    df_grps = pd.DataFrame(data=grps, columns=[column_to_groupby, 
                                               label])
    
    df_group = df_group.merge(df_grps[[column_to_groupby, 
                                       label]],
                              on=column_to_groupby)
    
    return df_group


def run_model_with_best_threshold(df,
              columns_not_include=['list_age'],
              model_type='random', 
              y_column= 'tweet_label',
             ):
    '''
    Trains the model and prints the result
    :param df: Dataframe that has features
    :param columns_not_include: Columns to remove from features
    :param model_type: Type of model
    :param y_column: Whether to do PCA or not

    :return Dataframe
    '''
    print(f'\n **** {model_type} ****')
    
    ### Remove unnecessary columns
    import pickle

    model_filename = filename
    
    columns_not_include.extend(
        ['poster_tweetid','tweet_label', 'replier_userid', 'replier_label'])
    
    columns_to_keep = list(set(df.columns) - set(columns_not_include))

    X = df[columns_to_keep]
    y = df[y_column]
  
    ### Choose model
    if model_type == 'logistic':
        model = LogisticRegression(random_state=0)
    elif model_type == 'random':
        print('Running Random Forest')
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'ada':
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(n_estimators=100,
                                 algorithm="SAMME", random_state=0)
    elif model_type == 'tree':
        from sklearn import tree
        model = tree.DecisionTreeClassifier()
    elif model_type == 'naive':
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
    
    ### Choose scoring function
    from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score

    # Creating a dictionary of scorers
    scoring = {
        'precision': make_scorer(precision_score, average='binary'),
        'recall': make_scorer(recall_score, average='binary'),
        'f1': make_scorer(f1_score, average='binary'),
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
    }

    cv_scores = [
        "train_precision",
        "test_precision",
        "train_recall",
        "test_recall",
        "train_f1",
        "test_f1",
        "train_roc_auc",
        "test_roc_auc",
    ]

    from sklearn.model_selection import TunedThresholdClassifierCV
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.metrics import f1_score

    model = make_pipeline(StandardScaler(), model)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=42)
    tuned_model = TunedThresholdClassifierCV(estimator=model,
                                             scoring='f1',
                                             store_cv_results = True,
                                             n_jobs=-1
                                            )

    cv_results_tuned_model = pd.DataFrame(
        cross_validate(
            tuned_model,
            X,
            y,
            scoring=scoring,
            cv=cv,
            return_train_score=True,
            return_estimator=True,
        )
    )
   
    from sklearn.metrics import f1_score

    decision_threshold = pd.Series(
        [est.best_threshold_ for est in cv_results_tuned_model["estimator"]],
    )
    cv_results_tuned_model['threshold'] = decision_threshold
    
    cv_results_tuned_model['algorithm'] = model_type
    
    return cv_results_tuned_model


def run_oversample_model_with_best_threshold(df, columns_not_include=['list_age'],
                                             model_type='random', y_column='tweet_label',
                                             filename=None, estimator=True):
    '''
    Trains the model and prints the result.
    :param df: Dataframe
    :param model_type: Type of model ('logistic', 'random', etc.)
    :param columns_not_include: columns to not include in the model
    :param y_column: The target column
    :param filename: Optional filename to save the model
    :param estimator: Whether to return the estimator
    :return: DataFrame with cross-validation results
    '''
    print(f'\n **** {model_type} ****')

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import  RepeatedStratifiedKFold, cross_validate
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import make_scorer, roc_auc_score
    from imblearn.ensemble import BalancedRandomForestClassifier
    from imblearn.over_sampling import RandomOverSampler
    import pandas as pd
    from imblearn.pipeline import make_pipeline

    # Filter columns
    columns_not_include.extend(['poster_tweetid', 'tweet_label',
                                'replier_userid', 'replier_label'])
    columns_to_keep = list(set(df.columns) - set(columns_not_include))
    X = df[columns_to_keep]
    y = df[y_column]

    # Choose model
    if model_type == 'logistic':
        model = LogisticRegression(class_weight='balanced', random_state=42, n_jobs=-1)
    elif model_type == 'random':
        model = BalancedRandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'ada':
        model = AdaBoostClassifier(n_estimators=100, random_state=42, algorithm="SAMME")
    elif model_type == 'tree':
        from sklearn import tree
        model = tree.DecisionTreeClassifier()
    elif model_type == 'naive':
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()

    # Make pipeline with oversampling
    model_pipeline = make_pipeline(StandardScaler(), 
                                   RandomOverSampler(sampling_strategy='minority', 
                                                     random_state=0),
                                   model)

    # Define scoring metrics
    scoring = {
        'precision': make_scorer(precision_score, average='binary'),
        'recall': make_scorer(recall_score, average='binary'),
        'f1': make_scorer(f1_score, average='binary'),
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
    }

    # Cross-validation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=42)
    tuned_model = TunedThresholdClassifierCV(estimator=model_pipeline,
                                             scoring='f1',
                                             store_cv_results=True, 
                                             n_jobs=-1)

    cv_results_tuned_model = pd.DataFrame(cross_validate(tuned_model, 
                                                         X, y, 
                                                         scoring=scoring, 
                                                         cv=cv,
                                                         return_train_score=True, 
                                                         return_estimator=estimator, 
                                                         n_jobs=-1))

    # If estimator is True, calculate threshold
    if estimator:
        decision_threshold = pd.Series([est.best_threshold_ for est in cv_results_tuned_model["estimator"]])
        cv_results_tuned_model['threshold'] = decision_threshold

    cv_results_tuned_model['algorithm'] = model_type

    return cv_results_tuned_model



def print_standard_error(values, label):
    '''
    Calculates the standard error
    :param values: List of values to calculate the
    standard deviation and mean
    :param label: What is the label for values

    :return mean_values: Mean of values
    :return std_values: Standard deviation from mean
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    mean_values = np.mean(values)
    
    # Standard deviation as error bars
    std_values = np.std(values)
    error = std_values/(np.sqrt(len(values)))

    print(f"Mean {label}: {mean_values:.3f} ± standard error {error}")

    return mean_values, std_values
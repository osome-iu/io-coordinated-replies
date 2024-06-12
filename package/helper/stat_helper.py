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
    #Variance
    # df_stat[f'var_{column_to_take}'] = df_stat[list_column].apply(
    #     lambda x: np.var(np.array(x)))
    
    # Coefficient of variation
    # df_stat[f'cof_{column_to_take}'] = df_stat[list_column].apply(
    #     lambda x: np.std(np.array(x)) / np.mean(np.array(x)) * 100)
    
    # # Mean Absolute Deviation
    # df_stat[f'mad_{column_to_take}'] = df_stat[list_column].apply(
    #     lambda x: np.mean(np.absolute(np.array(x) - np.mean(np.array(x))))
    # )
    
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



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def threshold_search(lr_probs, y_test):
    thresholds = np.linspace(0, 1, 100)

    best_threshold = 0
    best_score = 0

    # Iterate over the threshold values and evaluate the classifier at each threshold
    for threshold in thresholds:
        # Convert probabilities to binary predictions using the threshold
        y_pred = (lr_probs[:, 1] > threshold).astype(int)

        # Calculate the evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Calculate a single score to compare different thresholds
        score = 2 * precision * recall / (precision + recall)

        # Update the best threshold and score if a better threshold is found
        if score > best_score:
            best_threshold = threshold
            best_score = score

    # Print the best threshold and corresponding performance
    print("Best Threshold:", best_threshold)
    print("Best F1 Score:", best_score)
    
    
    return best_threshold


def run_model(df,
              columns_not_include=['list_age'],
              model_type='random', 
              pca=False,
              y_column = 'tweet_label',
              filename=None,
              just_f1=False,
              find_threshold=True,
              all_train=False
             ):
    '''
    Trains the model and prints the result
    :param df: Dataframe
    :param model_type: Type of model
    :param pca: Whether to do PCA or not
    :param columns_not_include: columns to not include
    '''
    print(f'\n **** {model_type} ****')
    
    import pickle

    model_filename =' k' #'user_classifier_without_pca_ran.sav'
        
    # print(df.columns)
    # name = columns_not_include[0].split('_')[1]
    
    columns_not_include.extend(
        ['poster_tweetid','tweet_label', 'replier_userid', 'replier_label'])
    
    columns_to_keep = list(set(df.columns) - set(columns_not_include))
    
    # if just_f1 == False:
    #     for x_col in columns_to_keep:
    #         print(x_col)

    X = df[columns_to_keep]
    y = df[y_column]
    
    # if just_f1 == True:
    # print('Columns: ', len(columns_to_keep))
    
    if 'mean_tensor' in columns_to_keep:
        t = df['mean_tensor'].tolist()
        t = torch.stack(t)
        t = t[:, :100]
        
        # print(len(columns_to_keep))
        columns_to_keep.remove('mean_tensor')
        
#         print(columns_to_keep)
        
#         print('after :', len(columns_to_keep))
#         print('columns_to_keep: ', columns_to_keep)
        z = df[columns_to_keep]
        # print(z)
        k = torch.tensor(z.values)
        X = torch.cat((t, k), dim=1)
    else:
        print(df[y_column].unique())
        X = df[columns_to_keep]
        
    #PCA 
    scaler = StandardScaler()
    # print(X)
    X = scaler.fit_transform(X)
    indices = df.index
    
    if pca == True:
        print('here')
        print(len(columns_to_keep))
        pca = PCA()

        # Fit the PCA object to the data and transform the data
        X = pca.fit_transform(X)
        print('After PCA shape ', X.shape)

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X,
                                                                                     y,
                                                                                     indices,
                                                        random_state=104, 
                                                        stratify=y,
                                                        test_size=0.20, 
                                                        shuffle=True)

    print('Xtrain: ', len(X_train))
    print('Xtrain shape: ', X_train.shape)
    print('Xtest: ', len(X_test))
    print('Ytrain: ', len(y_train))
    print('Ytest: ', len(y_test))

    if model_type == 'logistic':
        model = LogisticRegression(random_state=0)
    elif model_type == 'random':
        print('Running Random Forest')
        model = RandomForestClassifier(n_estimators=100, 
                                   random_state=42
                                  )
    elif model_type == 'ada':
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(n_estimators=100,
                                 algorithm="SAMME", random_state=0)
    elif model_type == 'tree':
        from sklearn import tree
        model = tree.DecisionTreeClassifier(random_state=0)
    elif model_type == 'naive':
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()

    if all_train == True:
        X_train = X
        y_train = y
        
    model.fit(X_train, y_train)
    
    if all_train == True:
        return model
            
            # pickle.dump(model, open(model_filename, 'wb'))
    
    # print(model.score(X_train, y_train))

    y_pred = model.predict(X_test)

    result = classification_report(y_test, y_pred, 
                                   labels=[0,1])
    prf_1 = precision_recall_fscore_support(y_test, 
                                y_pred,
                                average='binary',
                                pos_label=1
                               )    
    # print(result)
    prf_0 = precision_recall_fscore_support(y_test, 
                                y_pred,
                                average='binary',
                                pos_label=0
                               )    
    cm = confusion_matrix(y_test, y_pred)

    # print(cm)
    
    #Cross validation
    scoring = {'precision', 
               'recall',
               'f1',
               'roc_auc'
              }

    scores = cross_validate(model, X, y, scoring=scoring, cv=10)
    
    mean_score_f1 = round(scores['test_f1'].mean(), 2)
    std_score_f1 = round(scores['test_f1'].std(), 2)
    
    mean_score_precision = round(scores['test_precision'].mean(), 2)
    std_score_precision = round(scores['test_precision'].std(), 2)
    
    mean_score_recall = round(scores['test_recall'].mean(), 2)
    std_score_recall = round(scores['test_recall'].std(), 2)
    
    mean_score_auc = round(scores['test_roc_auc'].mean(), 2)
    std_score_auc = round(scores['test_roc_auc'].std(), 2)
    
    print(f'Cross validation: mean {mean_score_f1} f1 with a standard deviation of {std_score_f1}')
    
    print(f'Cross validation: mean {mean_score_precision} precision with a standard deviation of {std_score_precision}')
    
    print(f'Cross validation: mean {mean_score_recall} recall with a standard deviation of {std_score_recall}')
    
    print(f'Cross validation: mean {mean_score_auc} auc with standard deviation of {std_score_auc}')
    
    # auc_pr = round(average_precision_score(y_test, lr_probs[:,1]), 2)
    
    final_score = {
            'mean_f1': mean_score_f1,
            'mean_precision': mean_score_precision,
            'mean_recall': mean_score_recall,
            'mean_auc': mean_score_auc,
            # 'auc_pr': auc_pr
    }
    
    #feature importance
#     flag_imprt = (model_type == 'random') and (pca == False) and (feat_importance == True)
    
#     if flag_imprt:
#         model, X_test, y_test,
        # df_imp = pd.DataFrame({
        #     'Feature': columns_to_keep, 
        #     'Importance': model.feature_importances_})
        # df_imp = df_imp.sort_values('Importance', 
        #                                  ascending=False).set_index('Feature')
        # print(df_imp.head(10))
        # df_imp.to_pickle('feat_importance_without_pca.pkl.gz')

    #ROC curve
    lr_probs = model.predict_proba(X_test)
    
    fpr, tpr, thresholds = roc_curve(y_test, lr_probs[:, 1])
    
    # Compute the AUC score
    roc_auc = auc(fpr, tpr)
    
    # Plot the ROC curve
    # fig, ax = plt.subplots(figsize=(8,8))
    # ax.plot(fpr, tpr, color='darkorange', 
    #          lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    # ax.plot([0, 1], [0, 1], color='navy', 
    #          lw=2, linestyle='--')
    # ax.set_xlim([0.0, 1.0])
    # ax.set_ylim([0.0, 1.05])
    # ax.set_xlabel('False Positive Rate')
    # ax.set_ylabel('True Positive Rate')
    # ax.set_title(f'Receiver Operating Characteristic (ROC) Curve ')
    # ax.legend(loc="lower right")
    # plt.show()

    if filename != None:
        fig.savefig(f'{filename}')
    
    from sklearn.metrics import precision_recall_curve

    # y_true and y_scores are the true labels and predicted scores, respectively
    precision, recall, thresholds = precision_recall_curve(y_test,
                                                        lr_probs[:, 1])

    # plot the precision-recall curve
    # plt.plot(recall, precision)
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve')
    # plt.show()
    
    df_pred = df.loc[indices_test]
    # df_pred['pred'] = y_pred
    
    if find_threshold == True:
        threshold = threshold_search(lr_probs, y_test)
        
        return model, df_pred, roc_auc, prf_1, prf_0, mean_score_f1, std_score_f1, threshold, final_score
    
    return model, df_pred, roc_auc, prf_1, prf_0, mean_score_f1, std_score_f1, final_score


def KS_test(data1, data2):
    '''
    Tes the KS test for data1 and data2
    '''
    statistic, pvalue = ks_2samp(data1, 
                                 data2)


    print('KS test statistic:', statistic)
    print('p-value:', pvalue)
    
    
def KNN(df,
        columns_not_include=['list_age'],
        y_column = 'replier_label',
        filename='knearest_neighbor_time_diff.png',
        metric='cosine',
        cross_validation_file=None
       ):
    
    '''
    Performs kNN classification
    :param df: Dataframe
    :param columns_not_include: columns to not include in classification
    :param y_column: target column
    :param filename: filename of roc curve plot
    :param cross_validation_file: file to save the cross validation result
    '''
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    knnClassifier = KNeighborsClassifier(n_neighbors = 5, 
                                         metric = metric, 
                                        )
                    
    columns_not_include.extend(
        ['poster_tweetid','tweet_label', 'replier_userid', 'replier_label'])
    
    columns_to_keep = list(set(df.columns) - set(columns_not_include))
    
    print(df[y_column].unique())
    
    X = df[columns_to_keep]

    #PCA 
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    indices = df.index
    
    y = df[y_column]

    print('here')
    pca = PCA()

    # Fit the PCA object to the data and transform the data
    X = pca.fit_transform(X)
    print('After PCA shape ', X.shape)

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X,
                                                                                     y,
                                                                                     indices,
                                                        # random_state=104, 
                                                        stratify=y,
                                                        test_size=0.20, 
                                                        shuffle=True)

    print('Xtrain: ', len(X_train))
    print('Xtrain shape: ', X_train.shape)
    print('Xtest: ', len(X_test))
    print('Ytrain: ', len(y_train))
    print('Ytest: ', len(y_test))


    knnClassifier.fit(X_train, y_train)
    y_pred = knnClassifier.predict(X_test)

    result = classification_report(y_test,
                                   y_pred, 
                                   labels=[0,1])

    print(result)

    #Stratified Cross validation
    def stratified_cross_validation(classifier=knnClassifier, 
                                    cross_validation_file=cross_validation_file,
                                    X=X_train,
                                    y=y_train,
                                    k=5, 
                                    ):
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        # Perform stratified cross-validation
        skf = StratifiedKFold(n_splits=k)
        for train_index, test_index in skf.split(X, y):
            # Split data into train and test sets based on the cross-validation split indices
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Fit the classifier on the training data
            classifier.fit(X_train, y_train)

            # Make predictions on the test data
            y_pred = classifier.predict(X_test)

            # Calculate evaluation metrics for the fold and store the results
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred))
            recall_scores.append(recall_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))

        # Compute the mean and standard deviation of the evaluation metrics across all folds
        mean_accuracy = np.mean(accuracy_scores)
        std_accuracy = np.std(accuracy_scores)
        mean_precision = np.mean(precision_scores)
        std_precision = np.std(precision_scores)
        mean_recall = np.mean(recall_scores)
        std_recall = np.std(recall_scores)
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)

        # Print the mean and standard deviation of the evaluation metrics
        print("Mean Accuracy: {:.3f} (+/- {:.3f})".format(mean_accuracy, std_accuracy))
        print("Mean Precision: {:.3f} (+/- {:.3f})".format(mean_precision, std_precision))
        print("Mean Recall: {:.3f} (+/- {:.3f})".format(mean_recall, std_recall))
        print("Mean F1-Score: {:.3f} (+/- {:.3f})".format(mean_f1, std_f1))
        
        with open(f'{cross_validation_file}', 'w') as file:
            file.write("Mean Accuracy: {:.3f} (+/- {:.3f}) \n".format(mean_accuracy, std_accuracy))
            file.write("Mean Precision: {:.3f} (+/- {:.3f}) \n".format(mean_precision, std_precision))
            file.write("Mean Recall: {:.3f} (+/- {:.3f}) \n".format(mean_recall, std_recall))
            file.write("Mean F1-Score: {:.3f} (+/- {:.3f}) \n".format(mean_f1, std_f1))

    stratified_cross_validation()
    
    lr_probs = knnClassifier.predict_proba(X_test) 


    fpr, tpr, thresholds = roc_curve(y_test, lr_probs[:, 1])

    # Compute the AUC score
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(fpr, tpr, color='darkorange', 
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', 
             lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve \n Knearest neighbor \n Reply time diff')
    ax.legend(loc="lower right")
    plt.show()

    config = config_hp.config()
    plot_path = config['PLOTS']
    reply_plot_path = plot_path['reply_plot_path']

    fig.savefig(f'{reply_plot_path}/{filename}')
    
    
    
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
                              on=column_to_groupby)
    
    grps = df.groupby([column_to_groupby, 
                       label]).groups.keys()
    df_grps = pd.DataFrame(data=grps, columns=[column_to_groupby, 
                                               label])
    
    df_group = df_group.merge(df_grps[[column_to_groupby, 
                                       label]],
                              on=column_to_groupby)
    # df_group = df_group.fillna(0)
    
    return df_group




def run_imbalanced_model(
    df,
    columns_not_include=[],
    y_column = 'replier_label',
    model_type='random',
    threshold_search=False,
    k=10
):
    
    '''
    Performs random forest classification for 
    :param df: Dataframe
    :param columns_not_include: columns to not include in classification
    :param y_column: target column
    :param roc_filename: filename of roc curve plot
    :param cross_validation_file: file to save the cross validation result
    '''
    
    import pickle
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    from sklearn.metrics import confusion_matrix

   
    columns_not_include.extend(
        ['poster_tweetid','tweet_label', 'replier_userid', 'replier_label']
    )
    
    columns_to_keep = list(set(df.columns) - set(columns_not_include))
    
    X = df[columns_to_keep]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    indices = df.index
    
    y = df[y_column]

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, indices,
        random_state=104, 
        stratify=y,
        test_size=0.20, 
        shuffle=True
    )

    print('Xtrain: ', len(X_train))
    print('Xtrain shape: ', X_train.shape)
    print('Xtest: ', len(X_test))
    print('Ytrain: ', len(y_train))
    print('Ytest: ', len(y_test))

    
    if model_type == 'logistic':
        model = LogisticRegression(random_state=0)
    if model_type == 'random':
        model = RandomForestClassifier(n_estimators=100, 
                                       random_state=42
                                      )
        
    scoring = {'precision', 
               'recall',
               'f1',
               'roc_auc'
              }
    
    skf = StratifiedKFold(n_splits=k)

    scores = cross_validate(model, X, y, scoring=scoring, cv=skf)
    
    mean_score_f1 = round(scores['test_f1'].mean(), 2)
    std_score_f1 = round(scores['test_f1'].std(), 2)
    
    mean_score_precision = round(scores['test_precision'].mean(), 2)
    std_score_precision = round(scores['test_precision'].std(), 2)
    
    mean_score_recall = round(scores['test_recall'].mean(), 2)
    std_score_recall = round(scores['test_recall'].std(), 2)
    
    mean_score_auc = round(scores['test_roc_auc'].mean(), 2)
    std_score_auc = round(scores['test_roc_auc'].std(), 2)
    
    print(f'Cross validation: mean {mean_score_f1} f1 with a standard deviation of {std_score_f1}')
    
    print(f'Cross validation: mean {mean_score_precision} precision with a standard deviation of {std_score_precision}')
    
    print(f'Cross validation: mean {mean_score_recall} recall with a standard deviation of {std_score_recall}')
    
    print(f'Cross validation: mean {mean_score_auc} auc with standard deviation of {std_score_auc}')
    
    final_score = {
            'mean_f1': mean_score_f1,
            'mean_precision': mean_score_precision,
            'mean_recall': mean_score_recall,
            'mean_auc': mean_score_auc
    }
    
    model.fit(X_train, y_train)
    
    lr_probs = model.predict_proba(X_test) 

    fpr, tpr, thresholds = roc_curve(y_test, lr_probs[:, 1])
    
    if threshold_search == True:
        thresholds = np.linspace(0, 1, 100)

        best_threshold = 0
        best_score = 0

        # Iterate over the threshold values and evaluate the classifier at each threshold
        for threshold in thresholds:
            # Convert probabilities to binary predictions using the threshold
            y_pred = (lr_probs[:, 1] > threshold).astype(int)

            # Calculate the evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Calculate a single score to compare different thresholds
            score = 2 * precision * recall / (precision + recall)

            # Update the best threshold and score if a better threshold is found
            if score > best_score:
                best_threshold = threshold
                best_score = score

        # Print the best threshold and corresponding performance
        print("Best Threshold:", best_threshold)
        print("Best F1 Score:", best_score)

    
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_test,
                                                        lr_probs[:, 1])

    from sklearn.metrics import auc, average_precision_score

    auc_pr = round(average_precision_score(y_test, lr_probs[:,1]), 2)
    
    print(f'AUC-PR: {auc_pr}')
    
    if threshold_search == True:
        return final_score, threshold, auc_pr
    
    return final_score, auc_pr



def run_model_with_best_threshold(df,
              columns_not_include=['list_age'],
              model_type='random', 
              y_column = 'tweet_label',
              filename=None,
             ):
    '''
    Trains the model and prints the result
    :param df: Dataframe
    :param model_type: Type of model
    :param pca: Whether to do PCA or not
    :param columns_not_include: columns to not include
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
        model = RandomForestClassifier(n_estimators=100, 
                                   random_state=42
                                  )
    elif model_type == 'ada':
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(n_estimators=100,
                                 algorithm="SAMME", random_state=0)
    elif model_type == 'tree':
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
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
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
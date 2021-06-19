import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE


def data_imbalance_techniques():

    main_data = pd.read_csv('CSVFiles/ms-data.csv', index_col=0)
    clean_data = main_data[
        ['t_ch', 't_dev', 'l_core_and_top', 'l_core_and_occ', 'l_peripheral_and_top', 'l_peripheral_and_occ',
         'r_core_and_top', 'r_core_and_occ', 'r_peripheral_and_top', 'r_peripheral_and_occ', 'has_conflict']]

    # Shuffle Data
    clean_data = clean_data.sample(frac=1)

    y = clean_data.has_conflict.copy()
    X = clean_data.drop(['has_conflict'], axis=1)

    # Splitting dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)
    X_train.shape, X_test.shape

    # Combining Train data
    data_train = X_train
    data_train['has_conflict'] = y_train
    # print(data_train)
    # Combining Test data
    data_test = X_test
    data_test['has_conflict'] = y_test

    under_sampling(data_train)
    over_sampling(data_train)
    over_under_sampling(data_train)
    smote(data_train)
    borderline_smote(data_train)
    svm_smote(data_train)
    adasyn(data_train)

    # Write CSV
    data_test.to_csv('CSVFiles/data_test.csv')


def numpyTodf(X, y):
    y_df = pd.DataFrame(y, columns=['has_conflict'])
    main_df = pd.DataFrame(X, columns=['t_ch', 't_dev', 'l_core_and_top', 'l_core_and_occ', 'l_peripheral_and_top',
                                       'l_peripheral_and_occ', 'r_core_and_top', 'r_core_and_occ',
                                       'r_peripheral_and_top', 'r_peripheral_and_occ'])
    main_df['has_conflict'] = y_df
    return main_df

    # colors = {0: 'black', 1: 'red'}
    # fig = plt.Figure(figsize=(10, 7))
    # plt.scatter(data_train['t_ch'], data_train['t_dev'], c=data_train['has_conflict'].map(colors))
    # sns.despine()


def under_sampling(data_train):
    y = data_train.has_conflict.copy()
    X = data_train.drop(['has_conflict'], axis=1)

    # print('Before Random Under Sampling: ', Counter(y))
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_under, y_under = undersample.fit_resample(X, y)
    # print('After Random Under Sampling: ', Counter(y_under))

    # Write CSV
    main_df = numpyTodf(X_under, y_under)
    # print(main_df)
    main_df.to_csv('CSVFiles/RandomUnderSampling_datatrain.csv')


def over_sampling(data_train):
    y = data_train.has_conflict.copy()
    X = data_train.drop(['has_conflict'], axis=1)

    # print('Before Random Over Sampling: ', Counter(y))
    oversample = RandomOverSampler(sampling_strategy='minority')
    X_over, y_over = oversample.fit_resample(X, y)
    # print('After Random Over Sampling: ', Counter(y_over))

    # Write CSV
    main_df = numpyTodf(X_over, y_over)
    # print(main_df)
    main_df.to_csv('CSVFiles/RandomOverSampling_datatrain.csv')

    # Visualize the plot

    # colors = ['black' if v == 0 else 'red' for v in y_over]
    # fig = plt.Figure(figsize=(10,7))
    # plt.scatter(X_over[:,0], X_over[:,1], c=colors)
    # plt.title('Random Over Sampling')
    # # sns.despine()
    # plt.savefig('Random Over Sampling')
    # plt.show()


def over_under_sampling(data_train):
    y = data_train.has_conflict.copy()
    X = data_train.drop(['has_conflict'], axis=1)

    # print('Before Sampling: ', Counter(y))
    over = RandomOverSampler(sampling_strategy=0.1)
    X, y = over.fit_resample(X, y)

    under = RandomUnderSampler(sampling_strategy=0.5)
    X, y = under.fit_resample(X, y)
    # print('After Sampling: ', Counter(y))

    # Write CSV
    main_df = numpyTodf(X, y)
    # print(main_df)
    main_df.to_csv('CSVFiles/RandomOver&UnderSampling_datatrain.csv')


def smote(data_train):
    y = data_train.has_conflict.copy()
    X = data_train.drop(['has_conflict'], axis=1)

    # print('Before SMOTE: ', Counter(y))
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)
    # print('After SMOTE: ', Counter(y))

    # Write CSV
    main_df = numpyTodf(X, y)
    # print(main_df)
    main_df.to_csv('CSVFiles/SMOTE_datatrain.csv')


def borderline_smote(data_train):
    y = data_train.has_conflict.copy()
    X = data_train.drop(['has_conflict'], axis=1)

    # print('Before Borderline SMOTE: ', Counter(y))
    oversample = BorderlineSMOTE()
    X, y = oversample.fit_resample(X, y)
    # print('After Borderline SMOTE: ', Counter(y))

    # Write CSV
    main_df = numpyTodf(X, y)
    # print(main_df)
    main_df.to_csv('CSVFiles/BorderlineSMOTE_datatrain.csv')


def svm_smote(data_train):
    y = data_train.has_conflict.copy()
    X = data_train.drop(['has_conflict'], axis=1)

    # print('Before SVM SMOTE: ', Counter(y))
    oversample = SVMSMOTE()
    X, y = oversample.fit_resample(X, y)
    # print('After SVM SMOTE: ', Counter(y))

    # Write CSV
    main_df = numpyTodf(X, y)
    # print(main_df)
    main_df.to_csv('CSVFiles/SVMSMOTE_datatrain.csv')


def adasyn(data_train):
    y = data_train.has_conflict.copy()
    X = data_train.drop(['has_conflict'], axis=1)

    # print('Before ADASYN: ', Counter(y))
    oversample = ADASYN()
    X, y = oversample.fit_resample(X, y)
    # print('After ADASYN: ', Counter(y))

    # Write CSV
    main_df = numpyTodf(X, y)
    # print(main_df)
    main_df.to_csv('CSVFiles/ADASYN_datatrain.csv')

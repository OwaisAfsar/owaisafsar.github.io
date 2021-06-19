from imblearn.over_sampling import ADASYN, SMOTE
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import itertools
import statsmodels.api as sm
from numpy import mean, var
import scipy.stats as stats
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, \
    confusion_matrix, classification_report, roc_auc_score

column_names = ["Index", "Classifier", "Model", "Sampling_Method", "Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
csv_data = pd.DataFrame(columns=column_names)
report_data = []
pca_report_data = []


def PCA_variables(df):
    main_df = df[['t_ch', 't_dev', 'l_core_and_top', 'l_peripheral_and_top', 'l_peripheral_and_occ', 'has_conflict']]
    # print(main_df)
    return main_df


def get_hyperparameter_values(_classifier_name):
    hyperparameter_data = ''

    if _classifier_name == 'MLP':
        mlp_data = {
            'Sampling_Method': ['Under', 'Over', 'Both', 'Smote', 'BorderlineSmote', 'SVMSmote', 'Adasyn', 'Rose'],
            'Learning_Rate': [0.001, 0.001, 0.0004, 0.001, 0.0005, 0.001, 0.0005, 0.0002],
            'Dropout_Rate': [0.045, 0.044, 0.043, 0.01, 0.036, 0.057, 0.034, 0.021],
            'Neuron_Dense': [206, 104, 261, 74, 59, 262, 189, 273],
            'Hidden_Layers': [3, 5, 7, 5, 6, 5, 5, 4],
            'Epochs': [73, 47, 69, 38, 33, 56, 43, 71],
            'Batch_Size': [44, 27, 53, 53, 51, 45, 49, 39],
            'Activation': ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu']}
        hyperparameter_data = pd.DataFrame(mlp_data)
    elif _classifier_name == 'Sequential':
        seq_data = {
            'Sampling_Method': ['Under', 'Over', 'Both', 'Smote', 'BorderlineSmote', 'SVMSmote', 'Adasyn', 'Rose'],
            'Learning_Rate': [0.001, 0.0001, 0.002, 0.001, 0.001, 0.001, 0.002, 0.0001],
            'Dropout_Rate': [0.045, 0.011, 0.098, 0.038, 0.075, 0.269, 0.043, 0.023],
            'Neuron_Dense': [191, 234, 35, 32, 100, 242, 93, 225],
            'Hidden_Layers': [3, 5, 7, 5, 6, 5, 5, 4],
            'Epochs': [73, 83, 83, 52, 45, 100, 69, 86],
            'Batch_Size': [20, 14, 50, 62, 15, 51, 35, 49],
            'Activation': ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu']}
        hyperparameter_data = pd.DataFrame(seq_data)
    elif _classifier_name == 'LSTM':
        lstm_mlp_data = {
            'Sampling_Method': ['Under', 'Over', 'Both', 'Smote', 'BorderlineSmote', 'SVMSmote', 'Adasyn', 'Rose'],
            'Learning_Rate': [0.001, 0.001, 0.0003, 0.001, 0.001, 0.001, 0.003, 0.0005],
            'Dropout_Rate': [0.129, 0.037, 0.178, 0.122, 0.017, 0.036, 0.014, 0.013],
            'LSTM_Units': [6, 3, 10, 4, 4, 9, 6, 7],
            'Neuron_Dense': [61, 285, 224, 180, 59, 222, 130, 63],
            'Hidden_Layers': [3, 5, 7, 5, 6, 5, 5, 6],
            'Epochs': [13, 16, 46, 12, 7, 7, 14, 31],
            'Batch_Size': [8, 28, 11, 49, 32, 33, 46, 40],
            'Activation': ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu']}
        hyperparameter_data = pd.DataFrame(lstm_mlp_data)
    elif _classifier_name == 'Sequential_LSTM':
        lstm_seq_data = {
            'Sampling_Method': ['Under', 'Over', 'Both', 'Smote', 'BorderlineSmote', 'SVMSmote', 'Adasyn', 'Rose'],
            'Learning_Rate': [0.0008, 0.0003, 0.002, 0.003, 0.0005, 0.0003, 0.0012, 0.0003],
            'Dropout_Rate': [0.04, 0.116, 0.137, 0.073, 0.188, 0.01, 0.06, 0.095],
            'LSTM_Units': [4, 10, 7, 6, 4, 6, 8, 3],
            'Neuron_Dense': [183, 300, 75, 218, 110, 126, 169, 280],
            'Hidden_Layers': [2, 3, 6, 4, 6, 1, 1, 1],
            'Epochs': [72, 67, 33, 22, 60, 56, 59, 75],
            'Batch_Size': [53, 61, 56, 33, 17, 33, 36, 50],
            'Activation': ['relu', 'tanh', 'relu', 'tanh', 'relu', 'relu', 'relu', 'sigmoid']}
        hyperparameter_data = pd.DataFrame(lstm_seq_data)
    elif _classifier_name == 'GRU':
        gru_mlp_data = {
            'Sampling_Method': ['Under', 'Over', 'Both', 'Smote', 'BorderlineSmote', 'SVMSmote', 'Adasyn', 'Rose'],
            'Learning_Rate': [0.003, 0.0004, 0.003, 0.001, 0.002, 0.002, 0.0005, 0.0002],
            'Dropout_Rate': [0.017, 0.065, 0.300, 0.135, 0.011, 0.065, 0.182, 0.213],
            'GRU_Units': [8, 5, 7, 5, 8, 7, 9, 8],
            'Neuron_Dense': [70, 203, 228, 15, 113, 155, 106, 252],
            'Hidden_Layers': [3, 5, 7, 5, 6, 5, 5, 6],
            'Epochs': [46, 72, 53, 38, 90, 71, 38, 54],
            'Batch_Size': [8, 53, 32, 44, 57, 35, 51, 53],
            'Activation': ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu']}
        hyperparameter_data = pd.DataFrame(gru_mlp_data)
    elif _classifier_name == 'Sequential_GRU':
        gru_seq_data = {
            'Sampling_Method': ['Under', 'Over', 'Both', 'Smote', 'BorderlineSmote', 'SVMSmote', 'Adasyn', 'Rose'],
            'Learning_Rate': [0.0003, 0.0003, 0.0005, 0.0006, 0.0009, 0.0016, 0.0005, 0.0009],
            'Dropout_Rate': [0.128, 0.066, 0.01, 0.117, 0.048, 0.157, 0.079, 0.017],
            'GRU_Units': [8, 7, 6, 8, 8, 9, 8, 5],
            'Neuron_Dense': [138, 119, 264, 112, 170, 175, 93, 173],
            'Hidden_Layers': [3, 1, 5, 6, 2, 3, 7, 1],
            'Epochs': [75, 20, 73, 85, 77, 59, 97, 15],
            'Batch_Size': [33, 17, 47, 37, 51, 25, 41, 27],
            'Activation': ['tanh', 'sigmoid', 'tanh', 'tanh', 'tanh', 'sigmoid', 'sigmoid', 'relu']}
        hyperparameter_data = pd.DataFrame(gru_seq_data)

    return hyperparameter_data


def classificationMetrics(classifier, model, x_test, y_test, sampling_method, method):
    y_test_pred = model.predict(x_test)
    y_test_pred = y_test_pred.reshape(y_test_pred.shape[0], )

    yhat_probs = model.predict(x_test, verbose=0)
    yhat_probs = yhat_probs[:, 0]

    accuracy = round(accuracy_score(y_test, y_test_pred.round()), 2)
    kappa = cohen_kappa_score(y_test, y_test_pred.round())
    auc = round(roc_auc_score(y_test, yhat_probs), 2)
    matrix = confusion_matrix(y_test, y_test_pred.round())

    report = classification_report(y_test, y_test_pred.round())
    classification_report_csv(classifier, method, sampling_method, accuracy, report, auc)


def classification_report_csv(classifier, model, sampling_method, accuracy, report, auc):
    lines = report.split('\n')
    # print(type(lines))
    row_count = 0
    for line in lines[2:-3]:
        # print(row_count)
        if row_count < 2:
            row = []
            row = line.split(' ')
            row = list(filter(None, row))
            if len(row) != 0:
                # print(classifier, model, sampling_method, row[0], accuracy, row[1], row[2], row[3], auc)
                values = [classifier, model, sampling_method, row[0], accuracy, row[1], row[2], row[3], auc]
                report_data.append(values)
                row_count = row_count + 1


def classification_report_csv_df():
    dataframe = pd.DataFrame(data=report_data, columns=column_names)
    print(dataframe)
    dataframe.to_csv('CSVFiles/classification_report.csv', index=False)


def pca_classificationMetrics(classifier, model, x_test, y_test, sampling_method, method):
    y_test_pred = model.predict(x_test)
    y_test_pred = y_test_pred.reshape(y_test_pred.shape[0], )

    yhat_probs = model.predict(x_test, verbose=0)
    yhat_probs = yhat_probs[:, 0]

    accuracy = round(accuracy_score(y_test, y_test_pred.round()), 2)
    kappa = cohen_kappa_score(y_test, y_test_pred.round())
    auc = round(roc_auc_score(y_test, yhat_probs), 2)
    matrix = confusion_matrix(y_test, y_test_pred.round())

    report = classification_report(y_test, y_test_pred.round())
    pca_classification_report_csv(classifier, method, sampling_method, accuracy, report, auc)


def pca_classification_report_csv(classifier, model, sampling_method, accuracy, report, auc):
    lines = report.split('\n')
    # print(type(lines))
    row_count = 0
    for line in lines[2:-3]:
        # print(row_count)
        if row_count < 2:
            row = []
            row = line.split(' ')
            row = list(filter(None, row))
            if len(row) != 0:
                # print(classifier, model, sampling_method, row[0], accuracy, row[1], row[2], row[3], auc)
                values = [classifier, model, sampling_method, row[0], accuracy, row[1], row[2], row[3], auc]
                pca_report_data.append(values)
                row_count = row_count + 1


def pca_classification_report_csv_df():
    dataframe = pd.DataFrame(data=pca_report_data, columns=column_names)
    print(dataframe)
    dataframe.to_csv('CSVFiles/pca_classification_report.csv', index=False)


def standardized_data(X, y):
    sc = StandardScaler()
    X = sc.fit_transform(X)
    y = sc.transform(y)

    return X, y


def dataVisualization(_dataframe, _feature):
    print(_dataframe[_feature].value_counts())

    if (_feature == 'Experience'):
        sns.barplot(x="Experience", y="conflt", data=_dataframe)

        Beginnercnflt = _dataframe["conflt"][_dataframe["Experience"] == 'Beginner'].value_counts(normalize=True)[
                            1] * 100
        intermediatecnflt = \
            _dataframe["conflt"][_dataframe["Experience"] == 'Intermediate'].value_counts(normalize=True)[1] * 100
        advancecnflt = _dataframe["conflt"][_dataframe["Experience"] == 'Advanced'].value_counts(normalize=True)[
                           1] * 100

        print("Percentage of Beginner caused conflicts:", round(Beginnercnflt, 2))
        print("Percentage of Intermediate caused conflicts:", round(intermediatecnflt, 2))
        print("Percentage of Advanced caused conflicts:", round(advancecnflt, 2))
        plt.savefig('Experience.png')

        bgcnflt = (3573 / 9117) * 100
        intcnflt = (868 / 9117) * 100
        advcnflt = (4676 / 9117) * 100

        expcnflt = pd.DataFrame(
            {'Experience': ['Beginner', 'Advanced', 'Intermediate'], 'Conflict': [bgcnflt, advcnflt, intcnflt]})
        sns.barplot(x="Experience", y='Conflict', data=expcnflt)

        print("Beginner caused conflicts in total conflicts:", round(bgcnflt, 2), "%")
        print("Advanced caused conflicts in total conflicts:", round(advcnflt, 2), "%")
        print("Intermediate caused conflicts in total conflicts:", round(intcnflt, 2), "%")
        plt.savefig('Experience-Conflicts.png')


def setDataset(_dataframe, _target):
    dataframe = _dataframe
    X = None
    y = None

    if 'cont_id' in dataframe:
        dataframe = dataframe.drop(['cont_id'], axis=1)

    if _target == 'conflt':
        # print('conflt')
        dataframe = dataframe.drop(['ms_confl'], axis=1)
        y = dataframe.conflt.copy()
        X = dataframe.drop(['conflt'], axis=1)
    elif _target == 'ms_confl':
        # print('ms_confl')
        dataframe = dataframe.drop(['conflt'], axis=1)
        y = dataframe.ms_confl.copy()
        X = dataframe.drop(['ms_confl'], axis=1)
    return X, y


def smote(_X, _y):
    smt = SMOTE()
    X, y = smt.fit_sample(_X, _y)
    return X, y


def ad_smote(_X, _y):
    ada = ADASYN(random_state=130)
    X, y = ada.fit_sample(_X, _y)
    return X, y


# function to calculate Cohen's d for independent samples
def cohen_d(d1, d2):
    n1, n2 = len(d1), len(d2)
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    u1, u2 = mean(d1), mean(d2)
    return (u1 - u2) / s


def logitSummary(_X, _y):
    logit_model = sm.Logit(_y, _X)
    result = logit_model.fit()
    print(result.summary())


def listwise_corr_pvalues(df, method):
    df = df.dropna(how='any')._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pd.set_option('display.max_rows', None, 'display.max_columns', None)
    rvalues = dfcols.transpose().join(dfcols, how='outer')
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    length = str(len(df))

    if method == None:
        test = stats.pearsonr
        test_name = "Pearson"
    elif method == "spearman":
        test = stats.spearmanr
        test_name = "Spearman Rank"
    elif method == "kendall":
        test = stats.kendalltau
        test_name = "Kendall's Tau-b"

    for r in df.columns:
        for c in df.columns:
            rvalues[r][c] = round(test(df[r], df[c])[0], 4)

    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = format(test(df[r], df[c])[1], '.4f')

    # print("Correlation test conducted using list-wise deletion",
    #       "\n",
    #       "Total observations used: ", length, "\n", "\n",
    #       f"{test_name} Correlation Values", "\n", rvalues, "\n",
    #       "Significant Levels", "\n", pvalues)


def pairwise_corr_pvalues(df, method=None):
    correlations = {}
    pvalues = {}
    length = {}
    columns = df.columns.tolist()

    if method == None:
        test = stats.pearsonr
        test_name = "Pearson"
    elif method == "spearman":
        test = stats.spearmanr
        test_name = "Spearman Rank"
    elif method == "kendall":
        test = stats.kendalltau
        test_name = "Kendall's Tau-b"

    for col1, col2 in itertools.combinations(columns, 2):
        sub = df[[col1, col2]].dropna(how="any")
        correlations[col1 + " " + "&" + " " + col2] = format(test(sub.loc[:, col1], sub.loc[:, col2])[0], '.4f')
        pvalues[col1 + " " + "&" + " " + col2] = format(test(sub.loc[:, col1], sub.loc[:, col2])[1], '.4f')
        length[col1 + " " + "&" + " " + col2] = len(df[[col1, col2]].dropna(how="any"))

    corrs = pd.DataFrame.from_dict(correlations, orient="index")
    corrs.columns = ["r value"]
    pvals = pd.DataFrame.from_dict(pvalues, orient="index")
    pvals.columns = ["p-value"]

    l = pd.DataFrame.from_dict(length, orient="index")
    l.columns = ["N"]

    results = corrs.join([pvals, l])
    # print(f"{test_name} correlation", "\n", results)


def ChiSquareTest():
    columnnames = ["Analysis", "Chi-squared", "dof", "P-value"]
    maindata = []

    first_analysis = list(chiSquareTest_twoVariables(45297, 3290, 60609, 3409))
    first_values = ['First', first_analysis[0], first_analysis[1], first_analysis[2]]
    maindata.append(first_values)

    second_analysis = list(chiSquareTest_twoVariables(78740, 3950, 30003, 2800))
    second_values = ['Second', second_analysis[0], second_analysis[1], second_analysis[2]]
    maindata.append(second_values)

    third_analysis = list(chiSquareTest_fourVariables(40249, 2912, 15069, 1569, 52392, 2674, 26788, 2617))
    third_values = ['Second', third_analysis[0], third_analysis[1], third_analysis[2]]
    maindata.append(third_values)

    df = pd.DataFrame(maindata, columns=columnnames)
    df.to_csv('CSVFiles/ChiSquareTest.csv')


def chiSquareTest_twoVariables(ms1, cms1, ms2, cms2):
    data_array = np.array([[ms1, cms1], [ms2, cms2]])
    df = pd.DataFrame(data_array, columns=["No Conflicts", "Conflicts"])
    df.index = ["Top", "Occ"]
    value = calculation_chi_square(df)
    return value


def chiSquareTest_fourVariables(ms1, cms1, ms2, cms2, ms3, cms3, ms4, cms4):
    data_array = np.array([[ms1, cms1], [ms2, cms2], [ms3, cms3], [ms4, cms4]])
    df = pd.DataFrame(data_array, columns=["No Conflicts", "Conflicts"])
    df.index = ["Top", "TopOcc", "OccTop", "Occ"]
    value = calculation_chi_square(df)
    return value


def calculation_chi_square(df):
    statistics, p_value, dof, exp = chi2_contingency(df, correction=False)
    # print("Chi-squared: " + str(statistics))
    # print("Degree of Freedom: ", dof)
    # print("P-value: " + str(p_value))
    return statistics, dof, p_value

import tensorflow as tf
import matplotlib.pyplot as plt
import Utils
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
from tensorflow.keras import layers


def get_keras_GRU(gru_units,
                  num_hidden_layers,
                  neuron_dense,
                  dropout_rate,
                  activation,
                  x_train):
    inputs = tf.keras.Input(shape=(1, 5))
    x = layers.GRU(units=gru_units)(inputs)

    for i in range(num_hidden_layers):
        x = layers.Dense(neuron_dense,
                         activation=activation)(x)
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def KerasGRUModel(learning_rate, dropout_rate, gru_units, neuron_dense, hidden_layers, epochs,
                  batch_size, activation, x_train, y_train):
    keras_model = get_keras_GRU(gru_units, hidden_layers, neuron_dense, dropout_rate, activation, x_train)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Specify the training configuration.
    keras_model.compile(optimizer=optimizer,
                        loss=tf.keras.losses.BinaryCrossentropy(),
                        metrics=['accuracy'])

    data = x_train
    labels = y_train
    res = keras_model.fit(data, labels, epochs=epochs, batch_size=batch_size)
    return keras_model


def get_sequential_GRU(gru_units,
                       num_hidden_layers,
                       neuron_dense,
                       dropout_rate,
                       activation,
                       x_train):
    inputs = tf.keras.Input(shape=(1, 5))
    x = tf.keras.Sequential()(inputs)
    x = layers.GRU(units=gru_units)(x)

    for i in range(num_hidden_layers):
        x = layers.Dense(neuron_dense,
                         activation=activation)(x)
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def SequentialGRUModel(learning_rate, dropout_rate, gru_units, neuron_dense, hidden_layers, epochs,
                       batch_size, activation, x_train, y_train):
    sequential_model = get_sequential_GRU(gru_units, hidden_layers, neuron_dense, dropout_rate, activation, x_train)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    sequential_model.compile(optimizer=optimizer,
                             loss=tf.keras.losses.BinaryCrossentropy(),
                             metrics=['accuracy'])

    data = x_train
    labels = y_train
    res = sequential_model.fit(data, labels, epochs=epochs, batch_size=batch_size)
    return sequential_model


def GRU_Classifer():
    data_models = ['Under', 'Over', 'Both', 'Smote', 'BorderlineSmote', 'SVMSmote', 'Adasyn', 'Rose']
    # data_models = ['Under']
    hyperparameter_value = ['MLP', 'Sequential']
    # hyperparameter_value = ['MLP']

    for value in hyperparameter_value:
        hyperparameter_df = Utils.get_hyperparameter_values('Sequential_GRU')

        for model in data_models:
            if model == 'Under':
                data_train = pd.read_csv('CSVFiles/RandomUnderSampling_datatrain.csv')
            elif model == 'Over':
                data_train = pd.read_csv('CSVFiles/RandomOverSampling_datatrain.csv')
            elif model == 'Both':
                data_train = pd.read_csv('CSVFiles/RandomOver&UnderSampling_datatrain.csv')
            elif model == 'Smote':
                data_train = pd.read_csv('CSVFiles/SMOTE_datatrain.csv')
            elif model == 'BorderlineSmote':
                data_train = pd.read_csv('CSVFiles/BorderlineSMOTE_datatrain.csv')
            elif model == 'SVMSmote':
                data_train = pd.read_csv('CSVFiles/SVMSMOTE_datatrain.csv')
            elif model == 'Adasyn':
                data_train = pd.read_csv('CSVFiles/ADASYN_datatrain.csv')
            elif model == 'Rose':
                data_train = pd.read_csv('CSVFiles/rose_train_data.csv')

            # Shuffle the data
            data_train = data_train.sample(frac=1)

            df_train = Utils.PCA_variables(data_train)

            if model == 'Rose':
                data_test = pd.read_csv('CSVFiles/rose_test_data.csv')
            else:
                data_test = pd.read_csv('CSVFiles/data_test.csv')

            df_test = Utils.PCA_variables(data_test)

            train = df_train.loc[:, ~df_train.columns.str.contains('^Unnamed')]
            test = df_test.loc[:, ~df_test.columns.str.contains('^Unnamed')]

            ttrain = train.values
            ttest = test.values

            data_index = 5

            y_train = ttrain[:, data_index]
            x_train = ttrain[:, 0:data_index]

            y_test = ttest[:, data_index]
            x_test = ttest[:, 0:data_index]

            x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
            x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

            # Hyper-parameters
            data_df = hyperparameter_df.loc[hyperparameter_df['Sampling_Method'] == model]

            learning_rate = data_df['Learning_Rate'].iloc[0]
            dropout_rate = data_df['Dropout_Rate'].iloc[0]
            gru_units = data_df['GRU_Units'].iloc[0]
            neurons_dense = data_df['Neuron_Dense'].iloc[0]
            hidden_layers = data_df['Hidden_Layers'].iloc[0]
            epochs = data_df['Epochs'].iloc[0]
            batch_size = data_df['Batch_Size'].iloc[0]
            activation = data_df['Activation'].iloc[0]

            print('Hyper parameters:')
            print(model, learning_rate, dropout_rate, gru_units, neurons_dense, hidden_layers, epochs, batch_size,
                  activation)
            if value == 'MLP':
                # multi layers perceptron
                mlp_datamodel = KerasGRUModel(learning_rate, dropout_rate, gru_units, neurons_dense, hidden_layers,
                                              epochs,
                                              batch_size, activation, x_train, y_train)
                Utils.pca_classificationMetrics("GRU", mlp_datamodel, x_test, y_test, model, value)
            elif value == 'Sequential':
                # sequential
                seq_datamodel = SequentialGRUModel(learning_rate, dropout_rate, gru_units, neurons_dense, hidden_layers,
                                                   epochs, batch_size, activation, x_train, y_train)
                Utils.pca_classificationMetrics("GRU", seq_datamodel, x_test, y_test, model, value)


def ROCCurve(x_test, y_test, model):
    pred = model.predict(x_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test, pred)
        roc_auc[i] = auc(fpr[i], tpr[i])

    print(roc_auc_score(y_test, pred))
    plt.figure()
    plt.plot(fpr[1], tpr[1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.show()


def PRCurve(x_test, y_test, model):
    yhat = model.predict(x_test)
    model_probs = yhat[:, 0]
    # calculate the precision-recall auc
    precision, recall, _ = precision_recall_curve(y_test, model_probs)
    auc_score = auc(recall, precision)
    print('Logistic PR AUC: %.3f' % auc_score)
    # plot precision-recall curves
    # plot model precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, model_probs)
    plt.plot(recall, precision, marker='.', label='Binary')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

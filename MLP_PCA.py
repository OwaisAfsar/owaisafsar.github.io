import tensorflow as tf
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import Utils


def get_keras_model(num_hidden_layers,
                    neuron_dense,
                    dropout_rate,
                    activation,
                    x_train):
    inputs = tf.keras.Input(shape=(x_train.shape[1],))
    x = layers.Dropout(dropout_rate)(inputs)

    for i in range(num_hidden_layers):
        x = layers.Dense(neuron_dense,
                         activation=activation)(x)
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def KerasMLPModel(learning_rate, dropout_rate, neuron_dense, hidden_layers, epochs,
                  batch_size, activation, x_train, y_train):
    keras_model = get_keras_model(hidden_layers, neuron_dense, dropout_rate, activation, x_train)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    keras_model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    data = x_train
    labels = y_train
    res = keras_model.fit(data, labels, epochs=epochs, batch_size=batch_size)
    return keras_model


def get_keras_seq_model(num_hidden_layers,
                        neuron_dense,
                        dropout_rate,
                        activation,
                        x_train):
    inputs = tf.keras.Input(shape=(x_train.shape[1],))
    x = tf.keras.Sequential()(inputs)

    for i in range(num_hidden_layers):
        x = layers.Dense(neuron_dense,
                         activation=activation)(x)
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def KerasSequentialModel(learning_rate, dropout_rate, neuron_dense, hidden_layers, epochs,
                         batch_size, activation, x_train, y_train):
    keras_model = get_keras_seq_model(hidden_layers, neuron_dense, dropout_rate, activation, x_train)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    keras_model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    data = x_train
    labels = y_train
    res = keras_model.fit(data, labels, epochs=epochs, batch_size=batch_size)
    return keras_model


def MLP_Classifer():
    data_models = ['Under', 'Over', 'Both', 'Smote', 'BorderlineSmote', 'SVMSmote', 'Adasyn', 'Rose']
    # data_models = ['Under']
    hyperparameter_value = ['MLP', 'Sequential']
    # hyperparameter_value = ['MLP']

    for value in hyperparameter_value:
        hyperparameter_df = Utils.get_hyperparameter_values(value)

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

            y_train = train.has_conflict.copy()
            x_train = train.drop(['has_conflict'], axis=1)

            y_test = test.has_conflict.copy()
            x_test = test.drop(['has_conflict'], axis=1)

            # Hyper-parameters
            data_df = hyperparameter_df.loc[hyperparameter_df['Sampling_Method'] == model]

            learning_rate = data_df['Learning_Rate'].iloc[0]
            dropout_rate = data_df['Dropout_Rate'].iloc[0]
            neurons_dense = data_df['Neuron_Dense'].iloc[0]
            hidden_layers = data_df['Hidden_Layers'].iloc[0]
            epochs = data_df['Epochs'].iloc[0]
            batch_size = data_df['Batch_Size'].iloc[0]
            activation = data_df['Activation'].iloc[0]

            print('Hyper parameters:')
            print(model, learning_rate, dropout_rate, neurons_dense, hidden_layers, epochs, batch_size, activation)
            if value == 'MLP':
                # print('MULTI LAYER PERCEPTRON')
                mlp_datamodel = KerasMLPModel(learning_rate, dropout_rate, neurons_dense, hidden_layers, epochs,
                                              batch_size, activation, x_train, y_train)
                Utils.pca_classificationMetrics("DNN", mlp_datamodel, x_test, y_test, model, value)
            elif value == 'Sequential':
                # print('SEQUENTIAL')
                seq_datamodel = KerasSequentialModel(learning_rate, dropout_rate, neurons_dense, hidden_layers,
                                                     epochs, batch_size, activation, x_train, y_train)
                Utils.pca_classificationMetrics("DNN", seq_datamodel, x_test, y_test, model, value)


def ROCCurve(x_test, y_test, model):
    pred = model.predict(x_test)
    print(pred)
    # pred_prob = model.predict_proba(x_test)

    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh = {}

    n_class = 2

    for i in range(n_class):
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred, pos_label=i)

    # plotting
    plt.plot(fpr[0], tpr[0], linestyle='--', color='orange', label='Class 0')
    plt.plot(fpr[1], tpr[1], linestyle='--', color='green', label='Class 1')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.show()


def PRCurve(x_test, y_test, model):
    prediction = model.predict(x_test)
    model_probabilities = prediction[:, 0]
    precision, recall, _ = precision_recall_curve(y_test, model_probabilities)
    auc_score = auc(recall, precision)
    print('Logistic PR AUC: %.3f' % auc_score)
    precision, recall, _ = precision_recall_curve(y_test, model_probabilities)
    plt.plot(recall, precision, marker='.', label='Binary')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()

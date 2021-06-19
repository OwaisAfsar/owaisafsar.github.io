import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
import plotly.graph_objects as go
from ax.service.ax_client import AxClient
from ax.utils.notebook.plotting import render, init_notebook_plotting


def sequential_hyperparameter():
    def mlp_sequential():
        def get_keras_model(num_hidden_layers,
                            neurons_dense,
                            dropout_rate,
                            activation):
            # define the layers.
            inputs = tf.keras.Input(shape=(1, 10))
            x = tf.keras.Sequential()(inputs)

            # Add the hidden layers.
            for i in range(num_hidden_layers):
                x = layers.Dense(neurons_dense,
                                 activation=activation)(x)
                x = layers.Dropout(dropout_rate)(x)

            outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model

        # This function takes in the hyperparameters and returns a score (Cross validation).
        def keras_cv_score(parameterization, weight=None):
            tf.keras.backend.clear_session()
            model = get_keras_model(parameterization.get('num_hidden_layers'),
                                    parameterization.get('neurons_dense'),
                                    parameterization.get('dropout_rate'),
                                    parameterization.get('activation'))

            learning_rate = parameterization.get('learning_rate')
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            NUM_EPOCHS = parameterization.get('num_epochs')

            model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(),
                          metrics=[tf.keras.metrics.AUC()])

            # fit the model using a 20% validation set.
            res = model.fit(x=x_train,
                            y=y_train,
                            batch_size=parameterization.get('batch_size'),
                            epochs=NUM_EPOCHS,
                            validation_data=(x_test, y_test))

            last_score = np.array(res.history['val_auc'][-1:])
            return last_score, 0

        parameters = [
            {
                "name": "learning_rate",
                "type": "range",
                "bounds": [0.0001, 0.5],
                "log_scale": True,
            },
            {
                "name": "dropout_rate",
                "type": "range",
                "bounds": [0.01, 0.5],
                "log_scale": True,
            },
            {
                "name": "neurons_dense",
                "type": "range",
                "bounds": [1, 300],
                "value_type": "int"
            },
            {
                "name": "num_hidden_layers",
                "type": "range",
                "bounds": [1, 10],
                "value_type": "int"
            },
            {
                "name": "num_epochs",
                "type": "range",
                "bounds": [1, 100],
                "value_type": "int"
            },
            {
                "name": "activation",
                "type": "choice",
                "values": ['tanh', 'sigmoid', 'relu'],
            },
            {
                "name": "batch_size",
                "type": "range",
                "bounds": [8, 64],
                "value_type": "int"
            },
        ]

        datatrain = pd.read_csv('rose_train_data.csv')
        data_test = pd.read_csv('rose_test_data.csv')

        train = datatrain.loc[:, ~datatrain.columns.str.contains('^Unnamed')]
        test = data_test.loc[:, ~data_test.columns.str.contains('^Unnamed')]

        ttrain = train.values
        ttest = test.values

        y_train = ttrain[:, 10]
        x_train = ttrain[:, 0:10]

        y_test = ttest[:, 10]
        x_test = ttest[:, 0:10]

        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

        init_notebook_plotting()
        ax_client = AxClient()

        # create the experiment.
        ax_client.create_experiment(
            name="keras_experiment",
            parameters=parameters,
            objective_name='keras_cv',
            minimize=False)

        def evaluate(parameters):
            return {"keras_cv": keras_cv_score(parameters)}

        for i in range(25):
            parameters, trial_index = ax_client.get_next_trial()
            ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

        best_parameters, values = ax_client.get_best_parameters()

        # the best set of parameters.
        for k in best_parameters.items():
            print(k)

        print()

        # the best score achieved.
        means, covariances = values
        print(means)

    def lstm_sequential():
        def get_keras_model(lstm_units,
                            num_hidden_layers,
                            neurons_dense,
                            dropout_rate,
                            activation):
            # define the layers.
            inputs = tf.keras.Input(shape=(1, 10))
            x = tf.keras.Sequential()(inputs)
            x = layers.LSTM(units=lstm_units)(x)

            # Add the hidden layers.
            for i in range(num_hidden_layers):
                x = layers.Dense(neurons_dense,
                                 activation=activation)(x)
                x = layers.Dropout(dropout_rate)(x)

            outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model

        # This function takes in the hyperparameters and returns a score (Cross validation).
        def keras_cv_score(parameterization, weight=None):
            tf.keras.backend.clear_session()
            model = get_keras_model(parameterization.get('lstm_units'),
                                    parameterization.get('num_hidden_layers'),
                                    parameterization.get('neurons_dense'),
                                    parameterization.get('dropout_rate'),
                                    parameterization.get('activation'))

            learning_rate = parameterization.get('learning_rate')
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            NUM_EPOCHS = parameterization.get('num_epochs')

            model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(),
                          metrics=[tf.keras.metrics.AUC()])

            # fit the model using a 20% validation set.
            res = model.fit(x=x_train,
                            y=y_train,
                            batch_size=parameterization.get('batch_size'),
                            epochs=NUM_EPOCHS,
                            validation_data=(x_test, y_test))

            last_score = np.array(res.history['val_auc'][-1:])
            return last_score, 0

        parameters = [
            {
                "name": "learning_rate",
                "type": "range",
                "bounds": [0.0001, 0.5],
                "log_scale": True,
            },
            {
                "name": "dropout_rate",
                "type": "range",
                "bounds": [0.01, 0.5],
                "log_scale": True,
            },
            {
                "name": "neurons_dense",
                "type": "range",
                "bounds": [1, 300],
                "value_type": "int"
            },
            {
                "name": "num_hidden_layers",
                "type": "range",
                "bounds": [1, 10],
                "value_type": "int"
            },
            {
                "name": "num_epochs",
                "type": "range",
                "bounds": [1, 100],
                "value_type": "int"
            },
            {
                "name": "lstm_units",
                "type": "range",
                "bounds": [1, 10],
                "value_type": "int"
            },
            {
                "name": "activation",
                "type": "choice",
                "values": ['tanh', 'sigmoid', 'relu'],
            },
            {
                "name": "batch_size",
                "type": "range",
                "bounds": [8, 64],
                "value_type": "int"
            },
        ]

        datatrain = pd.read_csv('rose_train_data.csv')
        data_test = pd.read_csv('rose_test_data.csv')

        train = datatrain.loc[:, ~datatrain.columns.str.contains('^Unnamed')]
        test = data_test.loc[:, ~data_test.columns.str.contains('^Unnamed')]

        ttrain = train.values
        ttest = test.values

        y_train = ttrain[:, 10]
        x_train = ttrain[:, 0:10]

        y_test = ttest[:, 10]
        x_test = ttest[:, 0:10]

        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

        init_notebook_plotting()
        ax_client = AxClient()

        # create the experiment.
        ax_client.create_experiment(
            name="keras_experiment",
            parameters=parameters,
            objective_name='keras_cv',
            minimize=False)

        def evaluate(parameters):
            return {"keras_cv": keras_cv_score(parameters)}

        for i in range(25):
            parameters, trial_index = ax_client.get_next_trial()
            ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

        best_parameters, values = ax_client.get_best_parameters()

        # the best set of parameters.
        for k in best_parameters.items():
            print(k)

        print()

        # the best score achieved.
        means, covariances = values
        print(means)

    # lstm_sequential()

    def gru_sequential():
        def get_keras_model(gru_units,
                            num_hidden_layers,
                            neurons_dense,
                            dropout_rate,
                            activation):
            # define the layers.
            inputs = tf.keras.Input(shape=(1, 10))
            x = tf.keras.Sequential()(inputs)
            x = layers.LSTM(units=gru_units)(x)

            # Add the hidden layers.
            for i in range(num_hidden_layers):
                x = layers.Dense(neurons_dense,
                                 activation=activation)(x)
                x = layers.Dropout(dropout_rate)(x)

            outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model

        # This function takes in the hyperparameters and returns a score (Cross validation).
        def keras_cv_score(parameterization, weight=None):
            tf.keras.backend.clear_session()
            model = get_keras_model(parameterization.get('gru_units'),
                                    parameterization.get('num_hidden_layers'),
                                    parameterization.get('neurons_dense'),
                                    parameterization.get('dropout_rate'),
                                    parameterization.get('activation'))

            learning_rate = parameterization.get('learning_rate')
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            NUM_EPOCHS = parameterization.get('num_epochs')

            model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(),
                          metrics=[tf.keras.metrics.AUC()])

            # fit the model using a 20% validation set.
            res = model.fit(x=x_train,
                            y=y_train,
                            batch_size=parameterization.get('batch_size'),
                            epochs=NUM_EPOCHS,
                            validation_data=(x_test, y_test))

            last_score = np.array(res.history['val_auc'][-1:])
            return last_score, 0

        parameters = [
            {
                "name": "learning_rate",
                "type": "range",
                "bounds": [0.0001, 0.5],
                "log_scale": True,
            },
            {
                "name": "dropout_rate",
                "type": "range",
                "bounds": [0.01, 0.5],
                "log_scale": True,
            },
            {
                "name": "neurons_dense",
                "type": "range",
                "bounds": [1, 300],
                "value_type": "int"
            },
            {
                "name": "num_hidden_layers",
                "type": "range",
                "bounds": [1, 10],
                "value_type": "int"
            },
            {
                "name": "num_epochs",
                "type": "range",
                "bounds": [1, 100],
                "value_type": "int"
            },
            {
                "name": "gru_units",
                "type": "range",
                "bounds": [1, 10],
                "value_type": "int"
            },
            {
                "name": "activation",
                "type": "choice",
                "values": ['tanh', 'sigmoid', 'relu'],
            },
            {
                "name": "batch_size",
                "type": "range",
                "bounds": [8, 64],
                "value_type": "int"
            },
        ]

        datatrain = pd.read_csv('rose_train_data.csv')
        data_test = pd.read_csv('rose_test_data.csv')

        train = datatrain.loc[:, ~datatrain.columns.str.contains('^Unnamed')]
        test = data_test.loc[:, ~data_test.columns.str.contains('^Unnamed')]

        ttrain = train.values
        ttest = test.values

        y_train = ttrain[:, 10]
        x_train = ttrain[:, 0:10]

        y_test = ttest[:, 10]
        x_test = ttest[:, 0:10]

        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

        init_notebook_plotting()
        ax_client = AxClient()

        # create the experiment.
        ax_client.create_experiment(
            name="keras_experiment",
            parameters=parameters,
            objective_name='keras_cv',
            minimize=False)

        def evaluate(parameters):
            return {"keras_cv": keras_cv_score(parameters)}

        for i in range(25):
            parameters, trial_index = ax_client.get_next_trial()
            ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

        best_parameters, values = ax_client.get_best_parameters()

        # the best set of parameters.
        for k in best_parameters.items():
            print(k)

        print()

        # the best score achieved.
        means, covariances = values
        print(means)

    gru_sequential()


def hyperparameter():
    datatrain = pd.read_csv('SVMSMOTE_datatrain.csv')
    data_test = pd.read_csv('data_test.csv')

    train = datatrain.loc[:, ~datatrain.columns.str.contains('^Unnamed')]
    test = data_test.loc[:, ~data_test.columns.str.contains('^Unnamed')]

    ttrain = train.values
    ttest = test.values

    y_train = ttrain[:, 10]
    x_train = ttrain[:, 0:10]

    y_test = ttest[:, 10]
    x_test = ttest[:, 0:10]

    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    def get_keras_model(num_hidden_layers,
                        neurons_dense,
                        dropout_rate,
                        activation):
        # define the layers.
        inputs = tf.keras.Input(shape=(1, 10))
        x = tf.keras.layers.Dropout(dropout_rate)(inputs)

        # Add the hidden layers.
        for i in range(num_hidden_layers):
            x = layers.Dense(neurons_dense,
                             activation=activation)(x)
            x = layers.Dropout(dropout_rate)(x)

        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    # This function takes in the hyperparameters and returns a score (Cross validation).
    def keras_cv_score(parameterization, weight=None):
        tf.keras.backend.clear_session()
        model = get_keras_model(parameterization.get('num_hidden_layers'),
                                parameterization.get('neurons_dense'),
                                parameterization.get('dropout_rate'),
                                parameterization.get('activation'))

        learning_rate = parameterization.get('learning_rate')
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        NUM_EPOCHS = parameterization.get('num_epochs')

        model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.AUC()])

        # fit the model using a 20% validation set.
        res = model.fit(x=x_train,
                        y=y_train,
                        batch_size=parameterization.get('batch_size'),
                        epochs=NUM_EPOCHS,
                        validation_data=(x_test, y_test))

        last_score = np.array(res.history['val_auc'][-1:])
        return last_score, 0

    parameters = [
        {
            "name": "learning_rate",
            "type": "range",
            "bounds": [0.0001, 0.5],
            "log_scale": True,
        },
        {
            "name": "dropout_rate",
            "type": "range",
            "bounds": [0.01, 0.5],
            "log_scale": True,
        },
        {
            "name": "neurons_dense",
            "type": "range",
            "bounds": [1, 300],
            "value_type": "int"
        },
        {
            "name": "num_hidden_layers",
            "type": "range",
            "bounds": [1, 10],
            "value_type": "int"
        },
        {
            "name": "num_epochs",
            "type": "range",
            "bounds": [1, 100],
            "value_type": "int"
        },
        {
            "name": "activation",
            "type": "choice",
            "values": ['tanh', 'sigmoid', 'relu'],
        },
        {
            "name": "batch_size",
            "type": "range",
            "bounds": [8, 64],
            "value_type": "int"
        },
    ]

    init_notebook_plotting()
    ax_client = AxClient()

    # create the experiment.
    ax_client.create_experiment(
        name="keras_experiment",
        parameters=parameters,
        objective_name='keras_cv',
        minimize=False)

    def evaluate(parameters):
        return {"keras_cv": keras_cv_score(parameters)}

    for i in range(25):
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

    # look at all the trials.
    all_trials = ax_client.get_trials_data_frame().sort_values('trial_index')
    print(all_trials)

    best_parameters, values = ax_client.get_best_parameters()

    # the best set of parameters.
    for k in best_parameters.items():
        print(k)

    print()

    # the best score achieved.
    means, covariances = values
    print(means)

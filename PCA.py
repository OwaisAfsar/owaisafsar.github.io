import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def pcanalysis(_X, _y, _target, _columns):

    if 'cont_id' in _X:
        _X = _X.drop(['cont_id'], axis=1)

    scaler = StandardScaler()
    X = scaler.fit_transform(_X)
    pca_component = PCA(n_components=2)
    pca_component.fit(X, _y)
    principalComponents = pca_component.transform(X)
    principalDf = pd.DataFrame(data=principalComponents,
                                        columns=['PC 1', 'PC 2'])

    # print("ratio: ", pca_component.explained_variance_ratio_[0], pca_component.explained_variance_ratio_[1])
    target = _y.to_frame()
    # print(target.value_counts())
    features = principalDf
    target.reset_index(drop=True, inplace=True)
    print(features)
    # final_df = principalDf
    # final_df[_target] = target[_target]
    # # print(final_df)
    #
    # plot_components(final_df, _target, pca_component.explained_variance_ratio_[0],
    #                 pca_component.explained_variance_ratio_[1])
    # # _columns.remove(_target)
    # plot_pca_circle(pca_component, _columns)

    return features, target


def plot_components(_frame, _target, _explainedratio1, _explainedratio2):
    explainedRatio1 = "{:.2f}".format(_explainedratio1 * 100)
    explainedRatio2 = "{:.2f}".format(_explainedratio2 * 100)
    totalRatio = "{:.2f}".format((_explainedratio1 + _explainedratio2) * 100)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1: ' + str(explainedRatio1) + '%', fontsize=15)
    ax.set_ylabel('PC2: ' + str(explainedRatio2) + '%', fontsize=15)
    ax.set_title('2 components PCA: ' + str(totalRatio) + '%', fontsize=20)
    targets = [0, 1]
    colors = ['r', 'g', 'b']
    for target, color in zip(targets, colors):
        indicesToKeep = _frame[_target] == target
        ax.scatter(_frame.loc[indicesToKeep, 'PC 1'],
                   _frame.loc[indicesToKeep, 'PC 2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    fig.show()


def plot_pca_circle(_pca, _columns):
    PCs = _pca.components_

    # Use quiver to generate the basic plot
    fig = plt.figure(figsize=(10,10))
    plt.quiver(np.zeros(PCs.shape[1]), np.zeros(PCs.shape[1]),
               PCs[0,:], PCs[1,:],
               angles='xy', scale_units='xy', scale=1)

    for i,j,z in zip(PCs[1,:]+0.02, PCs[0,:]+0.04, _columns):
        plt.text(j, i, z, ha='center', va='center')

    # Add unit circle
    circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
    plt.gca().add_artist(circle)

    # Ensure correct aspect ratio and axis limits
    plt.axis('equal')
    plt.xlim([-1.0,1.0])
    plt.ylim([-1.0,1.0])

    plt.xlabel('PC 0')
    plt.ylabel('PC 1')

    plt.show()
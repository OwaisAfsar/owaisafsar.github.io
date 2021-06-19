import DataModelling
import Data_Imbalance_Technique
import GRU
import GRU_PCA
import LSTM
import LSTM_PCA
import MLP
import MLP_PCA
import Utils

# RESEARCH QUESTION 1 & 2
DataModelling.research_question()

# Chi-Square Test
Utils.ChiSquareTest()
#
# Data Imbalance Techniques
Data_Imbalance_Technique.data_imbalance_techniques()  # For 'rose', need to separately run the script 'Rose_DIT.R'

# Classifiers' Training
MLP.MLP_Classifer()
LSTM.LSTM_Classifier()
GRU.GRU_Classifer()
Utils.classification_report_csv_df()

# PCA Classifiers' Training
# For PCA graph, need to separately run the script 'PCA.R'
MLP_PCA.MLP_Classifer()
LSTM_PCA.LSTM_Classifier()
GRU_PCA.GRU_Classifer()
Utils.pca_classification_report_csv_df()





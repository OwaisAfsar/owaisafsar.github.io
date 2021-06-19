import numpy as np
import pandas as pd
import scipy
from scipy.stats import chi2, chi2_contingency

def ChiSquare(ms1, cms1, ms2, cms2):
    data_array = np.array([[ms1, cms1], [ms2, cms2]])
    df = pd.DataFrame(data_array, columns=["No Conflicts", "Conflicts"])
    df.index = ["Top", "Occ"]
    calculation_chi_square(df)

def calculation_chi_square(df):
    statistics, p_value, dof, exp = chi2_contingency(df, correction=False)  # "correction=False" means no Yates' correction is used!
    print("Chi-squared: " + str(statistics))
    print("Degree of Freedom: ", dof)
    print("P-value: " + str(p_value))


def calculation_chi_square_manual(df):
    df2 = df.copy()  # create contingency table with the marginal totals and the grand total.
    df2.loc['Column_Total'] = df2.sum(numeric_only=True, axis=0)
    df2.loc[:, 'Row_Total'] = df2.sum(numeric_only=True, axis=1)

    n_total = df2.at["Column_Total", "Row_Total"]  # grand total

    exp = df2.copy()  # create dataframe with expected counts
    for x in exp.index[0:-1]:
        for y in exp.columns[0:-1]:
            # round expected values to 6 decimal places to get the maximum available precision:
            v = (((df2.at[x, "Row_Total"]) * (df2.at["Column_Total", y])) / n_total).round(6)
            exp.at[x, y] = float(v)

    exp = exp.iloc[[0, 1], [0, 1]]
    statistics = np.sum(((df - exp) ** 2 / exp).values)  # calculate chi-squared test statistic
    dof = (len(df.columns) - 1) * (len(df.index) - 1)  # determine degrees of freedom
    p_value = 1 - chi2.cdf(statistics, dof)  # subtract the cumulative distribution function from 1

    print("Chi Square Statistics:")
    print("Statistics: ", statistics)
    print("Degree of Freedom: ", dof)
    print("P Value: ", p_value)
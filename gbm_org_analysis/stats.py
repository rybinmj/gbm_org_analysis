import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm


def save_to_txt(filename, text):
    """Saves formatted text to a .txt file."""
    if filename[-4:] == ".txt":
        full_filename = filename
    else:
        full_filename = filename + ".txt"

    file = open(full_filename, 'w')
    file.write(text)
    file.close


def anova(df):
    """Performs anova and tukey hsd post hoc on columns of df"""

    values_list = df.values.tolist()
    values_list_without_nan = []
    for values in values_list:
        values_list_without_nan.append(
            [value for value in values if str(value) != 'nan'])

    fvalue, pvalue = stats.f_oneway(*values_list_without_nan)

    df_tidy = pd.melt(df).dropna()
    tky = pairwise_tukeyhsd(
        endog=df_tidy[df_tidy.columns[1]],
        groups=df_tidy[df_tidy.columns[0]],
        alpha=0.05,
    )
    return fvalue, pvalue, tky


def anova_values_to_text(fvalue, pvalue, tky):
    """Accepts results from anova test and formats as string to be saved as text file"""
    f = "F value: " + str(fvalue)
    p = "P value: " + str(pvalue)

    text = f + "\n" + p + 2 * "\n" + str(tky)

    return text


def anova_results_as_text(df):
    """Performas anova on df and formats results as string"""

    fvalue, pvalue, tky = anova(df)
    f = "F value: " + str(fvalue)
    p = "P value: " + str(pvalue)

    text = f + "\n" + p + 2 * "\n" + str(tky)

    return text


def linreg(tidydf_col1, tidydf_col2, as_text=False):
    """
    Performs a linear regression analysis comparing two tidyform df columns.
    Example of how to format tidydf_col parameters: df_tidy[col_name]
    """
    model = sm.OLS(tidydf_col1, tidydf_col2).fit()

    if as_text is False:
        result = model.summary()
    else:
        result = str(model.summary())

    return result

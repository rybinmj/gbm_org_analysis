import pandas as pd
import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# get_ipython().run_line_magic('matplotlib', 'notebook')


class GbmInvasion:
    """
    DTS and CN Data

    Extracts Distance to Surface (DTS) and Cell Number (CN) Data for GBMinvasion experiments.

    Attributes
    ----------
    path : str
        file path to dts files
            Example: directory + "/*_Imaris/*/*_Shortest_Distance_to_Surfaces_Surfaces=Surfaces_2.csv"
    groups_order : list
        list of groups (as strings) in order to be displayed in figures
            Example: ["none_2days", "none_12days", ...]
            Note: only need to list each group once in the correct order
    time_points : list of int, optional
        experimental time points (e.g. 2 (for 2 days) or 12 (for 12 days))
    arbitrary_thresholds : list of int, optional
        values of thresholds of what is considered the "core" (first value in list)
        and "surface" (second value in list) of organoid.

    Methods
    -------
    hist(self, data, hist=True, kde=True)

    hist_groups(self, group, norm=False, shade=True, legend=False)

    kde_groups(self, group, norm=False, shade=True, legend=False)

    kde_groups_comb(self, treatment, norm=False, shade=True, legend=False)

    cn_boxplots(self, data, groups="all", norm="raw", save=None)

    stats_barplots(self, stats_or_cn, stat, groups="all", norm=False, save=None)

    scat_comb(self, norm=False, boxplot=True, save=None)

    scat_sep(self, day="all", norm=False, boxplot=True, save=None)

    krus(self, list_of_dfs, all_results=False)

    anova(self, data, norm="raw")

    anova2(self, df, df_tidy)


    Initialized Variables
    ---------------------

    # File Path Attributes

    self.dts_path = path + "/*_Imaris/*/*_Shortest_Distance_to_Surfaces_Surfaces=Surfaces_2.csv"
    self.xyz_path = path + "/*_Imaris/*_Statistics/*_Position.csv"
    self.time_points = time_points
    self.arbitrary_thresholds = arbitrary_thresholds

    # Defining Variables

    if self.time_points is not None:
        self.time1 = str(time_points[0])
        self.time2 = str(time_points[1])

    self.days = "days"
    self.separator = "_"

    if self.time_points is not None:
        self.tidycolname1 = "Group_Day_Org"
    else:
        self.tidycolname1 = "Group"
    self.dts_tidycolname2 = "DTS"
    self.cn_tidycolname2 = "Cell Number"
    self.great_tidycolname2 = "Num Greater Than X"
    self.less_tidycolname2 = "Num Less Than X"
    self.tidycolname3 = "Group"

    # Groups

    self.groups = None
    self.groups_order = groups_order
    # list of column names for wide form df (i.e. identifiers)
    self.df_cols = None
    # list of column names for tidy df (i.e. groups)
    self.dfcomb_cols = None
    # list of all groups according to order of groups_order list
    self.ordered_groups = None

    self.ordered_groups_time1 = None
    self.ordered_groups_time1_unique = []
    self.ordered_groups_time2 = None
    self.ordered_groups_time2_unique = []
    self.ordered_all = None
    self.ordered_all_time1 = None
    self.ordered_all_time2 = None

    # Distance to Surface Attibutes

    self.dts_dict = None  # dict of identifiers and its list of data values
    self.dts_df = None  # df of data for each identifier
    self.dts_df_tidy = None  # tidyform of dts_df
    self.dts_df_tidy_time1 = None
    self.dts_df_tidy_time2 = None
    self.dts_dictcomb = None  # dict of group identifiers and list of combined data values
    self.dts_dfcomb = None  # df of combined data
    self.dts_dfcomb_tidy = None  # tidyform of dts_dfcomb

    self.dts_df_normfirst = None
    self.dts_df_tidy_normfirst = None
    self.dts_dfcomb_normfirst = None
    self.dts_dfcomb_tidy_normfirst = None

    # Cell Number Attributes

    self.cn_df = None
    self.cn_df_tidy = None
    self.cn_dictcomb = None
    self.cn_dfcomb = None
    self.cn_dfcomb_tidy = None

    # Average of total cell number at 2 days for each group. Used for normtot.
    self.cn_dictcomb_avg = {}
    self.cn_dictcomb_normtot = {}
    self.cn_dfcomb_normtot = None
    self.cn_dfcomb_normtot_tidy = None

    self.cn_dictcomb_normless = {}
    self.cn_dfcomb_normless = None
    self.cn_dfcomb_normless_tidy = None

    # Number of Cells Greater Than X Attributes

    self.g_dict = None
    self.g_df = None
    self.g_df_tidy = None
    self.g_dictcomb = None
    self.g_dfcomb = None
    self.g_dfcomb_tidy = None

    # Average of greater than number at 2 days for each group. Used for normtot.
    self.g_dictcomb_avg = {}
    self.g_dictcomb_normtot = {}
    self.g_dfcomb_normtot = None
    self.g_dfcomb_normtot_tidy = None

    self.g_dictcomb_normless = {}
    self.g_dfcomb_normless = None
    self.g_dfcomb_normless_tidy = None

    # Number of Cells Less Than X Attributes

    self.l_dict = None
    self.l_df = None
    self.l_df_tidy = None
    self.l_dictcomb = None
    self.l_dfcomb = None
    self.l_dfcomb_tidy = None

    # Average of less than number at 2 days for each group. Used for normtot.
    self.l_dictcomb_avg = {}
    self.l_dictcomb_normtot = {}
    self.l_dfcomb_normtot = None
    self.l_dfcomb_normtot_tidy = None

    self.l_dictcomb_normless = {}
    self.l_dfcomb_normless = None
    self.l_dfcomb_normless_tidy = None

    # Quartile Statistics

    self.stats_list = [
        "min",
        "25%",
        "50%",
        "75%",
        "max",
        "range",
        "iqr",
        "q1q4",
        "q2q4",
        "q3q4",
    ]

    self.dts_df_descriptives = None  # df with all descriptives for each organoids.

    self.stats_dicts = []
    self.stats_dfs = []
    self.stats_dfs_tidy = []

    self.dts_df_normfirst_descriptives = None

    self.normfirst_stats_dicts = []
    self.normfirst_stats_dfs = []
    self.normfirst_stats_dfs_tidy = []

    # Number of Cells Greater Than Each Statistic

    self.cn_greater_dfs = []
    self.cn_greater_dfs_tidy = []

    self.cn_greater_dfs_normfirst = []
    self.cn_greater_dfs_tidy_normfirst = []

    # Normal Distribution Tests

    self.shapiro = None
    self.normal = None
    self.anderson = None


    # Attributes for Figures

    self.colors_dict = {}
    self.colors_ordered_all = []
    self.colors_ordered_all_time1 = []
    self.colors_ordered_all_time2 = []

    self.fig_style = {
        "figure.facecolor": "w",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }

    self.rc = {
        'font.size': 20.0,
        'axes.labelsize': 30.0,
        'axes.titlesize': 'large',
        'xtick.labelsize': 20.0,
        'ytick.labelsize': 20.0,
        'legend.fontsize': 'medium'
    }

    self.rc_for_scatter = {
        'font.size': 20.0,
        'axes.labelsize': 30.0,
        'axes.titlesize': 'large',
        'xtick.labelsize': 15.0,
        'ytick.labelsize': 15.0,
        'legend.fontsize': 'medium'
    }

    # font dict for histograms

    self.fontdict = {
        "size": 20,
    }
    """

    def __init__(self, path, groups_order, time_points=None, arbitrary_thresholds=None,):
        """
        Initialize DTS & Cell Number Attributes

        Parameters
        ----------
        path : str
            file path to dts files
                Example: directory + "/*_Imaris/*/*_Shortest_Distance_to_Surfaces_Surfaces=Surfaces_2.csv"
        groups_order : list
            list of groups (as strings) in order to be displayed in figures
                Example: ["none_2days", "none_12days", ...]
                Note: only need to list each group once in the correct order
        time_points : list of int, optional
            experimental time points (e.g. 2 (for 2 days) or 12 (for 12 days))
        arbitrary_thresholds : list of int, optional
            values of thresholds of what is considered the "core" (first value in list)
            and "surface" (second value in list) of organoid.

        """

        # File Path Attributes

        self.dts_path = path + "/*_Imaris/*/*_Shortest_Distance_to_Surfaces_Surfaces=Surfaces_2.csv"
        self.xyz_path = path + "/*_Imaris/*/*_Position.csv"
        self.time_points = time_points
        self.arbitrary_thresholds = arbitrary_thresholds

        # Defining Variables

        if self.time_points is not None:
            self.time1 = str(time_points[0])
            self.time2 = str(time_points[1])

        self.days = "days"
        self.separator = "_"

        if self.time_points is not None:
            self.tidycolname1 = "Group_Day_Org"
        else:
            self.tidycolname1 = "Group"
        self.dts_tidycolname2 = "DTS"
        self.cn_tidycolname2 = "Cell Number"
        self.great_tidycolname2 = "Num Greater Than X"
        self.less_tidycolname2 = "Num Less Than X"
        self.tidycolname3 = "Group"

        # Groups

        self.groups = None
        self.groups_order = groups_order
        # list of column names for wide form df (i.e. identifiers)
        self.df_cols = None
        # list of column names for tidy df (i.e. groups)
        self.dfcomb_cols = None
        # list of all groups according to order of groups_order list
        self.ordered_groups = None

        self.ordered_groups_time1 = None
        self.ordered_groups_time1_unique = []
        self.ordered_groups_time2 = None
        self.ordered_groups_time2_unique = []
        self.ordered_all = None
        self.ordered_all_time1 = None
        self.ordered_all_time2 = None

        # Distance to Surface Attibutes

        self.dts_dict = {}  # dict of identifiers and its list of data values
        self.dts_df = None  # df of data for each identifier
        self.dts_df_tidy = None  # tidyform of dts_df
        self.dts_df_tidy_time1 = None
        self.dts_df_tidy_time2 = None
        self.dts_dictcomb = None  # dict of group identifiers and list of combined data values
        self.dts_dfcomb = None  # df of combined data
        self.dts_dfcomb_tidy = None  # tidyform of dts_dfcomb

        self.dts_df_normfirst = None
        self.dts_df_tidy_normfirst = None
        self.dts_dfcomb_normfirst = None
        self.dts_dfcomb_tidy_normfirst = None

        self.dts_dict_rmout = None
        self.dts_df_rmout = None
        self.dts_df_rmout_tidy = None

        self.dts_dict_rmout_normfirst = None
        self.dts_df_rmout_normfirst = None
        self.dts_df_rmout_tidy_normfirst = None

        # Cell Number Attributes

        self.cn_df = None
        self.cn_df_tidy = None
        self.cn_dictcomb = None
        self.cn_dfcomb = None
        self.cn_dfcomb_tidy = None

        # Average of total cell number at 2 days for each group. Used for normtot.
        self.cn_dictcomb_avg = {}
        self.cn_dictcomb_normtot = {}
        self.cn_dfcomb_normtot = None
        self.cn_dfcomb_normtot_tidy = None

        self.cn_dictcomb_normless = {}
        self.cn_dfcomb_normless = None
        self.cn_dfcomb_normless_tidy = None

        # Number of Cells Greater Than X Attributes

        self.g_dict = None
        self.g_df = None
        self.g_df_tidy = None
        self.g_dictcomb = None
        self.g_dfcomb = None
        self.g_dfcomb_tidy = None

        self.g_dict_rmout = None
        self.g_df_rmout = None
        self.g_df_tidy_rmout = None
        self.g_dictcomb_rmout = None
        self.g_dfcomb_rmout = None
        self.g_dfcomb_tidy_rmout = None

        self.g_dictcomb_avg = {}  # Average of greater than number at 2 days for each group. Used for normtot.
        self.g_dictcomb_normtot = {}
        self.g_dfcomb_normtot = None
        self.g_dfcomb_normtot_tidy = None

        self.g_dictcomb_normless = {}
        self.g_dfcomb_normless = None
        self.g_dfcomb_normless_tidy = None

        self.g_dfcomb_normg2days = None
        self.g_dfcomb_normg2days_tidy = None

        # Number of Cells Less Than X Attributes

        self.l_dict = None
        self.l_df = None
        self.l_df_tidy = None
        self.l_dictcomb = None
        self.l_dfcomb = None
        self.l_dfcomb_tidy = None

        self.l_dictcomb_avg = {}  # Average of less than number at 2 days for each group. Used for normtot.
        self.l_dictcomb_normtot = {}
        self.l_dfcomb_normtot = None
        self.l_dfcomb_normtot_tidy = None

        self.l_dictcomb_normless = {}
        self.l_dfcomb_normless = None
        self.l_dfcomb_normless_tidy = None

        # Quartile Statistics

        self.stats_list = [
            "min",
            "25%",
            "50%",
            "75%",
            "max",
            "range",
            "iqr",
            "q1q4",
            "q2q4",
            "q3q4",
            "std3",
        ]

        self.dts_df_descriptives = None  # df with all descriptives for each organoids.

        self.stats_dicts = []
        self.stats_dfs = []
        self.stats_dfs_tidy = []

        self.dts_df_normfirst_descriptives = None

        self.normfirst_stats_dicts = []
        self.normfirst_stats_dfs = []
        self.normfirst_stats_dfs_tidy = []

        self.dts_df_rmout_descriptives = None

        self.rmout_stats_dicts = []
        self.rmout_stats_dfs = []
        self.rmout_stats_dfs_tidy = []

        # Number of Cells Greater Than Each Statistic

        self.cn_greater_dfs = []
        self.cn_greater_dfs_tidy = []

        self.cn_greater_dfs_normfirst = []
        self.cn_greater_dfs_tidy_normfirst = []

        # Normal Distribution Tests

        self.shapiro = None
        self.normal = None
        self.anderson = None


        # Attributes for Figures

        self.colors_dict = {}
        self.colors_ordered_all = []
        self.colors_ordered_all_time1 = []
        self.colors_ordered_all_time2 = []

        self.fig_style = {
            "figure.facecolor": "w",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }

        self.rc = {
            'font.size': 20.0,
            'axes.labelsize': 30.0,
            'axes.titlesize': 'large',
            'xtick.labelsize': 20.0,
            'ytick.labelsize': 20.0,
            'legend.fontsize': 'medium'
        }

        self.rc_for_scatter = {
            'font.size': 20.0,
            'axes.labelsize': 30.0,
            'axes.titlesize': 'large',
            'xtick.labelsize': 15.0,
            'ytick.labelsize': 15.0,
            'legend.fontsize': 'medium'
        }

        # font dict for histograms

        self.fontdict = {
            "size": 20,
        }


        #########################################
        # === Getting Data for Initialization ===
        #########################################

        # Making tuple of groups based on groups_order

        if self.time_points is None:
            self.groups = tuple([self.get_group(x) for x in groups_order])
        else:
            g = []
            for i in groups_order:
                l = i.split(self.separator)
                g.append(l[0])
            h = []
            [h.append(x) for x in g if x not in h]
            self.groups = tuple(h)

        #############################
        # === Distance to Surface ===
        #############################

        self.dts_dict = self.extract_data(self.dts_path)
        self.xyz_dict = self.extract_data(self.xyz_path)

        self.data_dict = {}
        df = self.mk_df(self.dts_dict, self.ordered_all)
        for k, v in self.xyz_dict.items():
            if k not in self.data_dict:
                self.data_dict[k] = pd.concat([v, df[k]], axis=1).dropna()

        for df in self.data_dict.values():
            for col in df.columns:
                index = df[df[col] < 0].index
                df.drop(index, inplace=True)

        dfs = self.data_dict.values()
        self.data_df = pd.concat(dfs, axis=1)

        for key, value in self.dts_dict.items():
            self.dts_dict.update({key: [i for i in value if i >= 0]})

#         self.dts_dict = dts_dict

        self.dts_df = self.mk_df(self.dts_dict, self.ordered_all)

        self.dts_df_tidy = self.mk_tidy(self.dts_df, self.dts_tidycolname2)

        if self.time_points is not None:
            t = self.dts_df_tidy
            self.dts_df_tidy_time1 = t[
                ~t[self.tidycolname1].str.contains(
                    f"{self.separator}{self.time2}{self.days}")
            ]
            self.dts_df_tidy_time1 = t[
                ~t[self.tidycolname1].str.contains(
                    f"{self.separator}{self.time1}{self.days}")
            ]

        # === Getting Groups ===

        columns = list(self.dts_df.columns)
        columns.sort
        self.df_cols = columns

        g = self.unique(list(map(self.get_group, columns)))
        g.sort()
        self.dfcomb_cols = g

        # Putting groups in order
        long = [self.get_group(i) for i in self.dts_df.columns]
        n = []
        for s in self.groups_order:
            for l in long:
                if l == s:
                    n.append(l)
        self.ordered_groups = n

        if self.time_points is not None:
            self.ordered_groups_time1 = [
                i for i in self.ordered_groups if f"{self.separator}{self.time1}{self.days}" in i
            ]

            t1 = self.ordered_groups_time1
            [self.ordered_groups_time1_unique.append(
                x) for x in t1 if x not in self.ordered_groups_time1_unique]

            self.ordered_groups_time2 = [
                i for i in self.ordered_groups if f"{self.separator}{self.time2}{self.days}" in i
            ]

            t2 = self.ordered_groups_time2
            [self.ordered_groups_time2_unique.append(
                x) for x in t2 if x not in self.ordered_groups_time2_unique]

        # Putting all groups and orgs in order
        n = []
        for s in self.groups_order:
            for l in self.df_cols:
                if l.startswith(s):
                    n.append(l)
        self.ordered_all = n

        if self.time_points is not None:
            self.ordered_all_time1 = [
                i for i in self.ordered_all if f"{self.separator}{self.time1}{self.days}" in i
            ]
            self.ordered_all_time2 = [
                i for i in self.ordered_all if f"{self.separator}{self.time2}{self.days}" in i
            ]

        self.dts_dfcomb = self.mk_dfcomb(self.dts_df)

        self.dts_dfcomb_tidy = self.mk_tidy(self.dts_dfcomb, self.dts_tidycolname2)

        # === Norm DTS to first value ===

        self.dts_df_normfirst = self.dts_df.apply(lambda col_values: col_values - col_values.min())

        self.dts_df_tidy_normfirst = self.mk_tidy(self.dts_df_normfirst, self.dts_tidycolname2)

        self.dts_dfcomb_normfirst = self.dts_dfcomb.apply(lambda col_values: col_values - col_values.min())

        self.dts_dfcomb_tidy_normfirst = self.mk_tidy(self.dts_dfcomb_normfirst, self.dts_tidycolname2)

        #################################
        # === DTS Quartile Statistics ===
        #################################

        # Raw

        self.dts_df_descriptives = self.dts_df.apply(lambda x: self.my_describe(x))

        for stat in self.stats_list:
            self.stats_dicts.append(self.get_stat(self.dts_df_descriptives, stat))

        for dct in self.stats_dicts:
            self.stats_dfs.append(self.mk_df(dct, self.groups_order))

        for df in self.stats_dfs:
            self.stats_dfs_tidy.append(self.mk_tidy(df, self.dts_tidycolname2))

        # Quartiles DTS Normalized to First

        self.dts_df_normfirst_descriptives = self.dts_df_normfirst.apply(lambda x: self.my_describe(x))

        for stat in self.stats_list:
            self.normfirst_stats_dicts.append(self.get_stat(self.dts_df_normfirst_descriptives, stat))

        for dct in self.normfirst_stats_dicts:
            self.normfirst_stats_dfs.append(self.mk_df(dct, self.groups_order))

        for df in self.normfirst_stats_dfs:
            self.normfirst_stats_dfs_tidy.append(self.mk_tidy(df, self.dts_tidycolname2))

        # Quartiles DTS Outliers Removed

        self.dts_dict_rmout = self.rm_outliers(self.dts_dict)
        self.dts_df_rmout = self.mk_df(self.dts_dict_rmout, self.ordered_all)
        self.dts_df_rmout_tidy = self.mk_tidy(self.dts_df_rmout, self.dts_tidycolname2)

        # Not working b/c rm_outliers takes a dict but dont have a normfirst dts dict.
        self.normfirst_dts_dict_rmout = self.rm_outliers(self.dts_df_normfirst)
        self.normfirst_dts_df_rmout = self.mk_df(self.normfirst_dts_dict_rmout, self.ordered_all)
        self.normfirst_dts_df_rmout_tidy = self.mk_tidy(self.normfirst_dts_df_rmout, self.dts_tidycolname2)

        self.dts_df_rmout_descriptives = self.dts_df_rmout.apply(lambda x: self.my_describe(x))

        for stat in self.stats_list:
            self.rmout_stats_dicts.append(self.get_stat(self.dts_df_rmout_descriptives, stat))

        for dct in self.rmout_stats_dicts:
            self.rmout_stats_dfs.append(self.mk_df(dct, self.groups_order))

        for df in self.rmout_stats_dfs:
            self.rmout_stats_dfs_tidy.append(self.mk_tidy(df, self.dts_tidycolname2))

        # === CN Greater Than Stat ===

        for stat in self.stats_list:
            df = self.cn_greater_than(self.dts_df, stat)
            self.cn_greater_dfs.append(df)

        for df in self.cn_greater_dfs:
            self.cn_greater_dfs_tidy.append(self.mk_tidy(df, self.cn_tidycolname2))

        for stat in self.stats_list:
            df = self.cn_greater_than(self.dts_df_normfirst, stat)
            self.cn_greater_dfs_normfirst.append(df)

        for df in self.cn_greater_dfs_normfirst:
            self.cn_greater_dfs_tidy_normfirst.append(self.mk_tidy(df, self.cn_tidycolname2))

        #####################
        # === Cell Number ===
        #####################

        self.cn_df = pd.DataFrame(self.dts_df.count()).transpose()
        self.cn_df_tidy = self.mk_tidy(self.cn_df, self.cn_tidycolname2)

        self.cn_dfcomb = self.mk_dfcomb(self.cn_df)

        self.cn_dfcomb_tidy = self.mk_tidy(self.cn_dfcomb, self.cn_tidycolname2)

        ##############################
        # === Arbitrary Thresholds ===
        ##############################

        if arbitrary_thresholds is not None:
            self.g_dict = self.mk_dict_greaterthan(self.dts_dict)

            self.g_df = self.mk_df(self.g_dict, self.ordered_all)

            self.g_df_tidy = self.mk_tidy(self.g_df, self.great_tidycolname2)

            self.g_dfcomb = self.mk_dfcomb(self.g_df)

            self.g_dfcomb_tidy = self.mk_tidy(self.g_dfcomb, self.great_tidycolname2)

            # Number greater with outliers removed

            self.g_dict_rmout = self.mk_dict_greaterthan(self.dts_dict_rmout)

            self.g_df_rmout = self.mk_df(self.g_dict_rmout, self.ordered_all)

            self.g_df_rmout_tidy = self.mk_tidy(self.g_df_rmout, self.great_tidycolname2)

            self.g_dfcomb_rmout = self.mk_dfcomb(self.g_df_rmout)

            self.g_dfcomb_rmout_tidy = self.mk_tidy(self.g_dfcomb_rmout, self.great_tidycolname2)

            # Number of Cells Less Than ("Surface of Organoid")

            self.l_dict = self.mk_dict_lessthan(self.dts_dict)

            self.l_df = self.mk_df(self.l_dict, self.ordered_all)

            self.l_df_tidy = self.mk_tidy(self.l_df, self.less_tidycolname2)

            self.l_dfcomb = self.mk_dfcomb(self.l_df)

            self.l_dfcomb_tidy = self.mk_tidy(self.l_dfcomb, self.less_tidycolname2)

            # === Averages for Normalizations ===
            # Note: these dictionaries contain only the key, value averages for 2 day time points.

            # Total Cell Number

            self.cn_dictcomb_avg = self.get_avgs_2days(self.cn_dfcomb)
            self.g_dictcomb_avg = self.get_avgs_2days(self.g_dfcomb)
            self.l_dictcomb_avg = self.get_avgs_2days(self.l_dfcomb)

            ###################################
            # === Cell Number Normalization ===
            ###################################

            # To Total Cells @ 2 Days:

            self.cn_dfcomb_normtot = self.norm_to_avg2days(self.cn_dictcomb_avg, self.cn_dfcomb)
            self.cn_dfcomb_normtot_tidy = self.mk_tidy(self.cn_dfcomb_normtot, self.cn_tidycolname2)

            # To Number of Cells Less than lX:

            self.cn_dfcomb_normless = self.norm_to_avg2days(self.l_dictcomb_avg, self.cn_dfcomb)
            self.cn_dfcomb_normless_tidy = self.mk_tidy(self.cn_dfcomb_normless, self.cn_tidycolname2)

            # === Number Greater Than gX Normalization ===

            # To Total Cells @ 2 days

            self.g_dfcomb_normtot = self.norm_to_avg2days(self.g_dictcomb_avg, self.g_dfcomb)
            self.g_dfcomb_normtot_tidy = self.mk_tidy(self.g_dfcomb_normtot, self.great_tidycolname2)

            # To Number Less Than lX @ 2 Days

            self.g_dfcomb_normless = self.norm_to_avg2days(self.l_dictcomb_avg, self.g_dfcomb)
            self.g_dfcomb_normless_tidy = self.mk_tidy(self.g_dfcomb_normless, self.great_tidycolname2)

            # === Number of Cell Less Than lX Normalization ===

            # To Total Cells @ 2 Days:

            self.l_dfcomb_normtot = self.norm_to_avg2days(self.cn_dictcomb_avg, self.l_dfcomb)
            self.l_dfcomb_normtot_tidy = self.mk_tidy(self.l_dfcomb_normtot, self.less_tidycolname2)

            # To Number Less than lX @ 2 days:

            self.l_dfcomb_normless = self.norm_to_avg2days(self.l_dictcomb_avg, self.l_dfcomb)
            self.l_dfcomb_normless_tidy = self.mk_tidy(self.l_dfcomb_normless, self.less_tidycolname2)

            # Cell Num gX to Number gX @ 2days:

            self.g_dfcomb_normg2days = self.norm_to_g2days(self.g_dfcomb)
            self.g_dfcomb_normg2days_tidy = self.mk_tidy(self.g_dfcomb_normg2days, self.great_tidycolname2)

        #################################################
        # === Colors and Groups of Colors For Figures ===
        #################################################

        if self.time_points is not None:
            c = sns.color_palette("Paired").as_hex()
            self.colors_dict = dict(zip(self.groups_order, c))
            for k, v in self.colors_dict.items():
                for i in self.ordered_all:
                    if i.startswith(k):
                        self.colors_ordered_all.append(v)

            for k, v in self.colors_dict.items():
                for i in self.ordered_all_time1:
                    if i.startswith(k):
                        self.colors_ordered_all_time1.append(v)

            for k, v in self.colors_dict.items():
                for i in self.ordered_all_time2:
                    if i.startswith(k):
                        self.colors_ordered_all_time2.append(v)
        else:
            c = sns.color_palette("muted").as_hex()
            self.colors_dict = dict(zip(self.groups_order, c))
            for k, v in self.colors_dict.items():
                for i in self.ordered_all:
                    if i.startswith(k):
                        self.colors_ordered_all.append(v)


        #########################
        # === Normality Tests ===
        #########################

        self.shapiro = self.dts_df.apply(lambda x: stats.shapiro(x.dropna()))

        self.normal = self.dts_df.apply(lambda x: stats.normaltest(x.dropna()))

        self.anderson = self.dts_df.apply(lambda x: stats.anderson(x.dropna()))


    ###################
    # === Functions ===
    ###################

    def extract_data(self, path):
        n = {}
        for fullname in glob.glob(path):
            if "vol" in fullname:
                continue
            n.update({fullname: fullname.split(self.separator)})

        dct = {}
        for key, value in n.items():
            for string in value:
                if string.startswith("org"):
                    org = string
                if string.endswith(self.days):
                    day = string
                if string.startswith(self.groups):
                    group = string

            if path == self.dts_path:
                data = pd.read_csv(key, header=2, usecols=[0])["Shortest Distance to Surfaces"].tolist()
            elif path == self.xyz_path:
                data = pd.read_csv(key, header=2, usecols=[0,1,2])

            if self.time_points is None:
                if path == self.xyz_path:
                    data = data.rename(
                        columns={
                            "Position X": "X_" + f"{group}{self.separator}{org}",
                            "Position Y": "Y_" + f"{group}{self.separator}{org}",
                            "Position Z": "Z_" + f"{group}{self.separator}{org}",
                        }
                    )
                dct.update({f"{group}{self.separator}{org}": data})
            else:
                if path == self.xyz_path:
                    data = data.rename(
                        columns={
                            "Position X": "X_" + f"{group}{self.separator}{day}{self.separator}{org}",
                            "Position Y": "Y_" + f"{group}{self.separator}{day}{self.separator}{org}",
                            "Position Z": "Z_" + f"{group}{self.separator}{day}{self.separator}{org}",
                        }
                    )
                dct.update({f"{group}{self.separator}{day}{self.separator}{org}": data})


        for key, value in dct.items():
            if type(value) is dict:
                for col, vals in value.items():
                    new_col = self.removespaces(col)
                    value[new_col] = vals
        return dct

    def removespaces(self, string):
        return string.replace(" ", "")

    def unique(self, l):
        """Removes duplicate values from a list"""
        return list(set(l))

    def get_group(self, s):
        """Returns characters of string before the first underscore"""
        if "_" in s:
            pos = s.index("_")
            try:
                return s[:s.index("_", pos + 1)]
            except ValueError:
                return s[:pos]
        else:
            return s

    def figtofile(self, fig, path, figname):
        """Saves figure as .pdf file"""
        pth = path + "/" + figname + ".pdf"
        fig.get_figure().savefig(pth)

    def mk_df(self, dct, columns):
        df = pd.DataFrame.from_dict(dct, orient="index").transpose().reindex(columns=columns)
        return df

    def mk_tidy(self, df, tidycolname2, col1_default_name=True):
        """Transforms a wide form df to long form. Uses DTS as tidycolname2"""
        if col1_default_name is True:
            col1 = self.tidycolname1
        else:
            col1 = col1_default_name
        df_tidy = pd.melt(df, var_name=col1, value_name=tidycolname2).dropna()
        return df_tidy

    def mk_dfcomb(self, df):
        """Making combined dataframe from individuals dfs"""
        groupsdict = {}
        for i in self.dfcomb_cols:
            groupsdict.update({i: ""})
            for key, value in groupsdict.items():
                if key == i:
                    col = [c for c in self.df_cols if c.startswith(i)]
                    groupsdict[key] = df[col]

        gcomb = {}
        for key, value in groupsdict.items():
            full = value.values.T.ravel()
            cleaned = [x for x in full if str(x) != "nan"]
            gcomb.update({key: cleaned})

        dfcomb = self.mk_df(gcomb, self.groups_order)
        return dfcomb

    def my_describe(self, s):
        """Adds additional descriptive stats to .describe()"""
        description = s.describe()
        description['range'] = description['max'] - description['min']
        description['iqr'] = description['75%'] - description['25%']
        description['q1q4'] = description['max'] - description['25%']
        description['q2q4'] = description['max'] - description['50%']
        description['q3q4'] = description['max'] - description['75%']
        description['std3'] = description['mean'] + 3 * description['std']
        return description

    def rm_outliers(self, dct):
        """Removes values greater than 3 Standard Deviations from the mean"""
        no_outliers_dict = {}
        for key, values in dct.items():
            std3 = self.dts_df_descriptives[key]['std3']
            no_outliers_list = [value for value in values if value < std3]
            no_outliers_dict.update({key: no_outliers_list})
        return no_outliers_dict

    def get_stat(self, df, stat):
        """ """
        result = {}
        for column_name, cur_column in df.items():
            group = column_name[:-5]
            if group not in result:
                result[group] = []

            min_value = cur_column[stat]
            result[group] = result[group] + [min_value]

        return result

    def cn_greater_than(self, dts_dataframe, stat):
        """Makes a dataframe of number of cells above a given quartile stat"""
        dct = {}
        stat_df_idx = int(self.stats_list.index(stat))
        for col, val in dts_dataframe.items():
            n = int(col[-1]) - 1
            group = col[:-5]
            if group not in dct:
                dct[group] = []
            stat_value = float(self.stats_dfs[stat_df_idx][group][n])
            values_greater = [x for x in val if float(x) > stat_value]
            dct[group] = dct[group] + [len(values_greater)]

        df = pd.DataFrame.from_dict(dct, orient="index").transpose().reindex(columns=self.dfcomb_cols)

        return df

    def mk_dict_greaterthan(self, dct):
        gX_dict = {}
        for key, value in dct.items():
            count = len([i for i in value if i > self.arbitrary_thresholds[0]])
            gX_dict.update({key: count})
        return gX_dict

    def mk_dict_lessthan(self, dct):
        lX_dict = {}
        for key, value in dct.items():
            count = len([i for i in value if i < self.arbitrary_thresholds[1]])
            lX_dict.update({key: count})
        return lX_dict

    def get_avgs_2days(self, dfcomb):
        """Returns dict of avg cn for each group when given a combined df of cell number"""
        avg_dct = {}
        for col, val in dfcomb.items():
            i = sum(val) / len(val)
            avg_dct.update({col: i})
        for key in [key for key in avg_dct if f"{self.separator}{self.time2}{self.days}" in key]:
            del avg_dct[key]
        return avg_dct

    def norm_to_avg2days(self, dictcomb_avg, dfcomb):
        """Normalizes dfcomb to dictcomb_avg. Used for normtot and normless"""
        dictcomb_normalized = {}
        for key, val in dictcomb_avg.items():
            for col, values in dfcomb.items():
                if key.startswith(self.get_group(col)):
                    dictcomb_normalized.update({col: [i / val for i in values]})
        df = self.mk_df(dictcomb_normalized, self.groups_order)
        return df


    def norm_to_g2days(self, dfcomb):
        """Normalizes given dataframe to df values at 2 days. Used for normgdays"""
        def get_day2_value(group):
            cond = self.get_group(group)
            for k, v in g_avg_2days.items():
                if k.startswith(cond):
                    return float(v)

        g_avg_2days = {}
        groups = self.ordered_groups_time1_unique
        groups2 = self.ordered_groups_time2_unique

        for col, values in dfcomb.items():
            if col not in g_avg_2days and f"{self.time2}{self.days}" not in col:
                g_avg_2days[col] = values.mean()

        df_new = pd.DataFrame()
        for group in self.groups_order:
            day2_value = day2_value = get_day2_value(group)
            df_new[group] = dfcomb[group] / day2_value

        return df_new

    def cols_to_list(self, df):
        """Returns values of each column of df as list of lists"""
        lst = []
        for col, val in df.items():
            values = [x for x in val.tolist() if str(x) != "nan"]
            lst.append(values)
        return(lst)

    def cols_to_arrays(self, df):
        """
        Transforms the columns of a df into a list of arrays.

        Useful for preparing argument for anova functions (stats.f_oneway) which takes array-like
        arguments. Can use *cols_to_arrays(df) to perform anova on columns of df.
        """

        l = []
        length = len(df.columns)
        for i in range(length):
            array = np.array(df.iloc[:,i])
            array = array[~np.isnan(array)]
            l.append(array)
        return l

    def rm_cols(self, df, list_of_columns):
        """Removes columns from dataframe that are in list"""
        for group in list_of_columns:
            if group in df:
                del df[group]
        return df


    #################
    # === Figures ===
    #################

    ####################
    # === Histograms ===
    ####################

    def hist(self, data, hist=True, kde=True):
        """
        Creates histogram of one organoids DTS data.

        Parameters
        ----------
        data : str
            org identifier corresponding to column headings of dts_df
        hist : bool, default False
            adds or removes histogram bins from plot
        kde : bool, default True
            adds or removes kde line from plot

        Returns
        -------
        Histogram (Seaborn displot) of individual organoids DTS data.
        """

        plt.figure()
        n = self.dts_df.columns.tolist().index(data)
        fig = sns.distplot(self.dts_df.iloc[:,n].dropna(), hist=hist, kde=kde)

        plt.xlabel("Distance to Surface", fontdict=self.fontdict)
        plt.ylabel("Frequency", fontdict=self.fontdict)

        sns.despine()
        return fig
        plt.clf()

    def hist_groups(self, group, norm=False, shade=True, legend=False):
        """
        Creates histogram of groups of organoids DTS data.

        Parameters
        ----------
        group : str
            group  (corresponding to column headings of self.dts_dfcomb)
        norm : bool, default False
            if true, uses normalized dts data
        shade : bool, default True
            adds or removes shading under kde plots
        legend : bool, default False
            adds or removes plot legend


        Returns
        -------
        Histogram (Seaborn displot) of groups of organoids DTS data.
        """
        if norm is False:
            data1 = self.dts_df
        else:
            data1 = self.dts_df_normfirst

        plt.figure()
        for org in self.ordered_all:
            if org.startswith(group):
                n = self.dts_df.columns.tolist().index(org)
                fig = plt.hist(
                    data1.iloc[:,n].dropna(),
                    histtype="step"
                )

                plt.xlabel("Distance to Surface", fontdict=self.fontdict)
                plt.ylabel("Number of Cells", fontdict=self.fontdict)
                plt.title(group, fontdict=self.fontdict)

                sns.despine()
        return fig
        plt.clf()

    ###################
    # === KDE PLOTS ===
    ###################

    def kde_groups(self, group, norm=False, shade=True, legend=False):
        """
        Creates histogram of groups of organoids DTS data.

        Parameters
        ----------
        group : str
            group  (corresponding to column headings of self.dts_dfcomb)
        norm : bool, default False
            if true, uses normalized dts data
        shade : bool, default True
            adds or removes shading under kde plots
        legend : bool, default False
            adds or removes plot legend


        Returns
        -------
        Histogram (Seaborn displot) of groups of organoids DTS data.
        """
        if norm is False:
            data1 = self.dts_df
        else:
            data1 = self.dts_df_normfirst

        plt.figure()
        for org in self.ordered_all:
            if org.startswith(group):
                n = self.dts_df.columns.tolist().index(org)
                fig = sns.kdeplot(
                    data1.iloc[:,n].dropna(),
                    legend=legend,
                    shade=shade,
                )

                fig.set_xlabel("Distance to Surface", fontdict=self.fontdict)
                fig.set_ylabel("Frequency", fontdict=self.fontdict)
                fig.set_title(group, fontdict=self.fontdict)

                sns.despine()
                plt.tight_layout()
        return fig
        plt.clf()

    def kde_groups_comb(self, treatment, norm=False, shade=True, legend=False):
        """
        Creates histogram of group's dts data combined into one hist.

        Parameters
        ----------
        treatment : str
            treatment group  (corresponding to value in self.groups)
        norm : bool, default False
            if true, uses normalized dts data
        shade : bool, default True
            adds or removes shading under kde plots
        legend : bool, default False
            adds or removes plot legend

        Returns
        -------
        Histogram (Seaborn displot) of groups of organoids DTS data.
        """
        if norm is False:
            data1 = self.dts_dfcomb
        else:
            data1 = self.dts_dfcomb_normfirst

        plt.figure()
        for group in self.groups_order:
            if group.startswith(treatment):
                n = self.dts_dfcomb.columns.tolist().index(group)
                fig = sns.kdeplot(
                    data1.iloc[:,n].dropna(),
                    legend=legend,
                    shade=shade,
                )

                fig.set_xlabel("Distance to Surface", fontdict=self.fontdict)
                fig.set_ylabel("Frequency", fontdict=self.fontdict)
                fig.set_title(treatment, fontdict=self.fontdict)

                sns.despine()
                plt.tight_layout()
        return fig
        plt.clf()




    ##########################################################
    # === Boxplots of Cell Number (raw, normtot, and normless)
    ##########################################################

    def cn_plots(self, data, box_or_bar="bar", groups="all", norm="raw", save=None):
        """
        Creates barplots of CN data.

        Note: Arbitrary thresholds must be initialized to plot greater than, less than,
        normtot, and normless. Without initializing thresholds can only plot raw cell number data.

        Parameters
        ----------
        data : str
            "cn", "g", or "l"
        groups : str, default "all"
            "*days", "*days", "all"
            *days correspond to time1 and time2 (e.g. "2days" or "12days")
        norm : str, default "raw"
            "raw", "normtot", or "normless"
        save : list, optional
            [figure_path, figure_name]
            If additional argument is passed then figure will be saved as a
            .pdf file to the directory specified by figure_path with the name
            specified by figure_name

        Returns
        -------
        Seaborn boxplot.

        """

        yrange1 = 0
        yrange2 = 2.5

        init_thresh_msg = "***** Need to Initialize Threshold Values *****"

        if self.time_points is not None:
            if groups == f"{self.time1}{self.days}":
                order = self.ordered_groups_time1_unique
                palette = ['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', ]
                t = self.time1 + " Days: "

            elif groups == f"{self.time2}{self.days}":
                order = self.ordered_groups_time2_unique
                palette = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', ]
                t = self.time2 + " Days: "
            elif groups == "all":
                order = self.groups_order
                palette = "Paired"
                t = ""
        else:
            if groups == "all":
                order = self.groups_order
                palette = "Paired"
                t = "All: "

        if data == "cn":
            ydata = self.cn_tidycolname2

            if norm == "raw":
                data1 = self.cn_dfcomb
                data2 = self.cn_dfcomb_tidy
                title = t + "Total # of Cells"
                yrange1 = 0
                yrange2 = 1200

            elif norm == "normtot":
                if self.arbitrary_thresholds is None:
                    print(init_thresh_msg)
                else:
                    data1 = self.cn_dfcomb_normtot
                    data2 = self.cn_dfcomb_normtot_tidy
                    title = t + "Total # of Cells Norm To Tot"

            elif norm == "normless":
                if self.arbitrary_thresholds is None:
                    print(init_thresh_msg)
                else:
                    data1 = self.cn_dfcomb_normless
                    data2 = self.cn_dfcomb_normless_tidy
                    title = t + "Total # of Cells Norm To Less"

        elif data == "g":
            if self.arbitrary_thresholds is None:
                print(init_thresh_msg)
            else:
                ydata = self.great_tidycolname2

                if norm == "raw":
                    data1 = self.g_dfcomb
                    data2 = self.g_dfcomb_tidy
                    title = t + "# Greater Than Raw"
                    yrange1 = 0
                    yrange2 = 300

                elif norm == "normtot":
                    data1 = self.g_dfcomb_normtot
                    data2 = self.g_dfcomb_normtot_tidy
                    title = t + "# Greater Norm To Tot"

                elif norm == "normless":
                    data1 = self.g_dfcomb_normless
                    data2 = self.g_dfcomb_normless_tidy
                    title = t + "# Greater Norm To Less"

        elif data == "l":
            if self.arbitrary_thresholds is None:
                print(init_thresh_msg)
            else:
                ydata = self.less_tidycolname2

                if norm == "raw":
                    data1 = self.l_dfcomb
                    data2 = self.l_dfcomb_tidy
                    title = t + "# Less Than Raw"
                    yrange1 = 0
                    yrange2 = 300

                elif norm == "normtot":
                    data1 = self.l_dfcomb_normtot
                    data2 = self.l_dfcomb_normtot_tidy
                    title = t + "# Less Norm To Tot"

                elif norm == "normless":
                    data1 = self.l_dfcomb_normless
                    data2 = self.l_dfcomb_normless_tidy
                    title = t + "# Less Norm To Less"

        if save is None:
            size = [12, 8]
        else:
            size = [8, 6]


        if box_or_bar is "box":
            plt.figure(figsize=size)
            plt.rcParams.update(**self.rc)

            plot = sns.boxplot(
                data=data1,
                order=order,
                palette=palette,
                linewidth=2.5,
                orient="h",
            )

            plot = sns.stripplot(
                x=ydata,
                y=self.tidycolname1,
                data=data2,
                order=order,
                color="gray",
                size=10,
                linewidth=0,
                jitter=0.1,
                orient="h",
            )

            plot.set_ylabel("")
            plot.set_xlabel("Number of Cells")

        else:
            plt.figure(figsize=size)
            plt.rcParams.update(**self.rc)

            plot = sns.barplot(
                data=data1,
                order=order,
                palette=palette,
                linewidth=2.5,
                orient="v",
                capsize=0.1,
                ci=68,
            )

            plot = sns.stripplot(
                x=self.tidycolname1,
                y=ydata,
                data=data2,
                order=order,
                color="gray",
                size=10,
                linewidth=0,
                jitter=0.1,
                orient="v",
            )

            plot.set_ylabel("Number of Cells")
            plot.set_xlabel("")

        plot.set_title(title, fontsize=30)

        sns.despine()
        plt.tight_layout()

        plt.show()

        if save is None:
            save = []
        else:
            self.figtofile(plot, save[0], save[1])

    def misc_barplots(self, df, df_tidy, tidycolname2, days="all", save=None):
        """

        Returns
        -------
        Seaborn boxplot.

        """

        if days == "all":
            order = self.groups_order
        elif days == f"{self.time1}{self.days}":
            order = self.ordered_groups_time1_unique
        elif days == f"{self.time2}{self.days}":
            order = self.ordered_groups_time2_unique

        palette = "Paired"

        yrange1 = 0
        yrange2 = 2.5

        if save is None:
            size = [12, 8]
        else:
            size = [8, 6]

        plt.figure(figsize=size)
        plt.rcParams.update(**self.rc)

        plot = sns.barplot(
            data=df,
            order=order,
            palette=palette,
            linewidth=2.5,
            orient="v",
            capsize=0.1,
            ci=68,
        )

        plot = sns.stripplot(
            x=self.tidycolname1,
            y=tidycolname2,
            data=df_tidy,
            order=order,
            color="gray",
            size=10,
            linewidth=0,
            jitter=0.1,
            orient="v",
        )

        plot.set_ylabel("Number of Cells")
        plot.set_xlabel("")

        sns.despine()
        plt.tight_layout()

        plt.show()

        if save is None:
            save = []
        else:
            self.figtofile(plot, save[0], save[1])



    ###################################################################
    # === Barplots of Statistics and Cell Number Greater Than Quartiles
    ###################################################################

    def stats_barplots(self, stats_or_cn, stat, groups="all", rm_outliers=False, norm=False, save=None):
        """
        Creates barplots of number of cells above quartile statistics.

        Parameters
        ----------
        stats_or_cn : str
            "stats" or "cn_greater"
        stat : str
            "min", "25%", "50%", "75%", "max", "range", "iqr", "q1q4", or "q3q4"
        groups : str, default "all"
            "*days", "*days", "all"
            *days correspond to time1 and time2 (e.g. "2days" or "12days")
        norm : bool, default False
            If true uses dts norm to first values to calculate cell numbers.
        save : list, optional
            [figure_path, figure_name]
            If additional argument is passed then figure will be saved as a
            .pdf file to the directory specified by figure_path with the name
            specified by figure_name

        Returns
        -------
        Seaborn barplot.

        """

        yrange1 = 0
        yrange2 = 2.5

        if self.time_points is not None:
            if groups == f"{self.time1}{self.days}":
                order = self.ordered_groups_time1_unique
                palette = ['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', ]
                t = self.time1 + " Days: "

            elif groups == f"{self.time2}{self.days}":
                order = self.ordered_groups_time2_unique
                palette = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', ]
                t = self.time2 + " Days: "
            elif groups == "all":
                order = self.groups_order
                palette = "Paired"
                t = ""
        else:
            if groups == "all":
                order = self.groups_order
                palette = "Paired"
                t = ""

        n = self.stats_list.index(stat)

        if stats_or_cn == "stats":
            ydata = self.dts_tidycolname2
            ylabel = "Distance to Surface"
            if norm is False:
                if rm_outliers is False:
                    data1 = self.stats_dfs[n]
                    data2 = self.stats_dfs_tidy[n]
                    title = t + stat + " (Raw)"
                else:
                    data1 = self.rmout_stats_dfs[n]
                    data2 = self.rmout_stats_dfs_tidy[n]
                    title = t + stat + " (Raw, Outliers Removed)"
            else:
                if rm_outliers is False:
                    data1 = self.normfirst_stats_dfs[n]
                    data2 = self.normfirst_stats_dfs_tidy[n]
                    title = t + stat + " (Normalized)"
                else:
                    print("***Haven't generated rmout, normfirst data yet.")
                    data1 = None
                    data2 = None
                    title = None

        if stats_or_cn == "cn_greater":
            ydata = self.cn_tidycolname2
            ylabel = "Number of Cells"
            if norm is False:
                data1 = self.cn_greater_dfs[n]
                data2 = self.cn_greater_dfs_tidy[n]
                title = t + "NumOfCells > " + stat + " (Raw)"
            else:
                data1 = self.cn_greater_dfs_normfirst[n]
                data2 = self.cn_greater_dfs_tidy_normfirst[n]
                title = t + "NumOfCells > " + stat + " (Normalized)"

        yrange1 = 0
        yrange2 = 1200

        if save is None:
            size = [20, 8]
        else:
            size = [10, 10]
            title = ""

        plt.figure(figsize=size)
        plt.rcParams.update(**self.rc)

        plot = sns.barplot(
            data=data1,
            order=order,
            palette=palette,
            capsize=0.1,
            ci=68,
        )

        plot = sns.stripplot(
            x=self.tidycolname1,
            y=ydata,
            data=data2,
            order=order,
            color="gray",
            size=10,
            linewidth=0,
            jitter=0.1,
        )

        plot.set_ylabel(ylabel)
        plot.set_xlabel("Group")
        plot.set_title(title, fontsize=30)

#         plt.ylim(yrange1, yrange2)
        sns.despine()
        plt.tight_layout()

        plt.show()

        if save is None:
            save = []
        else:
            self.figtofile(plot, save[0], save[1])

    ######################
    # === Scatterplots ===
    ######################

    # Combined Groups

    def scat_comb(self, norm=False, boxplot=True, save=None):
        """
        Creates a scatterplot of data combined by day.

        Parameters
        ----------
        norm : bool, default False
            If True, uses normalized DTS data.
        boxplot : bool, default True
            If False, does not plot boxplots.
        save : list, optional
            [figure_path, figure_name]
            If additional argument is passed then figure will be saved as a
            .pdf file to the directory specified by figure_path with the name
            specified by figure_name


        Returns
        -------
        Seaborn Stripplot of combined organoid dts values.

        """

        if save is None:
            size = [12.5, 12.5]
        else:
            size = [7.5, 9]

        plt.figure(figsize=size)
        plt.rcParams.update(**self.rc_for_scatter)

        if norm is False:
            data1 = self.dts_dfcomb_tidy
        else:
            data1 = self.dts_dfcomb_tidy_normfirst

        if self.time_points is None:
            palette = "muted"
        else:
            palette = "Paired"

        plot = sns.boxplot(
            x=self.dts_tidycolname2,
            y=self.tidycolname1,
            data=data1,
            order=self.groups_order,
            orient="h",
            fliersize=0,
            palette=palette,
        )

        if boxplot is False:
            plt.clf()

        plot = sns.stripplot(
            x=self.dts_tidycolname2,
            y=self.tidycolname1,
            data=data1,
            order=self.groups_order,
            orient="h",
            palette=palette,
            size=2.5,
            linewidth=0,
            jitter=0.25,
        )

        plot.set_xlabel("Distance to Surface")
        plot.set_ylabel("Group")
        plot.set_title("Distance to Surface", fontsize=30)
        sns.despine()
        plt.tight_layout()

        plt.show()

        if save is None:
            save = []
        else:
            self.figtofile(plot, save[0], save[1])

    def scat_sep(self, day="all", norm=False, rm_outliers=False, boxplot=True, save=None):
        """
        Creates scatter plot with individual organoid DTS values.

        Parameters
        ----------
        day : str, default "all"
            "all", "*days"
            *days corresponds to time1 or time2 (e.g. "2days" or "12days")
        norm : bool, default False
            If True, uses normalized DTS data.
        boxplot : bool, default True
            If False, does not plot boxplots.
        save : list, optional
            [figure_path, figure_name]
            If additional argument is passed then figure will be saved as a
            .pdf file to the directory specified by figure_path with the name
            specified by figure_name


        Returns
        -------
        Seaborn stripplot.

        """
        if self.time_points is not None:
            if day == "all":
                order = self.ordered_all
                colors = self.colors_ordered_all
                t = "All"

            elif day == f"{self.time1}{self.days}":
                order = self.ordered_all_time1
                colors = self.colors_ordered_all_time1
                t = self.time1 + " Days"

            elif day == f"{self.time2}{self.days}":
                order = self.ordered_all_time2
                colors = self.colors_ordered_all_time2
                t = self.time2 + " Days"
        else:
            if day == "all":
                order = self.ordered_all
                colors = self.colors_ordered_all
                t = ""

        if norm is True:
            if rm_outliers is True:
                data1 = self.normfirst_dts_df_rmout_tidy
            else:
                data1 = self.dts_df_tidy_normfirst
        else:
            if rm_outliers is True:
                data1 = self.dts_df_rmout_tidy
            else:
                data1 = self.dts_df_tidy

        if save is None:
            size = [12.5, 12.5]
        else:
            size = [7.5, 9]

        plt.figure(figsize=size)
        plt.rcParams.update(**self.rc_for_scatter)

        plot = sns.boxplot(
            x=self.dts_tidycolname2,
            y=self.tidycolname1,
            data=data1,
            order=self.ordered_all,
            orient="h",
            fliersize=0,
            palette=colors,
        )

        if boxplot is False:
            plt.clf()

        plot = sns.stripplot(
            x=self.dts_tidycolname2,
            y=self.tidycolname1,
            data=data1,
            order=order,
            orient="h",
            palette=colors,
            size=2.5,
            linewidth=0,
            jitter=0.25,
        )

        plot.set_xlabel("Distance to Surface")
        plot.set_ylabel("Groups")
#         plot.set_title(t, fontsize=30)
        sns.set_context(
            font_scale = 1.75,
            rc = self.fig_style
        )

        plt.tight_layout()
        sns.despine()

        plt.show()

        if save is None:
            save = []
        else:
            self.figtofile(plot, save[0], save[1])

    ####################
    # === Statistics ===
    ####################

    # Kruskal Wallis Test

    def krus(self, list_of_dfs, all_results=False):
        """
        Performs a Kruskal Wallis test on each df in list of dfs

        Returns a dictionary: {column of df: [kruskal statistic, p value]}

        Originally designed to be passed self.stats_dfs to run test on all quartile stats.

        Note: If all_results is false only add results to dictionary if p < 0.05.
        """
        i = 0
        results = {}
        for df in list_of_dfs:
            values = self.cols_to_list(df)
            try:
                krus_stat, pvalue = stats.kruskal(*values)
            except:
                continue
            if all_results is False:
                if pvalue < 0.05:
                    stat_name = self.stats_list[i]
                    results.update({stat_name: [krus_stat, pvalue]})
            else:
                stat_name = self.stats_list[i]
                results.update({stat_name: [krus_stat, pvalue]})
            i += 1
        return results

    # Anova

    def anova(self, data, norm="raw"):
        """
        Performs anova and tukey tests on DTS or CN data.

        Note: Must initialize thresholds to use normtot and normless.

        Parameters
        ----------
        data : str
            "dts", "cn", "g", "l"
        norm : str, default "raw"
            "raw", "normtot", "normless"

        Returns
        -------
        F value and P value from one-way anova &
        Pairwise Tukey HSD test results with alpha = 0.05.

        """

        if norm == "raw":
            if data == "dts":
                dictcomb = self.dts_dictcomb
                dfcomb_tidy = self.dts_dfcomb_tidy
                tidycolname = self.dts_tidycolname2
            if data == "cn":
                dictcomb = self.cn_dictcomb
                dfcomb_tidy = self.cn_dfcomb_tidy
                tidycolname = self.cn_tidycolname2
            elif data == "g":
                dictcomb = self.g_dictcomb
                dfcomb_tidy = self.g_dfcomb_tidy
                tidycolname = self.great_tidycolname2
            elif data == "l":
                dictcomb = self.l_dictcomb
                dfcomb_tidy = self.l_dfcomb_tidy
                tidycolname = self.less_tidycolname2
        if norm == "normtot":
            if data == "dts":
                print("***DTS Data Is Not Normalized. Same As Raw.", "\n")
                dictcomb = self.dts_dictcomb
                dfcomb_tidy = self.dts_dfcomb_tidy
                tidycolname = self.dts_tidycolname2
            if data == "cn":
                dictcomb = self.cn_dictcomb_normtot
                dfcomb_tidy = self.cn_dfcomb_normtot_tidy
                tidycolname = self.cn_tidycolname2
            elif data == "g":
                dictcomb = self.g_dictcomb_normtot
                dfcomb_tidy = self.g_dfcomb_normtot_tidy
                tidycolname = self.great_tidycolname2
            elif data == "l":
                dictcomb = self.l_dictcomb_normtot
                dfcomb_tidy = self.l_dfcomb_normtot_tidy
                tidycolname = self.less_tidycolname2
        if norm == "normless":
            if data == "dts":
                print("***DTS Data Is Not Normalized. Same As Raw.", "\n")
                dictcomb = self.dts_dictcomb
                dfcomb_tidy = self.dts_dfcomb_tidy
                tidycolname = self.dts_tidycolname2
            if data == "cn":
                dictcomb = self.cn_dictcomb_normtot
                dfcomb_tidy = self.cn_dfcomb_normtot_tidy
                tidycolname = self.cn_tidycolname2
            elif data == "g":
                dictcomb = self.g_dictcomb_normtot
                dfcomb_tidy = self.g_dfcomb_normtot_tidy
                tidycolname = self.great_tidycolname2
            elif data == "l":
                dictcomb = self.l_dictcomb_normtot
                dfcomb_tidy = self.l_dfcomb_normtot_tidy
                tidycolname = self.less_tidycolname2

        fvalue, pvalue = stats.f_oneway(
            *[dictcomb[i] for i in self.groups_order])

        print("F value: ", fvalue)
        print("P value: ", pvalue)
        print("\n")
        #  multiple comparisons of means
        tky = pairwise_tukeyhsd(
            endog=dfcomb_tidy[tidycolname],
            groups=dfcomb_tidy[self.tidycolname1],
            alpha=0.05,
        )
        print(tky)

    def anova2(self, df, df_tidy, tidycolname2):
        """Performs anova and tukey hsd post hoc on columns of any df"""

        fvalue, pvalue = stats.f_oneway(*self.cols_to_arrays(df))
        tky = pairwise_tukeyhsd(
            endog=df_tidy[tidycolname2],
            groups=df_tidy[self.tidycolname1],
            alpha=0.05,
        )
        print("F value: ", fvalue)
        print("P value: ", pvalue)
        print("\n")

        print(tky)

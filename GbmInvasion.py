import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# get_ipython().run_line_magic('matplotlib', 'notebook')


class GbmInvasion():
    """
    DTS and CN Data

    Extracts Distance to Surface (DTS) and Cell Number (CN) Data for GBMinvasion experiments.

    Attributes
    ----------
    path : str
        file path to dts files
            Example: directory + "/*_Imaris/*/*_Shortest_Distance_to_Surfaces_Surfaces=Surfaces_2.csv"
    time1 : int
        first timepoint of the experiment (e.g. 2; for 2 days)
    time2 : int
        see time1
    groups_order : list
        list of groups (as strings) in order to be displayed in figures
            Example: ["none_2days", "none_12days", ...]
            Note: only need to list each group once in the correct order
    greater_X : int
        value of threshold of what is considered the "core" for binning
    less_X : int
        value of threshold of what is considered the "surface" for binning


    Methods
    -------
    barplot(self, data, norm, groups, save = None):
        Creates barplots of cell number data.
    scat_comb(self, save = None):
        Creates a scatterplot of data combined by day.
    scat_sep(self, day, save = None):
        Creates scatter plot with individual organoid DTS values.
    scat_comb_withbox(self, save = None):
        Creates a scatterplot of combined data with boxplot.
    dts_violin(self, save = None):
        Creates violin plot of combined data.
    anova(self, data, norm):
        Performs anova and tukey tests on DTS or CN data.
    """

    def __init__(self, path, time1, time2, groups_order, greater_X, less_X):
        """
        Initialize DTS & Cell Number Attributes

        Parameters
        ----------
        path : str
            file path to dts files
                Example: directory + "/*_Imaris/*/*_Shortest_Distance_to_Surfaces_Surfaces=Surfaces_2.csv"
        time1 : int
            first timepoint of the experiment (e.g. 2; for 2 days)
        time2 : int
            see time1
        groups_order : list
            list of groups (as strings) in order to be displayed in figures
                Example: ["none_2days", "none_12days", ...]
                Note: only need to list each group once in the correct order
        greater_X : int
            value of threshold of what is considered the "core" for binning
        less_X : int
            value of threshold of what is considered the "surface" for binning

        """

        # File Path Attributes

        self.path = path  # filepath
        self.filenames = {}  # dict of full filenames and pieces of filenames

        # Defining Variables

        self.time1 = str(time1)
        self.time2 = str(time2)

        self.days = "days"
        self.separator = "_"

        self.tidycolname1 = "Group_Day_Org"
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

        # Color List Attributes for Figures

        self.colors_dict = {}
        self.colors_ordered_all = []
        self.colors_ordered_all_time1 = []
        self.colors_ordered_all_time2 = []

        # Functions

        def unique(l):
            return list(set(l))

        def get_group(s):
            pos = s.index("_")
            try:
                return s[:s.index("_", pos + 1)]
            except ValueError:
                return s[:pos]

        def get_group_only(s):
            pos = s.index("_")
            try:
                return s[:s.index("_", pos + 0)]
            except ValueError:
                return s[:pos]

        def figtofile(fig, path, figname):
            pth = path + "/" + figname + ".pdf"
            fig.get_figure().savefig(pth)

        # === Getting Data for Initialization ===

        # Making tuple of groups based on groups_order
        g = []
        for i in groups_order:
            l = i.split(self.separator)
            g.append(l[0])
        h = []
        [h.append(x) for x in g if x not in h]
        self.groups = tuple(h)

        # === Distance to Surface ===

        n = {}
        for fullname in glob.glob(self.path):
            n.update({fullname: fullname.split(self.separator)})
        self.filenames = n

        dts_dict = {}
        for key, value in self.filenames.items():
            for string in value:
                if string.startswith("org"):
                    org = string
                if string.endswith(self.days):
                    day = string
                if string.startswith(self.groups):
                    group = string
            dts_dict.update(
                {f"{group}{self.separator}{day}{self.separator}{org}": pd.read_csv(
                    key, header=2, usecols=[0])["Shortest Distance to Surfaces"].tolist()}
            )

        for key, value in dts_dict.items():
            dts_dict.update({key: [i for i in value if i >= 0]})

        self.dts_dict = dts_dict

        self.dts_df = pd.DataFrame.from_dict(
            self.dts_dict,
            orient="index",
        ).transpose()

        self.dts_df_tidy = pd.melt(
            self.dts_df,
            var_name=self.tidycolname1,
            value_name=self.dts_tidycolname2
        ).dropna()

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

        n = {}  # temp dict to hold group_day_org: group_day
        for k, v in self.dts_dict.items():
            n.update({k: get_group(k)})
        self.dts_df_tidy[self.tidycolname3] = self.dts_df_tidy[self.tidycolname1].map(
            n)

        columns = list(self.dts_df.columns)
        columns.sort
        self.df_cols = columns

        g = unique(list(map(get_group, columns)))
        g.sort()
        self.dfcomb_cols = g

        # Putting groups in order
        long = [get_group(i) for i in self.dts_df.columns]
        n = []
        for s in self.groups_order:
            for l in long:
                if l == s:
                    n.append(l)
        self.ordered_groups = n
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
        self.ordered_all_time1 = [
            i for i in self.ordered_all if f"{self.separator}{self.time1}{self.days}" in i
        ]
        self.ordered_all_time2 = [
            i for i in self.ordered_all if f"{self.separator}{self.time2}{self.days}" in i
        ]

        # Combining organoids from each day
        dts_groupsdict = {}
        for i in g:
            dts_groupsdict.update({i: ""})
            for key, value in dts_groupsdict.items():
                if key == i:
                    col = [c for c in self.df_cols if c.startswith(i)]
                    dts_groupsdict[key] = self.dts_df[col]

        dcomb = {}
        for key, value in dts_groupsdict.items():
            full = value.values.T.ravel()
            cleaned = [x for x in full if str(x) != "nan"]
            dcomb.update({key: cleaned})
        self.dts_dictcomb = dcomb

        self.dts_dfcomb = pd.DataFrame.from_dict(
            dcomb,
            orient="index",
        ).transpose().reindex(columns=g)

        self.dts_dfcomb_tidy = pd.melt(
            self.dts_dfcomb,
            var_name=self.tidycolname1,
            value_name=self.dts_tidycolname2,
        ).dropna()

        # === Cell Number ===

        self.cn_df = pd.DataFrame(self.dts_df.count()).transpose()
        self.cn_df_tidy = pd.melt(
            self.cn_df,
            var_name=self.tidycolname1,
            value_name=self.cn_tidycolname2,
        ).dropna

        cellnum_groupsdict = {}
        for i in g:
            cellnum_groupsdict.update({i: ""})
            for key, value in cellnum_groupsdict.items():
                if key == i:
                    col = [c for c in self.df_cols if c.startswith(i)]
                    cellnum_groupsdict[key] = self.cn_df[col]

        cncomb = {}
        for key, value in cellnum_groupsdict.items():
            full = value.values.T.ravel()
            cleaned = [x for x in full if str(x) != "nan"]
            cncomb.update({key: cleaned})
        self.cn_dictcomb = cncomb

        self.cn_dfcomb = pd.DataFrame.from_dict(
            cncomb,
            orient="index"
        ).transpose().reindex(columns=g)

        self.cn_dfcomb_tidy = pd.melt(
            self.cn_dfcomb,
            var_name=self.tidycolname1,
            value_name=self.cn_tidycolname2,
        ).dropna()

        # ==== Number of Cells Greater Than gX ===

        gX_dict = {}
        for key, value in self.dts_dict.items():
            count = len([i for i in value if i > greater_X])
            gX_dict.update({key: count})
        self.g_dict = gX_dict

        self.g_df = pd.DataFrame.from_dict(
            gX_dict,
            orient="index"
        ).transpose()

        self.g_df_tidy = pd.melt(
            self.g_df,
            var_name=self.tidycolname1,
            value_name=self.great_tidycolname2
        ).dropna()

        columns = list(self.dts_df.columns)
        columns.sort
        self.df_cols = columns

        g = unique(list(map(get_group, columns)))
        g.sort()
        self.dfcomb_cols = g

        gX_groupsdict = {}
        for i in g:
            gX_groupsdict.update({i: ""})
            for key, value in gX_groupsdict.items():
                if key == i:
                    col = [c for c in self.df_cols if c.startswith(i)]
                    gX_groupsdict[key] = self.g_df[col]

        gcomb = {}
        for key, value in gX_groupsdict.items():
            full = value.values.T.ravel()
            cleaned = [x for x in full if str(x) != "nan"]
            gcomb.update({key: cleaned})
        self.g_dictcomb = gcomb

        self.g_dfcomb = pd.DataFrame.from_dict(
            gcomb,
            orient="index"
        ).transpose().reindex(columns=g)

        self.g_dfcomb_tidy = pd.melt(
            self.g_dfcomb,
            var_name=self.tidycolname1,
            value_name=self.great_tidycolname2
        ).dropna()

        # === Number of Cells Less Than lX: ===

        lX_dict = {}
        for key, value in self.dts_dict.items():
            count = len([i for i in value if i < less_X])
            lX_dict.update({key: count})
        self.l_dict = lX_dict

        self.l_df = pd.DataFrame.from_dict(
            lX_dict,
            orient="index"
        ).transpose()

        self.l_df_tidy = pd.melt(
            self.l_df,
            var_name=self.tidycolname1,
            value_name=self.less_tidycolname2
        ).dropna()

        columns = list(self.dts_df.columns)
        columns.sort
        self.df_cols = columns

        g = unique(list(map(get_group, columns)))
        g.sort()
        self.dfcomb_cols = g

        lX_groupsdict = {}
        for i in g:
            lX_groupsdict.update({i: ""})
            for key, value in lX_groupsdict.items():
                if key == i:
                    col = [c for c in self.df_cols if c.startswith(i)]
                    lX_groupsdict[key] = self.l_df[col]

        lcomb = {}
        for key, value in lX_groupsdict.items():
            full = value.values.T.ravel()
            cleaned = [x for x in full if str(x) != "nan"]
            lcomb.update({key: cleaned})
        self.l_dictcomb = lcomb

        self.l_dfcomb = pd.DataFrame.from_dict(
            lcomb,
            orient="index"
        ).transpose().reindex(columns=g)

        self.l_dfcomb_tidy = pd.melt(
            self.l_dfcomb,
            var_name=self.tidycolname1,
            value_name=self.less_tidycolname2
        ).dropna()

        # === Averages for Normalizations ===
        # Note: these dictionaries contain only the key, value averages for 2 day time points.

        # Total Cell Number

        for k, v in self.cn_dictcomb.items():
            i = sum(v) / len(v)
            self.cn_dictcomb_avg.update({k: i})
        for key in [key for key in self.cn_dictcomb_avg if f"{self.separator}{self.time2}{self.days}" in key]:
            del self.cn_dictcomb_avg[key]

        # Greater than gX:

        for k, v in self.g_dictcomb.items():
            i = sum(v) / len(v)
            self.g_dictcomb_avg.update({k: i})
        for key in [key for key in self.g_dictcomb_avg if f"{self.separator}{self.time2}{self.days}" in key]:
            del self.g_dictcomb_avg[key]

        # Less than lX:

        for k, v in self.l_dictcomb.items():
            i = sum(v) / len(v)
            self.l_dictcomb_avg.update({k: i})
        for key in [key for key in self.l_dictcomb_avg if f"{self.separator}{self.time2}{self.days}" in key]:
            del self.l_dictcomb_avg[key]

        # === Cell Number Normalization ===

        # To Total Cells @ 2 Days:

        for key, val in self.cn_dictcomb_avg.items():
            for k, v in self.cn_dictcomb.items():
                if key.startswith(get_group(k)):
                    self.cn_dictcomb_normtot.update({k: [i / val for i in v]})

        self.cn_dfcomb_normtot = pd.DataFrame.from_dict(
            self.cn_dictcomb_normtot,
            orient="index",
        ).transpose().reindex(columns=g)

        self.cn_dfcomb_normtot_tidy = pd.melt(
            self.cn_dfcomb_normtot,
            var_name=self.tidycolname1,
            value_name=self.cn_tidycolname2,
        ).dropna()

        # To Number of Cells Less than lX:

        for key, val in self.l_dictcomb_avg.items():
            for k, v in self.cn_dictcomb.items():
                if key.startswith(get_group(k)):
                    self.cn_dictcomb_normless.update({k: [i / val for i in v]})

        self.cn_dfcomb_normless = pd.DataFrame.from_dict(
            self.cn_dictcomb_normless,
            orient="index",
        ).transpose().reindex(columns=g)

        self.cn_dfcomb_normless_tidy = pd.melt(
            self.cn_dfcomb_normless,
            var_name=self.tidycolname1,
            value_name=self.cn_tidycolname2,
        ).dropna()

        # === Number Greater Than gX Normalization ===

        # To Total Cells @ 2 days

        for key, val in self.g_dictcomb_avg.items():
            for k, v in self.g_dictcomb.items():
                if key.startswith(get_group(k)):
                    self.g_dictcomb_normtot.update({k: [i / val for i in v]})

        self.g_dfcomb_normtot = pd.DataFrame.from_dict(
            self.g_dictcomb_normtot,
            orient="index"
        ).transpose().reindex(columns=g)

        self.g_dfcomb_normtot_tidy = pd.melt(
            self.g_dfcomb_normtot,
            var_name=self.tidycolname1,
            value_name=self.great_tidycolname2,
        ).dropna()

        # To Number Less Than lX @ 2 Days

        for key, val in self.l_dictcomb_avg.items():
            for k, v in self.g_dictcomb.items():
                if key.startswith(get_group(k)):
                    self.g_dictcomb_normless.update({k: [i / val for i in v]})

        self.g_dfcomb_normless = pd.DataFrame.from_dict(
            self.g_dictcomb_normless,
            orient="index",
        ).transpose().reindex(columns=g)

        self.g_dfcomb_normless_tidy = pd.melt(
            self.g_dfcomb_normless,
            var_name=self.tidycolname1,
            value_name=self.great_tidycolname2,
        ).dropna()

        # === Number of Cell Less Than lX Normalization ===

        # To Total Cells @ 2 Days:

        for key, val in self.cn_dictcomb_avg.items():
            for k, v in self.l_dictcomb.items():
                if key.startswith(get_group(k)):
                    self.l_dictcomb_normtot.update({k: [i / val for i in v]})

        self.l_dfcomb_normtot = pd.DataFrame.from_dict(
            self.l_dictcomb_normtot,
            orient="index",
        ).transpose().reindex(columns=g)

        self.l_dfcomb_normtot_tidy = pd.melt(
            self.l_dfcomb_normtot,
            var_name=self.tidycolname1,
            value_name=self.less_tidycolname2,
        ).dropna()

        # To Number Less than lX @ 2 days:

        for key, val in self.l_dictcomb_avg.items():
            for k, v in self.l_dictcomb.items():
                if key.startswith(get_group(k)):
                    self.l_dictcomb_normless.update({k: [i / val for i in v]})

        self.l_dfcomb_normless = pd.DataFrame.from_dict(
            self.l_dictcomb_normless,
            orient="index",
        ).transpose().reindex(columns=g)

        self.l_dfcomb_normless_tidy = pd.melt(
            self.l_dfcomb_normless,
            var_name=self.tidycolname1,
            value_name=self.less_tidycolname2,
        ).dropna()

        # === Colors and Groups of Colors For Figures ===

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

    # === Figures ===

    def barplot(self, data, norm, groups, save=None):
        """
        Creates barplots of CN data.

        Parameters
        ----------
        data : str
            "cn", "g", or "l"
        norm : str
            "raw", "normtot", or "normless"
        groups : str
            "*days", "*days", "all"
            *days correspond to time1 and time2 (e.g. "2days" or "12days")
        save : list, optional
            [figure_path, figure_name]
            If additional argument is passed then figure will be saved as a
            .pdf file to the directory specified by figure_path with the name
            specified by figure_name

        Returns
        -------
        Seaborn barplot

        """

        yrange1 = 0
        yrange2 = 2.5

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
                data1 = self.cn_dfcomb_normtot
                data2 = self.cn_dfcomb_normtot_tidy
                title = t + "Total # of Cells Norm To Tot"

            elif norm == "normless":
                data1 = self.cn_dfcomb_normless
                data2 = self.cn_dfcomb_normless_tidy
                title = t + "Total # of Cells Norm To Less"

        elif data == "g":
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
            size = [12, 9]
        else:
            size = [6, 4]

        plt.figure(figsize=size)

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

        plot.set_ylabel("Number of Cells", fontsize=20)
        plot.set_xlabel("Group", fontsize=20)
        plot.set_title(title, fontsize=30)

#         plt.ylim(yrange1, yrange2)
        sns.despine()
        plt.tight_layout()

        plt.show()

        if save is None:
            save = []
        else:
            figtofile(plot, save[0], save[1])

    # === Scatterplots ===

    # Combined Groups

    def scat_comb_withbox(self, save=None):
        """
        Creates a scatterplot of combined data with boxplot.

        Parameters
        ----------
        save : list, optional
            [figure_path, figure_name]
            If additional argument is passed then figure will be saved as a
            .pdf file to the directory specified by figure_path with the name
            specified by figure_name

        Returns
        -------
        Seaborn boxplot overlayed with seaborn stripplot.

        """

        data1 = self.dts_dfcomb_tidy

        if save is None:
            size = [8, 6]
        else:
            size = [10, 8]

        plt.figure(figsize=size)

        plot = sns.boxplot(
            x=self.tidycolname1,
            y=self.dts_tidycolname2,
            data=data1,
            order=self.groups_order,
            fliersize=0,
            palette="Paired",
        )

        plot = sns.stripplot(
            x=self.tidycolname1,
            y=self.dts_tidycolname2,
            data=data1,
            order=self.groups_order,
            color="grey",
            size=2,
            linewidth=0,
            jitter=0.25,
        )

        plot.set_xlabel("Group", fontsize=20)
        plot.set_ylabel("Distance to Surface", fontsize=20)
        plot.set_title("Distance to Surface", fontsize=30)
        sns.despine()
        plt.tight_layout()

        plt.show()

        if save is None:
            save = []
        else:
            figtofile(plot, save[0], save[1])

    def scat_comb(self, save=None):
        """
        Creates a scatterplot of data combined by day.

        Parameters
        ----------
        save : list, optional
            [figure_path, figure_name]
            If additional argument is passed then figure will be saved as a
            .pdf file to the directory specified by figure_path with the name
            specified by figure_name


        Returns
        -------
        Seaborn boxplot overlayed with seaborn stripplot.

        """

        data1 = self.dts_dfcomb_tidy

        if save is None:
            size = [12.5, 12.5]
        else:
            size = [7.5, 9]

        plt.figure(figsize=size)

        plot = sns.stripplot(
            x=self.dts_tidycolname2,
            y=self.tidycolname1,
            data=data1,
            order=self.groups_order,
            orient="h",
            palette="Paired",
            size=2.5,
            linewidth=0,
            jitter=0.25,
        )

        plot.set_xlabel("Distance to Surface", fontsize=20)
        plot.set_ylabel("Group", fontsize=20)
        plot.set_title("Distance to Surface", fontsize=30)
        sns.despine()
        plt.tight_layout()

        plt.show()

        if save is None:
            save = []
        else:
            figtofile(plot, save[0], save[1])

    def scat_sep(self, day, save=None):
        """
        Creates scatter plot with individual organoid DTS values.


        Parameters
        ----------
        day : str
            "all", "*days"
            *days corresponds to time1 or time2 (e.g. "2days" or "12days")
        save : list, optional
            [figure_path, figure_name]
            If additional argument is passed then figure will be saved as a
            .pdf file to the directory specified by figure_path with the name
            specified by figure_name


        Returns
        -------
        Seaborn stripplot.

        """

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

        if save is None:
            size = [12.5, 12.5]
        else:
            size = [7.5, 9]

        plt.figure(figsize=size)

        plot = sns.stripplot(
            x=self.dts_tidycolname2,
            y=self.tidycolname1,
            data=self.dts_df_tidy,
            order=order,
            orient="h",
            palette=colors,
            size=2.5,
            linewidth=0,
            jitter=0.25,
        )

        plot.set_xlabel("Distance to Surface", fontsize=20)
        plot.set_ylabel("Groups", fontsize=20)
        plot.set_title(t, fontsize=30)

        plt.tight_layout()
        sns.despine()

        plt.show()

        if save is None:
            save = []
        else:
            figtofile(plot, save[0], save[1])

    # === Violin Plot ===

    def dts_violin(self, save=None):
        """
        Creates a violin plot of data combined by day.

        Parameters
        ----------
        save : list, optional
            [figure_path, figure_name]
            If additional argument is passed then figure will be saved as a
            .pdf file to the directory specified by figure_path with the name
            specified by figure_name


        Returns
        -------
        Seaborn violin plot.

        """

        plt.figure(figsize=[9.5, 5])

        plot = sns.violinplot(
            x=self.tidycolname1,
            y=self.dts_tidycolname2,
            data=self.dts_dfcomb_tidy,
            order=order,
            width=1,
            bw=0.1,
            palette="Paired",
            ci=68,
        )

        plot.set_xlabel("Group", fontsize=20)
        plot.set_ylabel("Distance to Surface", fontsize=20)
        plot.set_title("Distance to Surface", fontsize=30)
        sns.despine()

        plt.show()

        if save is None:
            save = []
        else:
            figtofile(plot, save[0], save[1])

    # === Statistics ===

    def anova(self, data, norm):
        """
        Performs anova and tukey tests on DTS or CN data.

        Parameters
        ----------
        data : str
            "dts", "cn", "g", "l"
        norm : str
            "raw", "normtot", "normless"

        Returns
        -------
        F value and P value from one-way anova.
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

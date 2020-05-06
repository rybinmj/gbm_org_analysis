import pandas as pd
import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from statistics import mean
from statsmodels.stats.multicomp import pairwise_tukeyhsd


class GbmInvasion2():
    """
    DTS and CN Data

    Extracts Distance to Surface (DTS) and Cell Number (CN) Data for GBMinvasion experiments.

    Attributes
    ----------
    path : str
        file path to dts files
            Example: directory + "/*_Imaris/*/*_Shortest_Distance_to_Surfaces_Surfaces=Surfaces_2.csv"
    _orderered_groups : list
        list of groups (as strings) in order to be displayed in figures
            Example: ["none_2days", "none_12days", ...]
            Note: only need to list each group once in the correct order

    Methods
    -------

    """

    def __init__(self, path, ordered_groups, invasion_quantile=0.75):
        """
        Initialize DTS & Cell Number Attributes

        Parameters
        ----------
        path : str
            file path to dts files
                Example: directory + "/*_Imaris/*/*_Shortest_Distance_to_Surfaces_Surfaces=Surfaces_2.csv"
        ordered_groups : list
            list of groups (as strings) in order to be displayed in figures
                Example: ["none_2days", "none_12days", ...]
                Note: only need to list each group once in the correct order

        """

        # File Path Attributes

        self.dts_path = path + \
            "/*_Imaris/Batch*/*/*_Shortest_Distance_to_Surfaces_Surfaces=Surfaces_2.csv"
        self.spotintensity_path = path + \
            "/*_Imaris/Batch*/*/*_Intensity_Sum_Ch=2_Img=1.csv"
        self.xyz_path = path + "/*_Imaris/Batch*/*/*_Position.csv"
        self.vol_path = path + "/*_Imaris/Batch*/*/*_vol_Volume.csv"
        self.surfarea_path = path + "/*_Imaris/Batch*/*/*_vol_Area.csv"
        self.fullvol_path = path + \
            "/200210_GBM_NagiCombos_4_Imaris/Batch2_GBM22/" \
            "200407_gbm22_batch2_group0_org1_fullvol_Statistics/" \
            "200407_gbm22_batch2_group0_org1_fullvol_Volume.csv"
        self.fullarea_path = path + \
            "/200210_GBM_NagiCombos_4_Imaris/Batch2_GBM22/" \
            "200407_gbm22_batch2_group0_org1_fullvol_Statistics/" \
            "200407_gbm22_batch2_group0_org1_fullvol_Area.csv"

        # Defining Variables

        self.ordered_groups = ordered_groups
        self.invasion_quantile = invasion_quantile
        self.separator = "_"

        self.tidycolname1 = "Org_or_Group"
        self.dts_tidycolname2 = "DTS"
        self.cn_tidycolname2 = "Cell_Number"
        self.g_tidycolname2 = "Num_Greater_Than"
        self.sa_tidycolname2 = "Surface_Area"
        self.vol_tidycolname2 = "Organoid_Volume"
        self.inten_tidycolname2 = "Spot_Intensity"
        self.tidycolname3 = "Group"

        #########################################
        # === Getting Data for Initialization ===
        #########################################

        # Extracting data into dictionary
        self.dts_dict_withneg = self.extract_data(self.dts_path)
        self.xyz_dict_withneg = self.extract_data(self.xyz_path)
        self.inten_dict_withneg = self.extract_data(self.spotintensity_path)

        self.vol_dict = self.extract_data(self.vol_path)
        self.sa_dict = self.extract_data(self.surfarea_path)
        self.fullvol = self.extract_data(self.fullvol_path)
        self.fullarea = self.extract_data(self.fullarea_path)

        # Ordered list of all organoids
        self.unordered_all = list(self.dts_dict_withneg.keys())
        self.ordered_all = []
        for group_id in self.ordered_groups:
            orgs_in_group = [
                org_id for org_id in self.unordered_all if org_id[:-5] == group_id]
            self.ordered_all = self.ordered_all + orgs_in_group

        #############################
        # === Distance to Surface ===
        #############################

        # Making dts dfs with negative values included
        self.dts_df_withneg = self.mk_df(
            self.dts_dict_withneg, self.ordered_all)
        self.dts_df_tidy_withneg = self.mk_tidy(
            self.dts_df_withneg, self.dts_tidycolname2)

        # Ordering xyz and dts data into dfs
        self.dfs = {}
        for k, v in self.xyz_dict_withneg.items():
            if k not in self.dfs:
                self.dfs[k] = pd.concat(
                    [v, self.dts_df_withneg[k]], axis=1).dropna()

        # Removing negative values
        for df in self.dfs.values():
            for col in df.columns:
                index = df[df[col] < 0].index
                df.drop(index, inplace=True)

        for org, df in self.dfs.items():
            df.index = np.arange(1, len(df) + 1)
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'cell'}, inplace=True)

        self.dts_dict = {}
        for key, value in self.dts_dict_withneg.items():
            if key not in self.dts_dict:
                self.dts_dict[key] = [i for i in value if i >= 0]

        self.dts_df = self.mk_df(self.dts_dict, self.ordered_all)
        self.dts_df_tidy = self.mk_tidy(self.dts_df, self.dts_tidycolname2)

        # Df of dts data for all organoids combined by group
        self.dts_dfcomb = self.mk_dfcomb(self.dts_df)
        self.dts_dfcomb_tidy = self.mk_tidy(
            self.dts_dfcomb, self.dts_tidycolname2)

        self.batch_names = [
            "gbm39_day18",
            "gbm22_day18",
            "gbm39_day25",
            "gbm22_day11",
        ]
        self.batch_dfs = self.split_batches(self.dts_df)
        self.batch_dfscomb = [self.mk_dfcomb(
            batch_df) for batch_df in self.batch_dfs]

        self.batch_dfs_tidy = [self.mk_tidy(
            batch_df, self.dts_tidycolname2) for batch_df in self.batch_dfs]
        self.batch_dfscomb_tidy = [self.mk_tidy(
            batch_dfcomb, self.dts_tidycolname2) for batch_dfcomb in self.batch_dfscomb]

        # Making spot intensity dfs
        self.inten_df_withneg = self.mk_df(
            self.inten_dict_withneg, self.ordered_all)
        self.inten_df_tidy_withneg = self.mk_tidy(
            self.inten_df_withneg, self.inten_tidycolname2)

        # Combining tidy dts df and tidy intensity df to prepare to remove negative values
        self.dts_inten_df_tidy_withneg = self.dts_df_tidy_withneg
        self.dts_inten_df_tidy_withneg["Spot_Intensity"] = self.inten_df_tidy_withneg["Spot_Intensity"]

        # Removing negative values from dts-intensity df
        negative_indexes = self.dts_inten_df_tidy_withneg.query(
            "DTS < 0").index.tolist()
        self.dts_inten_df_tidy = self.dts_inten_df_tidy_withneg.drop(
            negative_indexes).reset_index(drop=True)

        self.batch_inten_tidy = self.split_batches_tidy(self.dts_inten_df_tidy)

        # === Norm DTS to first value ===

        self.dts_df_normfirst = self.dts_df.apply(
            lambda col_values: col_values - col_values.min())
        self.dts_df_tidy_normfirst = self.mk_tidy(
            self.dts_df_normfirst, self.dts_tidycolname2)

        self.dts_dfcomb_normfirst = self.dts_dfcomb.apply(
            lambda col_values: col_values - col_values.min())
        self.dts_dfcomb_tidy_normfirst = self.mk_tidy(
            self.dts_dfcomb_normfirst, self.dts_tidycolname2)

        # Batch dfs norm to first value
        self.batch_dfs_normfirst = self.split_batches(self.dts_df_normfirst)
        self.batch_dfscomb_normfirst = [self.mk_dfcomb(
            batch_df) for batch_df in self.batch_dfs_normfirst]

        self.batch_dfs_tidy_normfirst = [self.mk_tidy(
            batch_df, self.dts_tidycolname2) for batch_df in self.batch_dfs_normfirst]
        self.batch_dfscomb_tidy_normfirst = [self.mk_tidy(
            batch_dfcomb, self.dts_tidycolname2) for batch_dfcomb in self.batch_dfscomb_normfirst]

        #####################
        # === Cell Number ===
        #####################

        self.cn_df = pd.DataFrame(self.dts_df.count()).transpose()
        self.cn_df_tidy = self.mk_tidy(self.cn_df, self.cn_tidycolname2)

        self.cn_dfcomb = self.mk_dfcomb(self.cn_df)
        self.cn_dfcomb_tidy = self.mk_tidy(
            self.cn_dfcomb, self.cn_tidycolname2)

        self.batch_cns = self.split_batches(self.cn_df)
        self.batch_cnscomb = [self.mk_dfcomb(
            batch_cn) for batch_cn in self.batch_cns]

        self.batch_cns_tidy = [self.mk_tidy(
            batch_cn, self.cn_tidycolname2) for batch_cn in self.batch_cns]
        self.batch_cnscomb_tidy = [self.mk_tidy(
            batch_cncomb, self.cn_tidycolname2) for batch_cncomb in self.batch_cnscomb]

        ################################
        # === Greater than threshold ===
        ################################

        # Listing invasion thresholds by batch
        self.invasion_thresholds = [
            self.get_invasion_threshold(df) for df in self.batch_dfs_normfirst]

        # Getting number of cells above threshold
        self.batch_gs_number = []
        for batch in self.batch_dfs_normfirst:
            batch_g = self.mk_df_greaterthan(batch)
            self.batch_gs_number.append(batch_g)

        # Combining and making tidy dfs
        self.batch_gs_numbercomb = [self.mk_dfcomb(
            batch_g) for batch_g in self.batch_gs_number]
        self.batch_gs_number_tidy = [self.mk_tidy(
            batch_g, self.g_tidycolname2) for batch_g in self.batch_gs_number]
        self.batch_gs_numbercomb_tidy = [self.mk_tidy(
            batch_gcomb, self.g_tidycolname2) for batch_gcomb in self.batch_gs_numbercomb]

        # Normalizing number above to total number to return fraction above threshold
        self.batch_gs = []
        for i in range(4):
            gs = self.batch_gs_number[i] / self.batch_cns[i]
            self.batch_gs.append(gs)

        # Combining and making tidy dfs
        self.batch_gscomb = [self.mk_dfcomb(
            batch_g) for batch_g in self.batch_gs]
        self.batch_gs_tidy = [self.mk_tidy(
            batch_g, self.g_tidycolname2) for batch_g in self.batch_gs]
        self.batch_gscomb_tidy = [self.mk_tidy(
            batch_gcomb, self.g_tidycolname2) for batch_gcomb in self.batch_gscomb]

        ##########################################
        # === Surf area and vol stats for orgs ===
        ##########################################

        # Surface area
        for org, sa_value in self.sa_dict.items():
            new_value = sa_value[0] - self.fullarea
            self.sa_dict[org] = [new_value]
        self.sa_df = self.mk_df(self.sa_dict, self.ordered_all)
        self.sa_df_tidy = self.mk_tidy(self.sa_df, self.sa_tidycolname2)

        self.sa_dfcomb = self.mk_dfcomb(self.sa_df)
        self.sa_dfcomb_tidy = self.mk_tidy(
            self.sa_dfcomb, self.sa_tidycolname2)

        self.batch_sas = self.split_batches(self.sa_df)
        self.batch_sascomb = [self.mk_dfcomb(
            batch_sa) for batch_sa in self.batch_sas]

        self.batch_sas_tidy = [self.mk_tidy(
            batch_sa, self.sa_tidycolname2) for batch_sa in self.batch_sas]
        self.batch_sascomb_tidy = [self.mk_tidy(
            batch_sacomb, self.sa_tidycolname2) for batch_sacomb in self.batch_sascomb]

        # Volume
        for org, vol_value in self.vol_dict.items():
            new_value = self.fullvol - vol_value[0]
            self.vol_dict[org] = [new_value]
        self.vol_df = self.mk_df(self.vol_dict, self.ordered_all)
        self.vol_df_tidy = self.mk_tidy(self.vol_df, self.vol_tidycolname2)

        self.vol_dfcomb = self.mk_dfcomb(self.vol_df)
        self.vol_dfcomb_tidy = self.mk_tidy(
            self.vol_dfcomb, self.vol_tidycolname2)

        self.batch_vols = self.split_batches(self.vol_df)
        self.batch_volscomb = [self.mk_dfcomb(
            batch_vol) for batch_vol in self.batch_vols]

        self.batch_vols_tidy = [self.mk_tidy(
            batch_vol, self.vol_tidycolname2) for batch_vol in self.batch_vols]
        self.batch_volscomb_tidy = [self.mk_tidy(
            batch_volcomb, self.vol_tidycolname2) for batch_volcomb in self.batch_volscomb]

        #####################################
        # === Colors By Group For Figures ===
        #####################################

        muted = sns.color_palette("muted").as_hex()
        dark = sns.color_palette("dark").as_hex()
        pastel = sns.color_palette("pastel").as_hex()
        colorblind = sns.color_palette("colorblind").as_hex()
        deep = sns.color_palette("deep").as_hex()

        self.group_colors = [
            pastel[7],
            colorblind[7],
            muted[4],
            muted[3],
            dark[3],
            muted[0],
            dark[0],
            muted[2],
            dark[2],
        ]

        self.batch_groups = []
        self.orgs_in_groups = []
        for batch in self.batch_names:
            batch_group = [
                group for group in self.ordered_groups if group.startswith(batch)]
            self.batch_groups.append(batch_group)
            lst_of_orgs = [
                org for org in self.ordered_all if org.startswith(batch)]
            self.orgs_in_groups.append(lst_of_orgs)

        self.batch_colors = []
        for batch in self.batch_groups:
            color_dict = dict(zip(batch, self.group_colors))
            org_colors = []
            for group_id, color in color_dict.items():
                for org in self.ordered_all:
                    if org[:-5] == group_id:
                        org_colors.append(color)
            self.batch_colors.append(org_colors)
    ###################
    # === Functions ===
    ###################

    def extract_data(self, path):
        n = {}
        for fullname in glob.glob(path):
            # if "vol" in fullname:
            #     continue
            n.update({fullname: fullname.split(self.separator)})

        dct = {}
        for key, value in n.items():
            for string in value:
                if string.startswith("org"):
                    org = string
                if string.startswith("batch"):
                    if string.endswith("1"):
                        batch = "gbm39_day18"
                    if string.endswith("2"):
                        batch = "gbm22_day18"
                    if string.endswith("3"):
                        batch = "gbm39_day25"
                    if string.endswith("4"):
                        batch = "gbm22_day11"
                if string.startswith("group"):
                    if string.endswith("0"):
                        group = "day1-none"
                    if string.endswith("1"):
                        group = "TAS120"
                    if string.endswith("2"):
                        group = "JQ1"
                    if string.endswith("3"):
                        group = "MK8628"
                    if string.endswith("4"):
                        group = "NM002"
                    if string.endswith("5"):
                        group = "TAS120+JQ1"
                    if string.endswith("6"):
                        group = "TAS120+MK8628"
                    if string.endswith("7"):
                        group = "TAS120+NM002"
                    if string.endswith("8"):
                        group = "DMSO"

            org_identifier = f"{batch}{self.separator}{group}{self.separator}{org}"

            if path == self.dts_path:
                if "vol" not in key:
                    data = pd.read_csv(key, header=2, usecols=[0])[
                        "Shortest Distance to Surfaces"].tolist()
                    if org_identifier not in dct:
                        dct[org_identifier] = data
            elif path == self.xyz_path:
                if "vol" not in key:
                    data = pd.read_csv(key, header=2, usecols=[0, 1, 2])
                    data = data.rename(
                        columns={
                            "Position X": "X_" + org_identifier,
                            "Position Y": "Y_" + org_identifier,
                            "Position Z": "Z_" + org_identifier,
                        }
                    )
                    if org_identifier not in dct:
                        dct[org_identifier] = data
            elif path == self.spotintensity_path:
                if "vol" not in key:
                    data = pd.read_csv(key, header=2, usecols=[0])[
                        "Intensity Sum"].tolist()

                    if org_identifier not in dct:
                        dct[org_identifier] = data
            elif path == self.vol_path:
                data = pd.read_csv(key, header=2, usecols=[0])[
                    "Volume"].tolist()
                if len(data) > 1:
                    data = [max(data)]
                if org_identifier not in dct:
                    dct[org_identifier] = data
            elif path == self.surfarea_path:
                data = pd.read_csv(key, header=2, usecols=[0])["Area"].tolist()
                if len(data) > 1:
                    data = [max(data)]
                if org_identifier not in dct:
                    dct[org_identifier] = data
            elif path == self.fullvol_path:
                data = pd.read_csv(key, header=2, usecols=[0])[
                    "Volume"].tolist()
                return float(data[0])
                break
                if org_identifier not in dct:
                    dct[org_identifier] = data
            elif path == self.fullarea_path:
                data = pd.read_csv(key, header=2, usecols=[0])[
                    "Area"].tolist()
                return float(data[0])
                break
                if org_identifier not in dct:
                    dct[org_identifier] = data

        return dct

    def mk_df(self, dct, columns):
        df = pd.DataFrame.from_dict(
            dct, orient="index").transpose().reindex(columns=columns)
        return df

    def mk_tidy(self, df, tidycolname2):
        """Transforms a wide form df to long form. Uses DTS as tidycolname2"""
        df_tidy = pd.melt(df, var_name=self.tidycolname1,
                          value_name=tidycolname2).dropna()
        return df_tidy

    def mk_dfcomb(self, df):
        """Making combined dataframe from individuals dfs"""
        combined_dict = {}
        for group_id in self.ordered_groups:
            keep_cols = [c for c in df.columns.tolist() if c[:-5] == group_id]
            combined_values = []
            for organoid in keep_cols:
                combined_values.append(df[organoid].values.tolist())
            flat_list = [
                item for sublist in combined_values for item in sublist]

            if group_id not in combined_dict:
                if flat_list == []:
                    continue
                combined_dict[group_id] = flat_list

        dfcomb = self.mk_df(combined_dict, combined_dict.keys())
        return dfcomb

    def split_batches(self, df):
        """Spliting dataframe into individual dataframes by batches"""

        batch_dfs = []
        for batch in self.batch_names:
            lst_of_orgs = [
                org for org in self.ordered_all if org.startswith(batch)]
            batch_dfs.append(df[lst_of_orgs])
        return batch_dfs

    def split_batches_tidy(self, df):
        """Spliting dataframe into individual dataframes by batches"""
        batch_dfs = []
        for batch in self.batch_names:
            batch_df = df[df["Org_or_Group"].str.startswith(
                batch)].reset_index(drop=True)
            batch_dfs.append(batch_df)
        return batch_dfs

    def cols_to_arrays(self, df):
        """
        Transforms the columns of a df into a list of arrays.

        Useful for preparing argument for anova functions (stats.f_oneway) which takes array-like
        arguments. Can use *cols_to_arrays(df) to perform anova on columns of df.
        """

        l = []
        length = len(df.columns)
        for i in range(length):
            array = np.array(df.iloc[:, i])
            array = array[~np.isnan(array)]
            l.append(array)
        return l

    def get_invasion_threshold(self, df):
        """Returns a invasion threshold of a given df based
        on the average invasion_quantile value of the DMSO group."""
        thresh_values = dict(df.quantile(self.invasion_quantile))
        values_to_average = []
        for org, thresh_value in thresh_values.items():
            if "DMSO" in org:
                values_to_average.append(thresh_value)
        invasion_threshold = mean(values_to_average)
        return invasion_threshold

    def mk_df_greaterthan(self, df):
        invasion_threshold = self.get_invasion_threshold(df)
        g_df = pd.DataFrame()
        for col, value in df.items():
            count = len([i for i in list(value) if i > invasion_threshold])
            g_df[col] = [count]
        return g_df

    def bar(self, df, df_tidy, yrange=[0, 1000], ylabel="Number of Cells", size=[15, 8], save=False):

        #         order = [
        #             group_id for group_id in self.ordered_groups if group_id.startswith(batch)
        #         ]
        colors = self.group_colors

        rc = {
            'font.size': 20.0,
            'axes.labelsize': 30.0,
            'axes.titlesize': 'large',
            'xtick.labelsize': 20.0,
            'ytick.labelsize': 20.0,
            'legend.fontsize': 'medium'
        }

        plt.figure(figsize=size)
        plt.rcParams.update(**rc)

        plot = sns.barplot(
            data=df,
            #             order=order,
            palette=colors,
            linewidth=2.5,
            orient="v",
            capsize=0.1,
            ci=68,
            errcolor="black",
        )

        plot = sns.stripplot(
            x=df_tidy.columns[0],
            y=df_tidy.columns[1],
            data=df_tidy,
            #             order=order,
            color="white",
            size=10,
            edgecolor="black",
            linewidth=1,
            jitter=0.1,
            orient="v",
        )

        plot.set_ylabel(ylabel)
        plot.set_xlabel("")

        sns.despine()
        plt.tight_layout()
        plt.ylim(yrange[0], yrange[1])

        plt.show()

        if save is not False:
            self.figtofile(plot, save[0], save[1])

    def scatter(self, batch, norm=True, x_range=None, save=None):

        if batch == "gbm39_day18":
            data_tidy = self.batch_dfs_tidy_normfirst[0]
            colors = self.batch_colors[0]
        elif batch == "gbm22_day18":
            data_tidy = self.batch_dfs_tidy_normfirst[1]
            colors = self.batch_colors[1]
        elif batch == "gbm39_day25":
            data_tidy = self.batch_dfs_tidy_normfirst[2]
            colors = self.batch_colors[2]
        elif batch == "gbm22_day11":
            data_tidy = self.batch_dfs_tidy_normfirst[3]
            colors = self.batch_colors[3]

        if norm is not True:
            if batch == "gbm39_day18":
                data_tidy = self.batch_dfs_tidy[0]
                colors = self.batch_colors[0]
            elif batch == "gbm22_day18":
                data_tidy = self.batch_dfs_tidy[1]
                colors = self.batch_colors[1]
            elif batch == "gbm39_day25":
                data_tidy = self.batch_dfs_tidy[2]
                colors = self.batch_colors[2]
            elif batch == "gbm22_day11":
                data_tidy = self.batch_dfs_tidy[3]
                colors = self.batch_colors[3]

        order = [
            org_id for org_id in self.ordered_all if org_id.startswith(batch)]

        size = [12.5, 12.5]

        rc_for_scatter = {
            'font.size': 20.0,
            'axes.labelsize': 30.0,
            'axes.titlesize': 'large',
            'xtick.labelsize': 15.0,
            'ytick.labelsize': 15.0,
            'legend.fontsize': 'medium'
        }

        fig_style = {
            "figure.facecolor": "w",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }

        plt.figure(figsize=size)
        plt.rcParams.update(**rc_for_scatter)

        plot = sns.boxplot(
            x=data_tidy.columns[1],
            y=self.tidycolname1,
            data=data_tidy,
            order=order,
            orient="h",
            fliersize=0,
            palette=colors,
        )

        plot = sns.stripplot(
            x=data_tidy.columns[1],
            y=self.tidycolname1,
            data=data_tidy,
            order=order,
            orient="h",
            palette=colors,
            size=2.5,
            linewidth=0,
            jitter=0.25,
        )

        plot.set_xlabel("Distance to Surface (µm)")
        plot.set_ylabel("Groups")
        #         plot.set_title(t, fontsize=30)
        sns.set_context(
            font_scale=1.75,
            rc=fig_style
        )

        plt.tight_layout()
        if x_range is not None:
            plt.xlim(x_range[0], x_range[1])
        sns.despine()

        plt.show()

        if save is not None:
            self.figtofile(plot, save[0], save[1])

    def anova(self, df, df_tidy):
        """Performs anova and tukey hsd post hoc on columns of any df"""

        fvalue, pvalue = stats.f_oneway(*self.cols_to_arrays(df))
        tky = pairwise_tukeyhsd(
            endog=df_tidy[df_tidy.columns[1]],
            groups=df_tidy[df_tidy.columns[0]],
            alpha=0.05,
        )
        print("F value: ", fvalue)
        print("P value: ", pvalue)
        print("\n")
        print(tky)

    def figtofile(self, fig, path, figname):
        """Saves figure as .pdf file"""
        pth = path + "/" + figname + ".pdf"
        fig.get_figure().savefig(pth)

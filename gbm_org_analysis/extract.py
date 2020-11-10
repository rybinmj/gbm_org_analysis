import pandas as pd
import glob
import os
from statistics import mean
import matplotlib.pyplot as plt
from gbm_org_analysis import stats, figs


def match_files_with_org_id(path, separator, groups_dict):
    """Returns a ditionary of the full file path : organoid identifier with group_org#"""

    name_pairs = []
    for fullname in glob.glob(path):
        name_pair = (fullname, fullname.split(separator))
        if any(portion_of_filename in name_pair[1] for portion_of_filename in list(groups_dict.keys())):
            name_pairs.append(name_pair)

    files_with_org_ids = {}
    for name_pair in name_pairs:
        full_filename = name_pair[0]
        name_split = name_pair[1]

        for string in name_split:
            if string.startswith("org"):
                org = string
            if string.startswith("group"):
                for group_num, group_name in groups_dict.items():
                    if group_num == string:
                        group = group_name

        org_identifier = f"{group}{separator}{org}"

        if full_filename not in files_with_org_ids:
            files_with_org_ids[full_filename] = org_identifier

    return files_with_org_ids


def comb_df_by_group(df, group_names):
    """Combines columns of a given df whose names start with strings in group_names"""
    all_dict = {}
    for group in group_names:
        keep_cols = [col for col in df if col.startswith(group)]
        combined_values = []
        for col in keep_cols:
            combined_values.append(df[col].values.tolist())
        collapsed_list = [
            item for sublist in combined_values for item in sublist]
        if group not in all_dict:
            all_dict[group] = collapsed_list

    df_comb = pd.DataFrame(
        dict([(k, pd.Series(v))for k, v in all_dict.items()]))

    return df_comb


def count_values_above_threshold(df, threshold):
    """
    Returns new df containing a count of the number
    of values that exceed the given threshold.
    """
    new_df = pd.DataFrame()
    for col_name, values in df.items():
        count = len([value for value in list(values) if value > threshold])
        new_df[col_name] = [count]
    return new_df


def mk_comb_and_tidy_dfs(df, tidycol2name):
    groups = list(dict.fromkeys([col[:-5] for col in df.columns]))

    df_bygroup = comb_df_by_group(df, groups)
    df_tidy = pd.melt(
        df, var_name='Org', value_name=tidycol2name).dropna()
    df_bygroup_tidy = pd.melt(
        df_bygroup, var_name='Group', value_name=tidycol2name).dropna()

    return df_bygroup, df_tidy, df_bygroup_tidy


class GbmCellData():
    """
    Extracts values from Imaris data corresponding to
    the distance of individual gbm cells to surface of the organoid
    (i.e "distance to surface" or "DTS")
    """

    def __init__(
        self,
        groups,
        file_location="Data_Raw/*/*_org?_Shortest_Distance_to_Surfaces_Surfaces*.csv",
        invasion_threshold_details=[0.75, 'DMSO'],
        export_data=False,
        export_data_location="Data_Processed/",
        export_stats=False,
        export_stats_location="Statistics/",
        figures=None,
        figure_colors=None,
        export_figs_location="Figures_Original/",
    ):
        """
        Extracts number of cells and distance to surface values
        from data exported from Imaris.

        Processes the data and shapes into pandas dataframes;
        provides the option of exporting dataframes to excel documents.

        Runs a set of statistical tests and provides the option
        to save the results to .txt files.

        Creates figures (bargraphs, boxplots, linear regression) of the data
        and allows to display or to save as .pdf files.

        Parameters
        ----------
        groups : dict
            Key to reveal blinded groups. Should be in order desired for figures.
            Example: {'group0': 'invasion_ctl', group1': 'DMSO'}
        file_location : str, optional
            Representative location of distance to surface.csv files from Imaris.
            Use glob wildcard charaters to allow extraction of all distance
            to surface data files within the directory.
        invasion_threshold_details : list, optional
            First value in list should be a float corresponding to
            the quantile to be averaged for the invasion threshold.
            Second value in the list should be a string corresponding to
            the group whose quantile values should be averaged to generate
            the invasion threshold.
            Default set to the 0.75 quantile of the DMSO group
            (i.e. [0.75, 'DMSO'])
        export_data : bool, optional
            If true exports processed dataframes to excel files.
            Default value false.
        export_data_location : str, optional
            Location where exported excel files will be stored.
            Default is "Data_Processed/" (a subdirectory of parent experiment directory).
        export_stats : bool, optional
            If true saves results of statistical tests as .txt files.
            Default value false.
        export_stats_location : str, optional
            Location of saved statistical results files.
            Default is "Statistics/".
        figures : str, optional
            If 'show' is passed: displays figures without saving.
            If 'save' is passed: saves figures as PDFs without displaying.
            If None is passed (default): does not generate, display or save figures.
        figure_colors : list, optional
            Specifies colors to be used for groups in the figures.
            When None is passed (default) uses default seaborn colorscheme.
            Use seaborn color palettes to specify different colors.
            Example: sns.color_palette("muted").as_hex()[4] is '#956cb4' which
            is light purple.
        export_figs_location : str, optional
            Location of saved figures.
            Default is "Figures_Original/"

        Notes
        -----
        Three groups of data generated abreviated as "dts" for distance to surface,
        "cn" for total number of cells and "inv" for invaded cells or cells whose
        distance exceeds the invasion threshold.

        To access dfs assocaited with each type of data use df, df_bygroup, df_tidy,
        or df_bygroup_tidy (where df is replaced with data abbreviations). Note that
        this naming scheme is replicated in the different sheets of the exported excel
        files if export_data=True.

        """

        # ===== DEFINING VARIABLES =====

        self.groups_dict = groups
        self.ordered_group_names = list(groups.values())
        self.invasion_thresh_quantile = invasion_threshold_details[0]
        self.invasion_thresh_group = invasion_threshold_details[1]
        self.separator = "_"

        # ===== DATAFRAMES OF DISTANCE TO SURFACE DATA =====

        # Extracting distance to surface data into a dfs
        self.filenames_dict = match_files_with_org_id(
            file_location, self.separator, self.groups_dict)
        self.dts = self.extract_dts_data(self.filenames_dict)

        # Making bygroup and tidyform dfs
        self.dts_bygroup, self.dts_tidy, self.dts_bygroup_tidy = \
            mk_comb_and_tidy_dfs(self.dts, 'DTS')

        # ===== DATAFRAMES OF TOTAL NUMBER OF CELLS DATA =====

        self.cn = pd.DataFrame()
        for col in self.dts.columns.tolist():
            self.cn[col] = [self.dts[col].count()]

        self.cn_bygroup, self.cn_tidy, self.cn_bygroup_tidy = \
            mk_comb_and_tidy_dfs(self.cn, 'NumOfCells')

        # ===== DATAFRAMES OF NUMBER OF CELLS GREATER THAN INVASION THRESHOLD DATA =====

        # Specify invasion threshold
        self.inv_thresh = self.get_invasion_threshold(self.dts)

        # Creating df of the number of cells whose dts values exceed invasion threshold
        self.inv = count_values_above_threshold(
            self.dts, self.inv_thresh)

        # Making bygroup and tidyform dfs
        self.inv_bygroup, self.inv_tidy, self.inv_bygroup_tidy = \
            mk_comb_and_tidy_dfs(self.inv, 'NumGreatThanThresh')

        # ===== EXPORT DTS, CN, AND INV DATA TO EXCEL FILES =====

        if export_data is True:
            if export_data_location == 'Data_Processed/':
                if 'Data_Processed' not in os.listdir():
                    os.mkdir('Data_Processed')

            # Distance to surface data
            dts_file = pd.ExcelWriter(export_data_location + 'dts.xlsx')
            self.dts.to_excel(dts_file, sheet_name='dts', index=False)
            self.dts_tidy.to_excel(
                dts_file, sheet_name='dts_tidy', index=False)
            self.dts_bygroup.to_excel(
                dts_file, sheet_name='dts_bygroup', index=False)
            self.dts_bygroup_tidy.to_excel(
                dts_file, sheet_name='dts_bygroup_tidy', index=False)
            dts_file.save()

            # Total cell number data
            cn_file = pd.ExcelWriter(export_data_location + 'cn.xlsx')
            self.cn.to_excel(cn_file, sheet_name='cn', index=False)
            self.cn_tidy.to_excel(
                cn_file, sheet_name='cn_tidy', index=False)
            self.cn_bygroup.to_excel(
                cn_file, sheet_name='cn_bygroup', index=False)
            self.cn_bygroup_tidy.to_excel(
                cn_file, sheet_name='cn_bygroup_tidy', index=False)
            cn_file.save()

            # Number greater than threshold data
            inv_file = pd.ExcelWriter(export_data_location + 'inv.xlsx')
            self.inv.to_excel(inv_file, sheet_name='inv', index=False)
            self.inv_tidy.to_excel(
                inv_file, sheet_name='inv_tidy', index=False)
            self.inv_bygroup.to_excel(
                inv_file, sheet_name='inv_bygroup', index=False)
            self.inv_bygroup_tidy.to_excel(
                inv_file, sheet_name='inv_bygroup_tidy', index=False)
            inv_file.save()

            print("GBM cell data exported to excel files.")

        # ===== STATISTICS =====

        # Total number of cells anova
        self.cn_anova_fvalue, self.cn_anova_pvalue, self.cn_anova_tky = \
            stats.anova(self.cn_bygroup)

        # Number greater than threshold anova
        self.inv_anova_fvalue, self.inv_anova_pvalue, self.inv_anova_tky = \
            stats.anova(self.inv_bygroup)

        # Total number vs. number great linear regression
        self.cn_inv_linreg = stats.linreg(
            self.inv_bygroup_tidy['NumGreatThanThresh'], self.cn_bygroup_tidy['NumOfCells'])

        # Saving results from statistics to text files
        if export_stats is True:
            # Creating directory to store exported files
            if export_stats_location == 'Statistics/':
                if 'Statistics' not in os.listdir():
                    os.mkdir('Statistics')

            # Total number of cells anova
            cn_anova_text = stats.anova_results_as_text(self.cn_bygroup)
            cn_anova_filename = export_stats_location + 'cn_anova.txt'
            stats.save_to_txt(cn_anova_filename, cn_anova_text)

            # Number greater than threshold anova
            inv_anova_text = stats.anova_results_as_text(self.inv_bygroup)
            inv_anova_filename = export_stats_location + 'inv_anova.txt'
            stats.save_to_txt(inv_anova_filename, inv_anova_text)

            # Linear regression
            linreg_filename = export_stats_location + 'linreg.txt'
            stats.save_to_txt(linreg_filename, str(self.cn_inv_linreg))

            print("Saved GBM cell statistical test results as text files.")

        # ===== FIGURES =====

        if figures in ['show', 'save']:

            # Figure colors
            if figure_colors is None:
                colors = None
            else:
                colors = figure_colors

            # Displaying figures
            if figures == 'show':
                figs.bargraph(self.cn_bygroup, group_colors=colors)
                figs.bargraph(self.inv_bygroup, group_colors=colors)
                figs.boxplot(self.dts_tidy, group_colors=colors)
                figs.linregplot(self.inv_tidy, self.cn_tidy,
                                group_colors=colors)

            # Saving figures
            if figures == 'save':
                # Creating directory to store exported figures
                if export_figs_location == 'Figures_Original/':
                    if 'Figures_Original' not in os.listdir():
                        os.mkdir('Figures_Original')

                self.cn_bargraph = figs.bargraph(
                    self.cn_bygroup, group_colors=colors)
                cn_bargraph_filename = export_figs_location + 'cn_bargraph.pdf'
                figs.figtofile(self.cn_bargraph, cn_bargraph_filename)
                print("Saved bargraph of total number of cells as PDF.")
                plt.close()

                self.inv_bargraph = figs.bargraph(
                    self.inv_bygroup, group_colors=colors)
                inv_bargraph_filename = export_figs_location + 'inv_bargraph.pdf'
                figs.figtofile(self.inv_bargraph, inv_bargraph_filename)
                print("Saved bargraph of number of invading cells as PDF.")
                plt.close()

                self.dts_boxplot = figs.boxplot(
                    self.dts_tidy, group_colors=colors)
                dts_boxplot_filename = export_figs_location + 'dts_boxplot.pdf'
                figs.figtofile(self.dts_boxplot, dts_boxplot_filename)
                print("Saved boxplot of distace to surface values as PDF.")
                plt.close()

                self.linregplot = figs.linregplot(
                    self.inv_tidy, self.cn_tidy, group_colors=colors)
                linregplot_filename = export_figs_location + 'linregplot.pdf'
                figs.figtofile(self.linregplot, linregplot_filename)
                print("Saved linear regression plot as PDF.")
                plt.close()
        else:
            if figures is not None:
                print(
                    "Pass figures='show' to display figures without saving. \n"
                    "Pass figures='save' to save without displaying."
                )

    def extract_dts_data(self, filenames_dict):
        """
        Uses dictionary of full filename paths and organoid
        identifiers to extract DTS data into pandas df.
        """

        df = pd.DataFrame()
        for filename, org_id in filenames_dict.items():
            data = pd.read_csv(filename, header=2, usecols=[0])
            data.rename(
                columns={data.columns[0]: org_id}, inplace=True)

            # Set negative DTS values to NaN
            neg_indexes = data[data[org_id] < 0].index
            data.drop(neg_indexes, inplace=True)

            # Normalize to min value
            data = data - data.min()

            # Append extracted dts data and unique org identifier to a combined df
            df = pd.concat([df, data], axis=1)

        # Reorder columns to match dictionary of group names
        cols = df.columns.tolist()
        columns_order = []
        for group_name in self.ordered_group_names:
            order = [
                org_id for org_id in cols if org_id.startswith(group_name)]
            columns_order = columns_order + sorted(order)
        df = df.reindex(columns=(columns_order))

        return df

    def get_invasion_threshold(self, df):
        """
        Returns the invasion threshold of a given df based on the average
        value of the invasion_quantile of the control (i.e. DMSO) group.
        Note: invasion_quantile specified in the __init__.
        """

        thresh_values = dict(df.quantile(self.invasion_thresh_quantile))
        values_to_average = []
        for org, thresh_value in thresh_values.items():
            if self.invasion_thresh_group in org:
                values_to_average.append(thresh_value)
        invasion_threshold = mean(values_to_average)

        return invasion_threshold


class OrganoidData():
    """Extract organoid surface area and volume stats from Imaris data"""

    def __init__(
        self,
        groups,
        vol_file_location="Data_Raw/*/*vol_Volume.csv",
        fullvol_file_location="Data_Raw/FullVolAndArea_Statistics/FullVolume.csv",
        sa_file_location="Data_Raw/*/*vol_Area.csv",
        fullsa_file_location="Data_Raw/FullVolAndArea_Statistics/FullArea.csv",
        export_data=False,
        export_data_location="Data_Processed/",
        export_stats=False,
        export_stats_location="Statistics/",
        figures="",
        figure_colors=None,
        export_figs_location="Figures_Original/",
    ):
        """
        Extract organoid surface area and volume stats from Imaris data.

        Processes the data and shapes into pandas dataframes;
        provides the option of exporting dataframes to excel documents.

        Runs a set of statistical tests and provides the option
        to save the results to .txt files.

        Creates bargraphs of the data and allows to display
        or to save as .pdf files.

        Parameters
        ----------
        groups : dict
            Key to reveal blinded groups. Should be in order desired for figures.
            Example: {'group0': 'invasion_ctl', group1': 'DMSO'}
        vol_file_location : str, optional
            Representative location of Volume.csv files (organoid volume data)
            from Imaris data.
            Use glob wildcard charaters to allow extraction of all distance
            to surface data files within the directory.
        fullvol_file_location : str, optional
            Location of file containing full organoid volume data. See note below.
        sa_file_location : str, optional
            Representative location of Area.csv files (organoid surface
            area data) from Imaris data.
            Use glob wildcard charaters to allow extraction of all distance
            to surface data files within the directory.
        fullsa_file_location : str, optional
            Location of file containing full organoid suface area data.
            See note below.
        export_data : bool, optional
            If true exports processed dataframes to excel files.
            Default value false.
        export_data_location : str, optional
            Location where exported excel files will be stored.
            Default is "Data_Processed/" (a subdirectory of parent experiment directory).
        export_stats : bool, optional
            If true saves results of statistical tests as .txt files.
            Default value false.
        export_stats_location : str, optional
            Location of saved statistical results files.
            Default is "Statistics/".
        figures : str, optional
            If 'show' is passed: displays figures without saving.
            If 'save' is passed: saves figures as PDFs without displaying.
            If None is passed (default): does not generate, display or save figures.
        figure_colors : list, optional
            Specifies colors to be used for groups in the figures.
            If None is passed (default), uses default seaborn colorscheme.
            Use seaborn color palettes to specify different colors.
            Example: sns.color_palette("muted").as_hex()[4] is '#956cb4' which
            is light purple.
        export_figs_location : str, optional
            Location of saved figures.
            Default is "Figures_Original/"

        Notes
        -----
        Two groups of data generated abreviated as "vol" for organoid volume,
        "sa" for organoid surface area.

        To access dfs assocaited with each type of data use df, df_bygroup, df_tidy,
        or df_bygroup_tidy (where df is replaced with data abbreviations). Note that
        this naming scheme is replicated in the different sheets of the exported excel
        files if export_data=True.

        Note of full surface and full volume files:
        The Imaris surface models the exterior of the organoid surface
        meaning that the volume of the organoid is actually the full volume of the 3D space
        minus the volume of the Imaris surface. Similar for surface area except
        total area subtracted from organoid surface is true surface of the organoid.
        See methods in paper for more detailed description of organoid and surface
        area modeling with Imaris.
        """

        # ===== DEFINING VARIABLES =====

        self.groups_dict = groups
        self.ordered_group_names = list(groups.values())
        self.separator = "_"

        # ===== FULL VOLUME AND AREA VALUES =====
        self.fullvol = float(pd.read_csv(
            fullvol_file_location, header=2, usecols=[0])['Volume'].tolist()[0])
        self.fullarea = float(pd.read_csv(
            fullsa_file_location, header=2, usecols=[0])['Area'].tolist()[0])

        # ===== TRUE VOLUME AND SURFACE AREA =====

        # Filename dictionaries
        self.vol_files_dict = match_files_with_org_id(
            vol_file_location, self.separator, self.groups_dict)
        self.sa_files_dict = match_files_with_org_id(
            sa_file_location, self.separator, self.groups_dict)

        # Extracting volume and surface area data
        self.vol = self.fullvol - self.extract_org_data(self.vol_files_dict)
        self.sa = self.extract_org_data(self.sa_files_dict) - self.fullarea

        # Making bygroup and tidyform dfs
        self.vol_bygroup, self.vol_tidy, self.vol_bygroup_tidy = \
            mk_comb_and_tidy_dfs(self.vol, 'Volume')
        self.sa_bygroup, self.sa_tidy, self.sa_bygroup_tidy = \
            mk_comb_and_tidy_dfs(self.sa, 'SurfArea')

        # ===== EXPORTING ORG VOL AND SURF AREA DATA TO EXCEL FILES =====

        if export_data is True:
            if export_data_location == 'Data_Processed/':
                if 'Data_Processed' not in os.listdir():
                    os.mkdir('Data_Processed')

            # Org volume data
            vol_file = pd.ExcelWriter(export_data_location + 'vol.xlsx')
            self.vol.to_excel(vol_file, sheet_name='vol', index=False)
            self.vol_tidy.to_excel(
                vol_file, sheet_name='vol_tidy', index=False)
            self.vol_bygroup.to_excel(
                vol_file, sheet_name='vol_bygroup', index=False)
            self.vol_bygroup_tidy.to_excel(
                vol_file, sheet_name='vol_bygroup_tidy', index=False)
            vol_file.save()

            # Org surface area data
            sa_file = pd.ExcelWriter(export_data_location + 'sa.xlsx')
            self.sa.to_excel(sa_file, sheet_name='sa', index=False)
            self.sa_tidy.to_excel(
                sa_file, sheet_name='sa_tidy', index=False)
            self.sa_bygroup.to_excel(
                sa_file, sheet_name='sa_bygroup', index=False)
            self.sa_bygroup_tidy.to_excel(
                sa_file, sheet_name='sa_bygroup_tidy', index=False)
            sa_file.save()
            print("Exported organoid volume and surface area data to excel files.")

        # ===== STATISTICS =====

        # Org vol and surf area anovas
        self.vol_anova_fvalue, self.vol_anova_pvalue, self.vol_anova_tky = \
            stats.anova(self.vol_bygroup)
        self.sa_anova_fvalue, self.sa_anova_pvalue, self.sa_anova_tky = \
            stats.anova(self.sa_bygroup)

        # Saving anova results to text file
        if export_stats is True:
            vol_anova_text = stats.anova_results_as_text(self.vol_bygroup)
            vol_anova_filename = export_stats_location + 'vol_anova.txt'
            stats.save_to_txt(vol_anova_filename, vol_anova_text)

            sa_anova_text = stats.anova_results_as_text(self.sa_bygroup)
            sa_anova_filename = export_stats_location + 'sa_anova.txt'
            stats.save_to_txt(sa_anova_filename, sa_anova_text)

            print("Saved organoid vol and surf area anova tests as text files.")

        # ===== FIGURES =====

        if figures in ['show', 'save']:
            # Figure colors
            if figure_colors is None:
                colors = None
            else:
                colors = figure_colors

            # Displaying figures
            if figures == 'show':
                figs.bargraph(self.vol_bygroup,
                              group_colors=colors, y_label='Volume')
                figs.bargraph(self.sa_bygroup, group_colors=colors,
                              y_label="Surface Area")

            # Saving figures
            if figures == 'save':
                # Creating directory to store exported figures
                if export_figs_location == 'Figures_Original/':
                    if 'Figures_Original' not in os.listdir():
                        os.mkdir('Figures_Original')

                self.vol_bargraph = figs.bargraph(
                    self.vol_bygroup, group_colors=colors, y_label='Volume')
                vol_bargraph_filename = export_figs_location + 'vol_bargraph.pdf'
                figs.figtofile(self.vol_bargraph, vol_bargraph_filename)
                print("Saved bargraph of organoid volume as PDF.")
                plt.close()

                self.sa_bargraph = figs.bargraph(
                    self.sa_bygroup, group_colors=colors, y_label='Surface Area')
                sa_bargraph_filename = export_figs_location + 'sa_bargraph.pdf'
                figs.figtofile(self.sa_bargraph, sa_bargraph_filename)
                print("Saved bargraph of organoid surface area as PDF.")
                plt.close()
        else:
            if figures is not None:
                print(
                    "Pass figures='show' to display figures without saving. \n"
                    "Pass figures='save' to save without displaying."
                )

    def extract_org_data(self, filenames_dict):
        """
        Uses dictionary of full filename paths and organoid
        identifiers to extract organoid data into pandas df.
        """

        df = pd.DataFrame()
        for filename, org_id in filenames_dict.items():
            data = pd.read_csv(filename, header=2, usecols=[0])
            data.rename(
                columns={data.columns[0]: org_id}, inplace=True)

            # Removing smaller objects which may have been
            # captured during Imaris rendering.
            if len(data) > 1:
                data = [max(data)]

            df[org_id] = data[org_id]

        # Reorder columns to match dictionary of group names
        cols = df.columns.tolist()
        columns_order = []
        for group_name in self.ordered_group_names:
            order = [
                org_id for org_id in cols if org_id.startswith(group_name)]
            columns_order = columns_order + sorted(order)
        df = df.reindex(columns=(columns_order))

        return df

import glob
import pandas as pd
import os
from gbm_org_analysis import stats, figs


class GbmCellData():
    """Combines GBM cell data from 3 experimental replicates"""

    def __init__(
        self,
        data_locations=[
            '../*_1/Data_Processed/',
            '../*_2/Data_Processed/',
            '../*_3/Data_Processed/',
        ],
        total_cells_filename='cn.xlsx',
        total_cells_sheetname='cn_bygroup',
        invaded_cells_filename='inv_raw.xlsx',
        invaded_cells_sheetname='inv_raw_bygroup',
        invasion_threshold_sheetname='inv_threshold',
        cell_distance_filename='dts.xlsx',
        cell_distance_sheetname='dts_tidy',
        normalization=False,
        export_data=False,
        export_data_location="Data_Processed/",
        export_stats=False,
        export_stats_location="Statistics/",
        figures="",
        figure_colors=None,
        export_figs_location="Figures_Original/",

    ):

        # ===== COMPILING DATA =====

        def cumulative(lst):
            new_list = []
            j = 0
            for i in range(0, len(lst)):
                j += lst[i]
                new_list.append(j)
            return new_list

        data_by_exp = {}
        orgs_per_group_by_exp = {}
        inv_thresholds = []
        n = 0
        for location in data_locations:
            n += 1
            exp_id = f'exp_{n}'

            # Extrating data
            cn_filename = glob.glob(location + total_cells_filename)[0]
            cn_raw = pd.read_excel(
                cn_filename, sheet_name=total_cells_sheetname, engine='openpyxl')
            cn_norm = cn_raw / cn_raw['DMSO'].mean()

            inv_filename = glob.glob(location + invaded_cells_filename)[0]
            inv_raw = pd.read_excel(
                inv_filename, sheet_name=invaded_cells_sheetname, engine='openpyxl')
            inv_norm = inv_raw / cn_raw

            inv_thresh = pd.read_excel(
                inv_filename, sheet_name=invasion_threshold_sheetname, engine='openpyxl')
            inv_thresholds.append(inv_thresh)

            dts_filename = glob.glob(location + cell_distance_filename)[0]
            dts = pd.read_excel(
                dts_filename, sheet_name=cell_distance_sheetname, engine='openpyxl')

            # Getting org and group names to change organoid number in combined data
            all_org_names = list(dict.fromkeys(list(dts['Org'])))
            groups = []
            for org in all_org_names:
                org_id_split = org.split("_")
                if org_id_split[0] not in groups:
                    groups.append(org_id_split[0])

            all_orgs_as_groupname = []
            for org in all_org_names:
                all_orgs_as_groupname.append(org[:-5])
            if exp_id not in orgs_per_group_by_exp:
                orgs_per_group_by_exp[exp_id] = [
                    (group, all_orgs_as_groupname.count(group)) for group in set(all_orgs_as_groupname)]

            orgs_per_group_by_group = {}
            for group in list(set(all_orgs_as_groupname)):
                if group not in orgs_per_group_by_group:
                    orgs_per_group_by_group[group] = []
            for exp_counts in orgs_per_group_by_exp.values():
                for group_tuple in exp_counts:
                    for group in orgs_per_group_by_group.keys():
                        if group_tuple[0] == group:
                            orgs_per_group_by_group[group] = orgs_per_group_by_group[group] + [
                                group_tuple[1]]

            orgs_per_group_cumulative = {}
            for group, counts in orgs_per_group_by_group.items():
                if group not in orgs_per_group_cumulative:
                    orgs_per_group_cumulative[group] = cumulative(counts)

            if n != 1:
                new_org_numbering = []
                for org in dts['Org']:
                    for group, org_count in orgs_per_group_cumulative.items():
                        if org[:-5] == group:
                            org_num = int(org[-1:]) + org_count[n - 2]
                            new_org = f"{org[:-1]}{org_num}"
                            new_org_numbering.append(new_org)
                dts['Org'] = new_org_numbering

            # Adding all data into data_by_exp dictionary
            if exp_id not in data_by_exp:
                data_by_exp[exp_id] = [cn_raw, cn_norm, inv_raw, inv_norm, dts]

        # Creating skeleton of dictionary that will hold data organized by type instead of experiment
        data_by_type = {
            'cn_raw': [],
            'cn_norm': [],
            'inv_raw': [],
            'inv_norm': [],
            'dts': [],
        }

        for value in list(data_by_exp.values()):
            data_by_type['cn_raw'] = data_by_type['cn_raw'] + [value[0]]
            data_by_type['cn_norm'] = data_by_type['cn_norm'] + [value[1]]
            data_by_type['inv_raw'] = data_by_type['inv_raw'] + [value[2]]
            data_by_type['inv_norm'] = data_by_type['inv_norm'] + [value[3]]
            data_by_type['dts'] = data_by_type['dts'] + [value[4]]

        # Making final data dictionary with concatenated dataframes
        data = {}
        for data_type, values in data_by_type.items():
            if data_type not in data:
                data[data_type] = pd.concat(values)

        inv_thresholds_concat = pd.concat(inv_thresholds)

        # Making list of correct order of organoids in dts data
        old_order = list(dict.fromkeys(list(data['dts']['Org'])))
        new_order = []
        for group in groups:
            group_order = [org for org in old_order if org.startswith(group)]
            new_order = new_order + group_order

        # Reorder dts dataframe with correct order
        df = data['dts'].pivot(columns='Org', values='DTS')
        df = df.reindex(columns=new_order)
        data['dts'] = pd.melt(df, var_name='Org', value_name='DTS').dropna()

        # ===== EXPORTING DATA =====
        org_ids = data['dts']['Org'].values.tolist()
        group_ids = [group.split("_")[0] for group in org_ids]
        self.dts_bygroup_tidy = pd.DataFrame()
        self.dts_bygroup_tidy['Group'] = group_ids
        self.dts_bygroup_tidy['DTS'] = data['dts']['DTS'].values.tolist()

        if export_data is True:
            if export_data_location == 'Data_Processed/':
                if 'Data_Processed' not in os.listdir():
                    os.mkdir('Data_Processed')

            cn_file = pd.ExcelWriter(
                export_data_location + 'cn.xlsx')
            data['cn_raw'].to_excel(
                cn_file, sheet_name='cn_raw', index=False)
            if normalization is True:
                data['cn_norm'].to_excel(
                    cn_file, sheet_name='cn_norm', index=False)
            cn_file.save()

            inv_file = pd.ExcelWriter(
                export_data_location + 'inv.xlsx')
            data['inv_raw'].to_excel(
                inv_file, sheet_name='inv_raw', index=False)
            if normalization is True:
                data['inv_norm'].to_excel(
                    inv_file, sheet_name='inv_norm', index=False)
            inv_thresholds_concat.to_excel(
                inv_file, sheet_name='inv_threshold', index=False)
            inv_file.save()

            dts_file = pd.ExcelWriter(
                export_data_location + 'dts.xlsx')
            data['dts'].to_excel(
                dts_file, sheet_name='dts_tidy', index=False)
            self.dts_bygroup_tidy.to_excel(
                dts_file, sheet_name='dts_bygroup_tidy', index=False)
            dts_file.save()

            print("Integrated GBM cell data exported to excel files.")

        # ===== EXPORTING STATISTICS =====

        if export_stats is True:
            if export_stats_location == 'Statistics/':
                if 'Statistics' not in os.listdir():
                    os.mkdir('Statistics')

            for key, values in data.items():
                if key == 'dts':
                    continue
                if normalization is False:
                    if key == 'cn_norm' or key == 'inv_norm':
                        continue
                anova_text = stats.anova_results_as_text(values)
                filename = export_stats_location + f'{key}_anova.txt'
                stats.save_to_txt(filename, anova_text)
            print("Saved GBM cell statistical test results as text files.")

        # ===== EXPORTING FIGURES =====

        if figures in ['show', 'save']:
            if figure_colors is None:
                colors = None
            else:
                colors = figure_colors

            if figures == 'show':
                for key, value in data.items():
                    if normalization is False:
                        if key == 'cn_norm' or key == 'inv_norm':
                            continue
                    if key == 'dts':
                        figs.boxplot(value, group_colors=colors)
                    else:
                        if key.startswith('cn'):
                            y_label = 'Number of Cells'
                        elif key == 'inv_raw':
                            y_label = 'Number of Invaded Cells'
                        elif key == 'inv_norm':
                            y_label = 'Invaded Cells \n (as fraction of total)'
                        figs.bargraph(value, group_colors=colors,
                                      y_label=y_label)
                figs.norm_stacked_kde(
                    self.dts_bygroup_tidy, group_colors=colors)

            if figures == 'save':
                if export_figs_location == 'Figures_Original/':
                    if 'Figures_Original' not in os.listdir():
                        os.mkdir('Figures_Original')

                for key, value in data.items():
                    if normalization is False:
                        if key == 'cn_norm' or key == 'inv_norm':
                            continue
                    if key == 'dts':
                        dts_boxplot = figs.boxplot(value, group_colors=colors)
                        dts_filename = export_figs_location + 'dts_boxplot.pdf'
                        figs.figtofile(dts_boxplot, dts_filename)
                    else:
                        if key.startswith('cn'):
                            y_label = 'Number of Cells'
                        elif key == 'inv_raw':
                            y_label = 'Number of Invaded Cells'
                        elif key == 'inv_norm':
                            y_label = 'Invaded Cells \n (as fraction of total)'
                        plot = figs.bargraph(
                            value, group_colors=colors, y_label=y_label)
                        filename = export_figs_location + f'{key}_bargraph.pdf'
                        figs.figtofile(plot, filename)

                kde = figs.norm_stacked_kde(
                    self.dts_bygroup_tidy, group_colors=colors)
                kde_filename = export_figs_location + 'kde.pdf'
                figs.figtofile(kde, kde_filename)
                print('Saved GBM cell figures as PDFs.')
        else:
            if figures is not None:
                print(
                    "Pass figures='show' to display GBM cell figures without saving. \n"
                    "Pass figures='save' to save without displaying."
                )


class OrganoidData():

    def __init__(
        self,
        data_locations=[
            '../*_1/Data_Processed/',
            '../*_2/Data_Processed/',
            '../*_3/Data_Processed/',
        ],
        surf_area_filename='sa.xlsx',
        surf_area_sheetname='sa_bygroup',
        volume_filename='vol.xlsx',
        volume_sheetname='vol_bygroup',
        normalization=False,
        export_data=False,
        export_data_location="Data_Processed/",
        export_stats=False,
        export_stats_location="Statistics/",
        figures="",
        figure_colors=None,
        export_figs_location="Figures_Original/",

    ):

        data_by_exp = {}
        n = 0
        for location in data_locations:
            n += 1
            exp_id = f'exp_{n}'

            # Extrating data
            sa_filename = glob.glob(location + surf_area_filename)[0]
            sa_raw = pd.read_excel(
                sa_filename, sheet_name=surf_area_sheetname, engine='openpyxl')
            sa_norm = sa_raw / sa_raw['DMSO'].mean()

            vol_filename = glob.glob(location + volume_filename)[0]
            vol_raw = pd.read_excel(
                vol_filename, sheet_name=volume_sheetname, engine='openpyxl')
            vol_norm = vol_raw / vol_raw['DMSO'].mean()

            if exp_id not in data_by_exp:
                data_by_exp[exp_id] = [sa_raw, sa_norm, vol_raw, vol_norm]

        data_by_type = {
            'sa_raw': [],
            'sa_norm': [],
            'vol_raw': [],
            'vol_norm': [],
        }

        for value in list(data_by_exp.values()):
            data_by_type['sa_raw'] = data_by_type['sa_raw'] + [value[0]]
            data_by_type['sa_norm'] = data_by_type['sa_norm'] + [value[1]]
            data_by_type['vol_raw'] = data_by_type['vol_raw'] + [value[2]]
            data_by_type['vol_norm'] = data_by_type['vol_norm'] + [value[3]]

        data = {}
        for data_type, values in data_by_type.items():
            if data_type not in data:
                data[data_type] = pd.concat(values)

        # ===== EXPORTING DATA =====

        if export_data is True:
            if export_data_location == 'Data_Processed/':
                if 'Data_Processed' not in os.listdir():
                    os.mkdir('Data_Processed')

            sa_file = pd.ExcelWriter(
                export_data_location + 'sa.xlsx')
            data['sa_raw'].to_excel(
                sa_file, sheet_name='sa_raw', index=False)
            if normalization is True:
                data['sa_norm'].to_excel(
                    sa_file, sheet_name='sa_norm', index=False)
            sa_file.save()

            vol_file = pd.ExcelWriter(
                export_data_location + 'vol.xlsx')
            data['vol_raw'].to_excel(
                vol_file, sheet_name='vol_raw', index=False)
            if normalization is True:
                data['vol_norm'].to_excel(
                    vol_file, sheet_name='vol_norm', index=False)
            vol_file.save()

            print("Integrated organoid data exported to excel files.")

        # ===== EXPORTING STATISTICS =====

        if export_stats is True:
            if export_stats_location == 'Statistics/':
                if 'Statistics' not in os.listdir():
                    os.mkdir('Statistics')

            for key, values in data.items():
                if normalization is False:
                    if key == 'sa_norm' or key == 'vol_norm':
                        continue
                anova_text = stats.anova_results_as_text(values)
                filename = export_stats_location + f'{key}_anova.txt'
                stats.save_to_txt(filename, anova_text)

            print("Saved organoid statistical test results as text files.")

        # ===== EXPORTING FIGURES =====

        if figures in ['show', 'save']:
            if figure_colors is None:
                colors = None
            else:
                colors = figure_colors

            if figures == 'show':
                for key, value in data.items():
                    if normalization is False:
                        if key == 'sa_norm' or key == 'vol_norm':
                            continue
                    if key.startswith('sa'):
                        y_label = 'Surface Area'
                    if key.startswith('vol'):
                        y_label = 'Volume'
                    figs.bargraph(value, group_colors=colors,
                                  y_label=y_label)

            if figures == 'save':
                if export_figs_location == 'Figures_Original/':
                    if 'Figures_Original' not in os.listdir():
                        os.mkdir('Figures_Original')

                for key, value in data.items():
                    if normalization is False:
                        if key == 'sa_norm' or key == 'vol_norm':
                            continue
                    if key.startswith('sa'):
                        y_label = 'Surface Area'
                    if key.startswith('vol'):
                        y_label = 'Volume'
                    plot = figs.bargraph(
                        value, group_colors=colors, y_label=y_label)
                    filename = export_figs_location + f'{key}_bargraph.pdf'
                    figs.figtofile(plot, filename)

                print('Saved organoid figures as PDFs.')
        else:
            if figures is not None:
                print(
                    "Pass figures='show' to display organoid figures without saving. \n"
                    "Pass figures='save' to save without displaying."
                )

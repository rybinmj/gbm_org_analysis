import glob
import pandas as pd
import os
from gbm_org_analysis import stats, figs


class GbmCellData():
    """Combines GBM cell data from 3 experimental replicates"""

    def __init__(
        self,
        data_locations=[
            '*_1/Data_Processed/',
            '*_2/Data_Processed/',
            '*_3/Data_Processed/',
        ],
        total_cells_filename='cn.xlsx',
        total_cells_sheetname='cn_bygroup',
        invaded_cells_filename='inv_raw.xlsx',
        invaded_cells_sheetname='inv_raw_bygroup',
        cell_distance_filename='dts.xlsx',
        cell_distance_sheetname='dts_tidy',
        export_data=False,
        export_data_location="Integrated_Results/Data_Processed/",
        export_stats=False,
        export_stats_location="Integrated_Results/Statistics/",
        figures="",
        figure_colors=None,
        export_figs_location="Integrated_Results/Figures_Original/",

    ):

        # ===== COMPILING DATA =====

        data_by_exp = {}
        orgs_per_group = {}
        n = 0
        for location in data_locations:
            n += 1
            exp_id = f'exp_{n}'

            cn_filename = glob.glob(location + total_cells_filename)[0]
            cn_raw = pd.read_excel(
                cn_filename, sheet_name=total_cells_sheetname, engine='openpyxl')
            cn_norm = cn_raw / cn_raw['DMSO'].mean()

            inv_filename = glob.glob(location + invaded_cells_filename)[0]
            inv_raw = pd.read_excel(
                inv_filename, sheet_name=invaded_cells_sheetname, engine='openpyxl')
            inv_norm = inv_raw / cn_raw

            dts_filename = glob.glob(location + cell_distance_filename)[0]
            dts = pd.read_excel(
                dts_filename, sheet_name=cell_distance_sheetname, engine='openpyxl')

            all_org_names = list(dict.fromkeys(list(dts['Org'])))
            all_orgs_as_groupname = []
            for org in all_org_names:
                all_orgs_as_groupname.append(org[:-5])
            if exp_id not in orgs_per_group:
                orgs_per_group[exp_id] = [
                    (group, all_orgs_as_groupname.count(group)) for group in set(all_orgs_as_groupname)]

            if n != 1:
                new_org_numbering = []
                for org in dts['Org']:
                    for org_counts in orgs_per_group[f'exp_{n-1}']:
                        if org[:-5] == org_counts[0]:
                            org_num = int(org[-1:]) + org_counts[1]
                            new_org = f"{org[:-1]}{org_num}"
                            new_org_numbering.append(new_org)
                dts['Org'] = new_org_numbering

            if exp_id not in data_by_exp:
                data_by_exp[exp_id] = [cn_raw, cn_norm, inv_raw, inv_norm, dts]

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

        data = {}
        for data_type, values in data_by_type.items():
            if data_type not in data:
                data[data_type] = pd.concat(values)

        # ===== EXPORTING DATA =====

        if export_data is True:
            if export_data_location == 'Integrated_Results/Data_Processed/':
                if 'Integrated_Results' not in os.listdir():
                    os.mkdir('Integrated_Results')
                if 'Data_Processed' not in os.listdir('Integrated_Results/'):
                    os.mkdir('Integrated_Results/Data_Processed')

            cn_file = pd.ExcelWriter(
                export_data_location + 'cn.xlsx')
            data['cn_raw'].to_excel(
                cn_file, sheet_name='cn_raw', index=False)
            data['cn_norm'].to_excel(
                cn_file, sheet_name='cn_norm', index=False)
            cn_file.save()

            inv_file = pd.ExcelWriter(
                export_data_location + 'inv.xlsx')
            data['inv_raw'].to_excel(
                inv_file, sheet_name='inv_raw', index=False)
            data['inv_norm'].to_excel(
                inv_file, sheet_name='inv_norm', index=False)
            inv_file.save()

            dts_file = pd.ExcelWriter(
                export_data_location + 'dts.xlsx')
            data['dts'].to_excel(
                dts_file, sheet_name='dts_tidy', index=False)
            dts_file.save()

            print("Integrated data exported to excel files")

        # ===== EXPORTING STATISTICS =====

        if export_stats is True:
            if export_stats_location == 'Integrated_Results/Statistics/':
                if 'Integrated_Results' not in os.listdir():
                    os.mkdir('Integrated_Results')
                if 'Statistics' not in os.listdir('Integrated_Results/'):
                    os.mkdir('Integrated_Results/Statistics')

            for key, values in data.items():
                if key == 'dts':
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

            if figures == 'save':
                if export_figs_location == 'Integrated_Results/Figures_Original/':
                    if 'Integrated_Results' not in os.listdir():
                        os.mkdir('Integrated_Results')
                    if 'Figures_Original' not in os.listdir('Integrated_Results/'):
                        os.mkdir('Integrated_Results/Figures_Original')

                for key, value in data.items():
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
                print('Saved figures as PDFs.')
        else:
            if figures is not None:
                print(
                    "Pass figures='show' to display figures without saving. \n"
                    "Pass figures='save' to save without displaying."
                )


# class OrganoidData():

#     def __init(
#         self,
#         data_locations=[
#             '/*_1/Data_Processed/',
#             '/*_2/Data_Processed/',
#             '/*_3/Data_Processed/',
#         ],
#         surf_area_filename='sa.xlsx',
#         surf_area_sheetname='sa_bygroup',
#         vol_filename='vol.xlsx',
#         vol_sheetname='vol_bygroup',


#     ):
#         a = 1

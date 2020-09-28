import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def figtofile(fig, filename, filetype='.pdf'):
    """Saves figures as file."""
    if filename[-4:] == ".pdf":
        full_filename = filename
    else:
        full_filename = filename + filetype
    fig.savefig(full_filename)


class FigureColors():
    """Color palettes and groups of colors for figures."""

    # Colors for figures
    muted = sns.color_palette("muted").as_hex()
    dark = sns.color_palette("dark").as_hex()
    pastel = sns.color_palette("pastel").as_hex()
    colorblind = sns.color_palette("colorblind").as_hex()
    deep = sns.color_palette("deep").as_hex()

    colorlist = [
        pastel[7],  # light gray (inv_ctrl)
        colorblind[7],  # Dark gray (DMSO)
        muted[4],  # light purple (TAS120)
        muted[3],  # light red (JQ1)
        muted[0],  # light blue (MK8628)
        muted[2],  # light green (UM002)
    ]

    colorlist_for_combinations = [
        pastel[7],  # light gray (inv_ctrl)
        colorblind[7],  # Dark gray (DMSO)
        muted[4],  # light purple (TAS120)
        muted[3],  # light red (JQ1)
        dark[3],  # dark red (JQ1+TAS120)
        muted[0],  # light blue (MK8628)
        dark[0],  # dark blue (MK8628+TAS120)
        muted[2],  # light green (UM002)
        dark[2],  # dark green (UM002+TAS120)
    ]


def get_colors_for_tidy(df_tidy, colors):
    all_orgs = df_tidy[df_tidy.columns[0]].tolist()
    orgs = []
    for org in all_orgs:
        if org not in orgs:
            orgs.append(org)

    groups = []
    for org in orgs:
        if org[:-5] not in groups:
            groups.append(org[:-5])

    needed_colors = colors[0:len(groups)]
    color_dict = dict(zip(groups, needed_colors))

    all_colors = []
    for group, color in color_dict.items():
        for org in orgs:
            if org[:-5] == group:
                all_colors.append(color)

    return all_colors


def bargraph(
    df,  # designed to take 'bygroup' df (e.g. cn_bygroup, inv_bygroup)
    group_colors=None,
    y_range=None,  # Range of y axis. Default set to 110% of max value
    y_label="Number of Cells",  # Label on y axis
    size=[15, 8],  # Size of figure [length, height]
):
    """
    Creates bargraph (incl. data points) of given wideform df.
    Designed to plot cell number data.
    """

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

    if group_colors is None:
        colors = None
    else:
        colors = group_colors

    plot = sns.barplot(
        data=df,
        palette=colors,
        linewidth=2.5,
        orient="v",
        capsize=0.1,
        ci=68,
        errcolor="black",
    )

    df_tidy = pd.melt(df).dropna()
    plot = sns.stripplot(
        x=df_tidy.columns[0],
        y=df_tidy.columns[1],
        data=df_tidy,
        color="white",
        size=10,
        edgecolor="black",
        linewidth=1,
        jitter=0.1,
        orient="v",
    )

    plot.set_ylabel(y_label)
    plot.set_xlabel("")

    sns.despine()
    plt.tight_layout()

    if y_range is not None:
        ymax = df.max().max() * 1.1
    else:
        ymax = y_range
    plt.ylim(0, ymax)

    return plt


def boxplot(
    df_tidy,
    group_colors=None,
    size=[12.5, 12.5],
    x_range=None,
    x_label="Distance to Surface (µm)",
    y_label="Groups",
):
    """
    Creates boxplot (incl. data points) of given tidyform df.
    Designed to plot cell distance to surface data.
    """

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

    if group_colors is None:
        colors = None
    else:
        colors = get_colors_for_tidy(df_tidy, group_colors)

    plot = sns.boxplot(
        x=df_tidy.columns[1],
        y=df_tidy.columns[0],
        data=df_tidy,
        orient="h",
        fliersize=0,
        palette=colors,
    )

    plot = sns.stripplot(
        x=df_tidy.columns[1],
        y=df_tidy.columns[0],
        data=df_tidy,
        orient="h",
        palette=colors,
        size=2.5,
        linewidth=0,
        jitter=0.25,
    )

    plot.set_xlabel(x_label)
    plot.set_ylabel(y_label)
    #         plot.set_title(t, fontsize=30)
    sns.set_context(
        font_scale=1.75,
        rc=fig_style
    )

    plt.tight_layout()
    sns.despine()

    if x_range is None:
        xmax = df_tidy.max(numeric_only=True).max() * 1.1
    else:
        xmax = x_range
    plt.xlim(0, xmax)

    return plt


def linregplot(
    df1,  # Designed to take inv_tidy
    df2,  # Designed to take cn_tidy
    group_colors=None,
    size=[6, 6],
    x_label='Total Cells',
    y_label='Invaded Cells',
):
    """Plots linear regression"""

    combined_df = df2.copy()
    combined_df["NumOfInvadedCells"] = df1[df1.columns[1]]

    plt.figure(figsize=size)

    if group_colors is None:
        colors = None
    else:
        colors = get_colors_for_tidy(df1, group_colors)

    plot = sns.regplot(
        x="NumOfCells",
        y="NumOfInvadedCells",
        data=combined_df,
        truncate=False,
        scatter_kws={'color': colors, 's': 75, 'edgecolor': "black"},
        line_kws={'color': 'black'}
    )

    plot.set_xlabel(x_label)
    plot.set_ylabel(y_label)

    sns.despine()
    plt.tight_layout()

    return plt

# Last updated July 25, 2023
# Version 0.1.2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
import plotly.graph_objs as go
import matplotlib.ticker as mticker
from typing import List
from seaborn import FacetGrid


# Parameters for plots
my_colors = sns.color_palette()
median_params = dict(color="black", linewidth=1.5)
barlabel_fontsize = 10
title_fontsize = 14


def plot_available_data(df: pd.DataFrame, title: str) -> Figure:
    """Function to draw bar plots with the percentage
    of available data in the dataframe columns."""
    available_data = (1 - df.isna().mean()) * 100
    fig = plt.figure()
    ax = sns.barplot(
        y=available_data.index, x=available_data.values, color=my_colors[0]
    )
    # Add data labels
    for i in ax.containers:
        ax.bar_label(
            i,
            labels=format_container_labels(i, fmt="{:.1f}%"),
            padding=2,
            fontsize=barlabel_fontsize,
        )
    # Format x ticks
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0f}%")
    ax.set_ylabel("Columns")
    plt.title(title, fontsize=title_fontsize)
    return fig


def population_pyramid(
    df: pd.DataFrame,
    columns: List[str],
    x_max: float,
    title: str,
) -> Figure:
    """Function to draw a population pyramid from a pivot table."""
    df = df / (df.sum().sum()) * 100
    fig, axs = plt.subplots(ncols=2)
    axs[0].set_xlim([x_max, 0])
    for k in range(2):
        sns.barplot(
            y=df.index, x=df[columns[k]], ax=axs[k], color=my_colors[k]
        )
        axs[k].set(
            xlabel="", title=format_pyramid_title(df=df, column=columns[k])
        )
        for i in axs[k].containers:
            axs[k].bar_label(
                i,
                labels=format_container_labels(i, fmt="{:.1f}%"),
                padding=2,
                fontsize=barlabel_fontsize,
            )
        axs[k].xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    axs[1].set(xlim=[0, x_max], ylabel="")
    # Format x and y axis
    axs[0].axes.get_yaxis().set_visible(False)
    axs[0].spines[["left"]].set_visible(False)
    axs[0].spines[["right"]].set_visible(True)
    fig.suptitle(title, fontsize=title_fontsize)
    return fig


def barplot_counts(
    sr: pd.Series,
    ylabel: str,
    title: str,
    sort: bool = True,
    percentage: bool = False,
    min_value: float = 0,
    custom_yticks: list = None,
) -> Figure:
    """Function to draw barplot from value counts."""
    fig = plt.figure()
    # Set the minimum value to show
    sr = sr[sr >= min_value]
    if sort:
        sr = sr.sort_values(ascending=False)
    ax = sns.barplot(y=sr.index.astype(str), x=sr.values, color=my_colors[0])
    # Optional custom y tick labels
    if custom_yticks is not None:
        ax.set_yticklabels(custom_yticks)
    # Add appropriate data labels
    if percentage:
        ax.xaxis.set_major_formatter(
            mticker.PercentFormatter(xmax=1, decimals=0)
        )
        for i in ax.containers:
            ax.bar_label(
                i,
                labels=format_container_labels(i, fmt="{:.1%}"),
                padding=2,
                fontsize=barlabel_fontsize,
            )
    else:
        for i in ax.containers:
            ax.bar_label(i, padding=2, fontsize=barlabel_fontsize)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=title_fontsize)
    return fig


def histplot_2groups(
    df: pd.DataFrame,
    x: str,
    hue: str,
    hue_order: List[str],
    title: str,
    kde: bool = False,
) -> Figure:
    """Function to draw histplot of two groups."""
    fig = plt.figure()
    ax = sns.histplot(
        data=df,
        x=x,
        hue=hue,
        hue_order=hue_order,
        kde=kde,
        alpha=0.5,
        legend=False,
    )
    # Set title cases to legend
    ax.legend(
        handles=[ax.containers[1][0], ax.containers[0][0]],
        title=hue.replace("_", " ").title(),
        labels=[i.title() for i in hue_order],
    )
    ax.grid(axis="x")
    plt.xlabel(x.title())
    plt.title(title, fontsize=title_fontsize)
    return fig


def histplots_facet(
    df: pd.DataFrame,
    x: str,
    hue: str,
    hue_order: List[str],
    title: str,
    binwidth: float = None,
    col: str = None,
    sharey: bool = False,
    xlabel: str = None,
) -> FacetGrid:
    """Function to draw facetgrid of histplots with kde for two groups."""
    grid = sns.displot(
        data=df,
        x=x,
        hue=hue,
        col=col,
        hue_order=hue_order,
        binwidth=binwidth,
        facet_kws=dict(sharey=sharey),
        kde=True,
        palette=my_colors[: len(hue_order)],
        alpha=0.5,
    )
    # Set title cases to subplots and remove vertical grid
    grid.set_titles("{col_name}")
    for ax in grid.axes.flatten():
        ax.xaxis.grid(False)
        ax.set_title(ax.get_title().title())
    # Set title cases to legend
    grid.figure.legend(
        handles=grid.legend.get_patches(),
        title=hue.replace("_", " ").title(),
        labels=[i.title() for i in hue_order],
        loc="center right",
    )
    grid.legend.set(visible=False)
    if xlabel is not None:
        grid.set_xlabels(xlabel)
    else:
        grid.set_xlabels(x.title())
    # Set top level title
    grid.figure.subplots_adjust(top=0.85)
    grid.figure.suptitle(title, fontsize=title_fontsize)
    return grid


def histplots_count_percent(
    df: pd.DataFrame, y: str, hue: str, hue_order: List[str], title: str
) -> Figure:
    """Function to draw related histplots showing group counts and their percentage."""
    fig, axs = plt.subplots(
        ncols=2, sharey=True, gridspec_kw={"width_ratios": [4, 1]}
    )
    # Draw histogram with counts
    sns.histplot(
        data=df,
        y=y,
        hue=hue,
        stat="count",
        multiple="stack",
        ax=axs[0],
        palette=my_colors[: len(hue_order)],
        hue_order=hue_order,
    )
    # draw histogram with percentages
    sns.histplot(
        data=df,
        y=y,
        hue=hue,
        stat="percent",
        multiple="fill",
        ax=axs[1],
        palette=my_colors[: len(hue_order)],
        hue_order=hue_order,
    )
    # Add data labels
    for i in axs[0].containers:
        axs[0].bar_label(
            i,
            labels=format_container_labels(i, fmt="{:.0f}"),
            label_type="center",
            fontsize=barlabel_fontsize,
        )
    for i in axs[1].containers:
        if i is axs[1].containers[0]:
            axs[1].bar_label(
                i,
                labels=format_container_labels(i, fmt="{:.1%}"),
                label_type="center",
                fontsize=barlabel_fontsize,
                padding=-10,
            )
        else:
            axs[1].bar_label(
                i,
                labels=format_container_labels(i, fmt="{:.1%}"),
                label_type="center",
                fontsize=barlabel_fontsize,
            )
    # Set title cases to legend
    axs[0].legend(
        handles=axs[0].get_legend().legendHandles,
        title=hue.replace("_", " ").title(),
        labels=[i.title() for i in hue_order],
    )
    # Format x and y axis
    axs[0].grid(axis="y")
    axs[0].set_ylabel(y.title())
    axs[1].get_legend().remove()
    axs[1].grid(False)
    axs[1].set(xticks=[], xticklabels=[])
    fig.suptitle(title, fontsize=title_fontsize)
    return fig, axs


def histplot_filled_percent(
    df: pd.DataFrame,
    y: str,
    hue: str,
    hue_order: List[str],
    title: str,
    legend_title: str = None,
    ylabel: str = None,
) -> Figure:
    "Function to draw sns.histplot representing percentage with annotations."
    fig = plt.figure()
    ax = sns.histplot(
        data=df,
        y=y,
        hue=hue,
        hue_order=hue_order,
        stat="percent",
        multiple="fill",
    )
    for i in ax.containers:
        ax.bar_label(
            i,
            labels=format_container_labels(i, fmt="{:.1%}"),
            label_type="center",
            fontsize=barlabel_fontsize,
        )
    ax.legend(
        handles=ax.get_legend().legend_handles,
        title=legend_title,
        labels=hue_order,
        bbox_to_anchor=(1.02, 1),
    )
    ax.grid(False)
    ax.set(xticks=[], xticklabels=[], ylabel=ylabel)
    ax.set_title(title, fontsize=title_fontsize)
    return fig, ax


def box_strip_plot(
    df: pd.DataFrame, x: str, y: str, order: List[str], title: str, ylabel: str
) -> Figure:
    """Function to plot stripplot on top of boxplot."""
    fig = plt.figure()
    sns.boxplot(
        data=df,
        x=x,
        y=y,
        order=order,
        width=0.4,
        fliersize=0,
        color="lightgrey",
        medianprops=median_params,
    )
    sns.stripplot(
        data=df,
        x=x,
        y=y,
        order=order,
        alpha=0.5,
        linewidth=1,
        color=my_colors[0],
    )
    plt.ylabel(ylabel)
    plt.xlabel(x.replace("_", " ").title())
    plt.title(title, fontsize=title_fontsize)
    return fig


def timeseries_new_accumulated(
    df: pd.DataFrame,
    accumulated: str,
    new: str,
    title: str,
) -> Figure:
    """Function to draw new and accumulated timeseries data from a dataframe with date index."""
    weekday = df.index.day_name()  # for weekly fluctuation awareness
    new_data = go.Bar(
        x=df.index,
        y=df[new],
        name=new.split("_")[0].title(),
        opacity=0.6,
        marker={"color": my_colors[0]},
        customdata=weekday,
        hovertemplate="%{y} (%{customdata})",
    )
    accumulated_data = go.Scatter(
        x=df.index,
        y=df[accumulated],
        name=accumulated.split("_")[0].title(),
        yaxis="y2",
        marker={"color": my_colors[1]},
    )
    fig = go.Figure()
    fig.add_trace(new_data)
    fig.add_trace(accumulated_data)

    fig.update_layout(
        title=title,
        hovermode="x unified",
        xaxis={"title": "Date"},
        yaxis={
            "title": new.replace("_", " ").title(),
            "range": [0, max(df[new])],
        },
        yaxis2={
            "title": accumulated.replace("_", " ").title(),
            "overlaying": "y",
            "side": "right",
            "range": [0, max(df[accumulated])],
        },
        showlegend=False,
        template=None,
    )
    return fig


def update_legend(ax: Axes, title: str, labels: List[str]) -> Axes:
    """Function to update axes legend."""
    ax.legend(
        handles=ax.get_legend().legend_handles, title=title, labels=labels
    )
    return ax


def add_text(fig: Figure, s: str) -> Figure:
    """Function to add some explanatory text to the figure bottom."""
    fig.text(x=0.1, y=-0.05, s=s, color="grey")
    return fig


def format_container_labels(container: BarContainer, fmt: str) -> List[str]:
    """Function to format labels for bar containers."""
    return [fmt.format(x) if x != 0 else "" for x in container.datavalues]


def format_pyramid_title(df: pd.DataFrame, column: str) -> str:
    """Function to format titles of 2sided barplot."""
    return f"{column.capitalize()} (Total = {df[column].sum():.1f}%)"

import math
from typing import Collection

import matplotlib.pyplot as plt
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
from IPython.display import display, Markdown as Md


def get_fig_distplot(
    df: pd.DataFrame | Collection[pd.DataFrame],
    col: str | Collection[str],
    legend: None | str | Collection[str] = None,
    show_rug: bool = True,
) -> go.Figure:
    if legend is None:
        legend = col

    if isinstance(df, pd.DataFrame):
        if not isinstance(col, str) or not isinstance(legend, str):
            raise ValueError

        return ff.create_distplot(
            hist_data=[
                df[col],
            ],
            group_labels=[
                legend,
            ],
            show_hist=False,
            show_rug=show_rug,
        ).update_layout(
            margin=dict(t=40, l=40, b=40, r=40),
        )

    else:
        if isinstance(col, str) or isinstance(legend, str):
            raise ValueError
        if len(df) != len(col):
            raise ValueError

        return ff.create_distplot(
            hist_data=[
                d[c] for d, c in zip(df, col)
            ],
            group_labels=legend,
            show_hist=False,
            show_rug=show_rug,
        ).update_layout(
            margin=dict(t=40, l=40, b=40, r=40),
        )


def plot_linear_corr(df: pd.DataFrame, y_col: str, x_cols):
    n_cols = 3
    n_rows = math.ceil(len(x_cols) / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 4 * n_rows),
        constrained_layout=True,
    )

    axes = axes.ravel()

    for ax, col in zip(axes, x_cols):
        sns.regplot(
            data=df,
            x=col,
            y=y_col,
            scatter_kws={"s": 10, "alpha": 0.5},
            line_kws={"color": "red"},
            ax=ax,
        )
        ax.set_title(col)

    for ax in axes[len(x_cols):]:
        fig.delaxes(ax)

    plt.show()


def show_qtd_rm_lines(
    df_orig: pd.DataFrame,
    dfs: Collection[pd.DataFrame],
    dfs_names: Collection[str],
    text: str = 'Linhas removidas em',  # 'Outliers removidos em',
) -> None:
    """Quantidade e proporção de linhas (normalmente outliers) removidas"""

    if len(dfs) != len(dfs_names):
        raise ValueError
    for df, df_name in zip(dfs, dfs_names):
        display(
            Md(
                f"""{text} `{df_name}`: {df_orig.shape[0] - df.shape[0]} (de {df_orig.shape[0]}) ({
                    (1 - df.shape[0] / df_orig.shape[0]) * 100:.2f}%), restando {df.shape[0]}"""
            )
        )

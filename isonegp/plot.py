# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 17:35:15 2020

@author: Frankie
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_data(
    data: np.ndarray,
    title: str = "",
    xtitle: str = "",
    ytitle: str = "",
    is_save: bool = False,
    run_prefix: str = "",
    save_filename: str = "",
) -> None:
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    if is_save:
        plt.savefig(f"sessions/{run_prefix}/figs/{save_filename}.png", dpi=600)
        plt.clf()
    else:
        plt.show()


def plot_overlapping_data(
    data: list[np.ndarray],
    title: str = "",
    xtitle: str = "",
    ytitle: str = "",
    is_save: bool = False,
    run_prefix: str = "",
    save_filename: str = "",
) -> None:
    for datum in data:
        plt.plot(datum)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    if is_save:
        plt.savefig(f"sessions/{run_prefix}/figs/{save_filename}.png", dpi=600)
        plt.clf()
    else:
        plt.show()


def plot_autocorr(data: np.ndarray, title: str = "Autocorrelation", **kwargs) -> None:
    self_corr = np.correlate(data, data, "full")
    plot_data(self_corr, title, **kwargs)


def plot_crosscorr(
    x: np.ndarray, y: np.ndarray, title: str = "Crosscorrelation", **kwargs
) -> None:
    crosscorr = np.correlate(x, y, "full")
    plot_data(crosscorr, title, **kwargs)


def plot_cat_data(
    data: list[np.ndarray],
    matplot_colors: list[str],
    legend_titles: list[str],
    title: str = "",
    xtitle: str = "",
    ytitle: str = "",
    is_save: bool = False,
    run_prefix: str = "",
    save_filename: str = "",
) -> None:
    assert len(data) == len(matplot_colors)
    assert len(matplot_colors) == len(legend_titles)
    total_elems = sum([i.size for i in data])
    # elems_range = [i for i in range(total_elems)]
    elems_range = np.linspace(0, total_elems - 1, total_elems)
    plt.ylabel(ytitle)
    plt.xlabel(xtitle)
    plt.title(title)

    start, end = 0, 0
    for datum, color, legend_title in zip(data, matplot_colors, legend_titles):
        end += datum.size
        plt.plot(elems_range[start:end], datum, color, label=legend_title)
        start += datum.size
    plt.legend()
    if is_save:
        plt.savefig(f'sessions/{run_prefix}/figs/{save_filename}.png', dpi=600)
        plt.clf()
    else:
        plt.show()

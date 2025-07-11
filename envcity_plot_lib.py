#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 17:52:29 2023

@author: mateus
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

import matplotlib.dates as mdates

from matplotlib.gridspec import GridSpec
from matplotlib.ticker import EngFormatter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_data_by_time_and_regr_plot(dict_data_e1, dict_data_e2, labels, latex_labels, start=None, end=None, e1_label = 'Station 1', e2_label = 'Station 2', style_plot = 'dark'):

    for idx, l in enumerate(labels):

        if start != None and end != None:
            e1 = dict_data_e1[l].loc[start:end]
            e2 = dict_data_e2[l].loc[start:end]
        else:
            e1 = dict_data_e1[l]
            e2 = dict_data_e2[l]
            
        concatenated = pd.concat([e1, e2], axis=1, keys=[e1_label, e2_label])

        #with sns.axes_style(style=style_plot):
        sns.set_palette(style_plot)

        # fig, ax = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
        fig = plt.figure(figsize=(12, 4), layout='tight')
        gs = GridSpec(1, 2, width_ratios=[3,1])

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        sns.lineplot(data=concatenated, linewidth=1, ax=ax1)
        sns.regplot(data=concatenated, x = e1_label, y = e2_label, ax = ax2, marker = '.')
        ax2.axline((0,0), slope=1)
        # Set axis labels and title
        ax1.set_title(f"$\mathrm{{{latex_labels[idx]}}}$ Concentration from {start} to {end}")
        ax1.set_ylabel("Concentration (ppb)")
        ax1.set_xlabel("Date")

        ax2.set_ylabel(e2_label)
        ax2.set_xlabel(e1_label)

        ax2.set_box_aspect(1)

        # Format the x-axis labels
        # date_format = mdates.DateFormatter('%d/%m/%y')
        # ax1.xaxis.set_major_formatter(date_format)
        # ax1.tick_params(axis='x', labelrotation=45)

        # ax1.set_xlim([0, dict_data_e1[l].index[-1]])

        # Using metric prefixes
        # ax1.yaxis.set_major_formatter(EngFormatter())
        # ax2.yaxis.set_major_formatter(EngFormatter())
        # ax2.xaxis.set_major_formatter(EngFormatter())

        # ax1.margins(0.1, 0.1)
        # ax2.margins(0.2, 0.2)

        # ax1.autoscale(enable = None, axis="x", tight=True)
        # ax2.autoscale(enable=None, axis="x", tight=True)
        fig.autofmt_xdate()
        # ax2.legend(loc='upper right', frameon=False, bbox_to_anchor=(1.25, 1.05))
        # fig.savefig(f"{l}_time.pdf", format='pdf', dpi=fig.dpi, bbox_inches='tight')

        plt.show()

def plot_data_by_time(dict_data_e1, dict_data_e2, labels, latex_labels, start, end, style_plot = 'dark'):

    for idx, l in enumerate(labels):

        e1 = dict_data_e1[l].loc[start:end]
        e2 = dict_data_e2[l].loc[start:end]

        concatenated = pd.concat([e1, e2], axis=1, keys=['Station 1', 'Station 2'])

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        sns.lineplot(data=concatenated, linewidth=1, ax=ax)

        # Set axis labels and title
        ax.set_title(f"$\mathrm{{{latex_labels[idx]}}}$ Concentration from {start} to {end}")
        ax.set_ylabel("Concentration (ppb)")
        ax.set_xlabel("Date")

        # Format the x-axis labels
        date_format = mdates.DateFormatter('%d/%m/%y')
        ax.xaxis.set_major_formatter(date_format)
        ax.tick_params(axis='x', labelrotation=45)

        # format the y-axis labels
        ax.yaxis.set_major_formatter(EngFormatter())

        # ax.margins(0.2, 0.2)
        ax.autoscale(enable=None, tight=True)

        ax.legend(loc='upper right', frameon=False, bbox_to_anchor=(1.3, 1.05))
        fig.savefig(f"{l}_time.pdf", format='pdf', dpi=fig.dpi, bbox_inches='tight')

        plt.tight_layout()
        plt.show()

def plot_boxplot(dict_data_e1, dict_data_e2, labels, latex_labels, start, end, style_plot = 'dark'):
# def plot_boxplot(df, labels, latex_labels, start, end, style_plot = 'dark'):

    for idx, l in enumerate(labels):

        e1 = dict_data_e1[l].loc[start:end]
        e2 = dict_data_e2[l].loc[start:end]
            
        # For comparing all equations just uncomment these two lines
        # e1_concat = pd.concat(e1, axis = 1, keys = ["Equação 1", "Equação 2", "Equação 3", "Equação 4"])
        # e2_concat = pd.concat(e2, axis = 1, keys = ["Equação 1", "Equação 2", "Equação 3", "Equação 4"])
        concatenated = pd.concat([e1, e2], axis=1, keys=['Station 1', 'Station 2'])

        melted = pd.melt(concatenated, var_name=['Data', 'sensor'], value_name='value', ignore_index=False)

        sns.set_palette(style_plot)
        #with sns.axes_style(style=style_plot):

        fig, ax = plt.subplots(1, 1, figsize = (5, 2))
        bp = sns.boxplot(data=melted, x="sensor", y = 'value', hue = 'Data', ax = ax, width=0.7, linewidth=1, fliersize=5)
        bp.set(xlabel = "")
        ax.set_title(f"$\mathrm{{{latex_labels[idx]}}}$ Concentration from {start} to {end}", fontsize = 10)
        ax.set_ylabel("ppb")
        ax.legend(loc='upper right', frameon = False, bbox_to_anchor=(1.35, 1.05))
        ax.set(xticklabels=[])
        ax.tick_params(bottom=False)
        ax.yaxis.set_major_formatter(EngFormatter())

        fig.savefig(f"{l}.pdf", format = 'pdf', dpi=fig.dpi, bbox_inches =  'tight')
        plt.show()

def plot_boxplot_alphasense_equations(dict_data_e1, dict_data_e2, labels, latex_labels, start, end, style_plot = 'dark'):

    for idx, l in enumerate(labels):

        e1 = dict_data_e1[l].loc[start:end]
        e2 = dict_data_e2[l].loc[start:end]

        concatenated = pd.concat([e1, e2], axis=1, keys=['Station 1', 'Station 2'])

        melted = pd.melt(concatenated, var_name=['Data', 'sensor'], value_name='value', ignore_index=False)

        sns.set_palette(style_plot)
        #with sns.axes_style(style=style_plot):

        fig, ax = plt.subplots(1, 1, figsize = (5, 2))
        bp = sns.boxplot(data=melted, x="sensor", y = 'value', hue = 'Data', ax = ax, width=0.7, linewidth=1, fliersize=3)
        ax.set_title(f"$\mathrm{{{latex_labels[idx]}}}$ Concentration from {start} to {end}", fontsize = 10)
        ax.set_ylabel("ppb")
        ax.legend(loc='upper right', frameon = False, bbox_to_anchor=(1.4, 1.05))
        ax.yaxis.set_major_formatter(EngFormatter())

        fig.savefig(f"{l}.pdf", format = 'pdf', dpi=fig.dpi, bbox_inches =  'tight')
        plt.show()

def plot_regr_plus_stats(dict_data_e1, dict_data_e2, labels, latex_labels, start, end, style_plot = 'dark'):
    
    res_dict = {}
    for idx,l in enumerate(labels):
    
      e1 = dict_data_e1[l]
      e2 = dict_data_e2[l]
    
      e1, e2 = reindex_df(e1, e2)
      i = e1['Equation 1'].notna() & e2['Equation 1'].notna()
      e1 = e1[i]
      e2 = e2[i]
    
      # Filtering the data
      # e1 = pd.Series(data= butter_lowpass_filter(cutoff=cutoff, nyq=nyq, data=e1.values, order=order), index = e1.index)
      # e2 = pd.Series(data=butter_lowpass_filter(cutoff=cutoff, nyq=nyq, data=e2.values, order=order), index = e2.index)
    
      # Create feature and target arrays
      X = e1.values.reshape(-1, 1)
      y = e2.values.reshape(-1, 1)
    
      # Fit linear regression model
      model = LinearRegression().fit(X, y)
    
      # Get regression coefficients
      print(model.coef_)
      print('Intercept:', model.intercept_[0])
      print('Coefficient:', model.coef_[0][0])
      print("r2", res := pearsonr(e2['Equation 1'], e1['Equation 1']))
      print("cvmae", res_cvmae := cvmae(y=e2['Equation 1'], yref=e1['Equation 1']))
      print("rmse", res_rmse := rmse(y=e2['Equation 1'], yref=e1['Equation 1']))
    
      res_dict[l] = {'r2' : res[0], 'rmse' : res_rmse, 'cvmae' : res_cvmae}
    
      #plt.annotate(f'R-squared = {res[0]:.2f}', xy=(0.95, 0.95),
      #             xycoords='axes fraction', ha='right')
    
      fig, ax = plt.subplots(1, 1, figsize = (6, 2))
      props = dict(boxstyle='round', alpha=0.5)
    
      str_annotate = f"y = {model.coef_[0][0]:.02}x + {model.intercept_[0]:.02f}\n$r^2={res[0]:.04}$\nCvMAE={res_cvmae:.02}\nRMSE={res_rmse:.02f}"
    
      # place a text box in upper left in axes coords
      ax.text(1.01, 0.7, str_annotate, transform=ax.transAxes,
                     fontsize=8, verticalalignment='bottom',
                     horizontalalignment = 'left')
    
      sns.regplot(x = e1.values, y = e2.values, marker = '.', scatter_kws = {"alpha" : 0.4}, ax=ax)
      plt.suptitle(f"Comparison between $\mathrm{{{latex_labels[idx]}}}$ sensors")
      plt.ylabel("Sensor 2")
      plt.xlabel("Sensor 1")
      fig.savefig(f"reg_{l}.pdf", format = 'pdf', dpi=fig.dpi, bbox_inches =  'tight')
    
      plt.show()
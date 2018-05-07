#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 12:16:50 2018

@author: saskiad
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import core, os

def example_code():

    cres
    areas = ['VISp','VISl','VISal','VISpm','VISam','VISrl']
    depths = [100,200,300,500]

    table = pd.DataFrame(columns=('area','cre','depth','number_experiments','number_cells_dg','responsive_cells_dg', 'percent_dg'), index=range(114))
    for ai, a in enumerate(areas):
        for ci, c in enumerate(cre_depth):
            i = (ci)+(19*ai)
            table.area.iloc[i] = a
            table.cre.iloc[i] = c[0]
            table.depth.iloc[i] = c[1]
            table.number_experiments.iloc[i] = len(peak_dg[(peak_dg.area==a)&(peak_dg.tld1_name==c[0])&(peak_dg.depth_range==c[1])].experiment_container_id.unique())
            table.number_cells_dg.iloc[i] = len(peak_dg[(peak_dg.area==a)&(peak_dg.tld1_name==c[0])&(peak_dg.depth_range==c[1])])
            table.responsive_cells_dg.iloc[i] = len(peak_dg[(peak_dg.area==a)&(peak_dg.tld1_name==c[0])&(peak_dg.depth_range==c[1])&(peak_dg.responsive_dg)])
            if len(peak_dg[(peak_dg.area==a)&(peak_dg.tld1_name==c[0])&(peak_dg.depth_range==c[1])])>0:
                table.percent_dg.iloc[i] = len(peak_dg[(peak_dg.area==a)&(peak_dg.tld1_name==c[0])&(peak_dg.depth_range==c[1])&(peak_dg.responsive_dg)])/float(len(peak_dg[(peak_dg.area==a)&(peak_dg.tld1_name==c[0])&(peak_dg.depth_range==c[1])]))


            table.number_cells_sg.iloc[i] = len(peak_sg[(peak_sg.area==a)&(peak_sg.tld1_name==c[0])&(peak_sg.depth_range==c[1])])
            table.responsive_cells_sg.iloc[i] = len(peak_sg[(peak_sg.area==a)&(peak_sg.tld1_name==c[0])&(peak_sg.depth_range==c[1])&(peak_sg.responsive_sg)])
            table.number_cells_ns.iloc[i] = len(peak_ns[(peak_ns.area==a)&(peak_ns.tld1_name==c[0])&(peak_ns.depth_range==c[1])])
            table.responsive_cells_ns.iloc[i] = len(peak_ns[(peak_ns.area==a)&(peak_ns.tld1_name==c[0])&(peak_ns.depth_range==c[1])&(peak_ns.responsive_ns)])

    table_sg = pd.DataFrame(columns=('area','cre','depth','number_experiments','number_cells_sg','responsive_cells_sg', 'percent_sg'), index=range(114))
    for ai, a in enumerate(areas):
        for ci, c in enumerate(cre_depth):
            i = (ci)+(19*ai)
            table_sg.area.iloc[i] = a
            table_sg.cre.iloc[i] = c[0]
            table_sg.depth.iloc[i] = c[1]
            table_sg.number_experiments.iloc[i] = len(peak_sg[(peak_sg.area==a)&(peak_sg.tld1_name==c[0])&(peak_sg.depth_range==c[1])].experiment_container_id.unique())
            table_sg.number_cells_sg.iloc[i] = len(peak_sg[(peak_sg.area==a)&(peak_sg.tld1_name==c[0])&(peak_sg.depth_range==c[1])])
            table_sg.responsive_cells_sg.iloc[i] = len(peak_sg[(peak_sg.area==a)&(peak_sg.tld1_name==c[0])&(peak_sg.depth_range==c[1])&(peak_sg.responsive_sg)])
            if len(peak_sg[(peak_sg.area==a)&(peak_sg.tld1_name==c[0])&(peak_sg.depth_range==c[1])])>0:
                table_sg.percent_sg.iloc[i] = len(peak_sg[(peak_sg.area==a)&(peak_sg.tld1_name==c[0])&(peak_sg.depth_range==c[1])&(peak_sg.responsive_sg)])/float(len(peak_sg[(peak_sg.area==a)&(peak_sg.tld1_name==c[0])&(peak_sg.depth_range==c[1])]))


    table_ns = pd.DataFrame(columns=('area','cre','depth','number_experiments','number_cells_ns','responsive_cells_ns', 'percent_ns'), index=range(114))
    for ai, a in enumerate(areas):
        for ci, c in enumerate(cre_depth):
            i = (ci)+(19*ai)
            table_ns.area.iloc[i] = a
            table_ns.cre.iloc[i] = c[0]
            table_ns.depth.iloc[i] = c[1]
            table_ns.number_experiments.iloc[i] = len(peak_ns[(peak_ns.area==a)&(peak_ns.tld1_name==c[0])&(peak_ns.depth_range==c[1])].experiment_container_id.unique())
            table_ns.number_cells_ns.iloc[i] = len(peak_ns[(peak_ns.area==a)&(peak_ns.tld1_name==c[0])&(peak_ns.depth_range==c[1])])
            table_ns.responsive_cells_ns.iloc[i] = len(peak_ns[(peak_ns.area==a)&(peak_ns.tld1_name==c[0])&(peak_ns.depth_range==c[1])&(peak_ns.responsive_ns)])
            if len(peak_ns[(peak_ns.area==a)&(peak_ns.tld1_name==c[0])&(peak_ns.depth_range==c[1])])>0:
                table_ns.percent_ns.iloc[i] = len(peak_ns[(peak_ns.area==a)&(peak_ns.tld1_name==c[0])&(peak_ns.depth_range==c[1])&(peak_ns.responsive_ns)])/float(len(peak_ns[(peak_ns.area==a)&(peak_ns.tld1_name==c[0])&(peak_ns.depth_range==c[1])]))

    table_nm = pd.DataFrame(columns=('area','cre','depth','number_experiments','number_cells_nm','responsive_cells_nm', 'percent_nm'), index=range(114))
    for ai, a in enumerate(areas):
        for ci, c in enumerate(cre_depth):
            i = (ci)+(19*ai)
            table_nm.area.iloc[i] = a
            table_nm.cre.iloc[i] = c[0]
            table_nm.depth.iloc[i] = c[1]
            table_nm.number_experiments.iloc[i] = len(peak_nm[(peak_nm.area==a)&(peak_nm.tld1_name==c[0])&(peak_nm.depth_range==c[1])].experiment_container_id.unique())
            table_nm.number_cells_nm.iloc[i] = len(peak_nm[(peak_nm.area==a)&(peak_nm.tld1_name==c[0])&(peak_nm.depth_range==c[1])])
            table_nm.responsive_cells_nm.iloc[i] = len(peak_nm[(peak_nm.area==a)&(peak_nm.tld1_name==c[0])&(peak_nm.depth_range==c[1])&(peak_nm.responsive_nm3)])
            if len(peak_nm[(peak_nm.area==a)&(peak_nm.tld1_name==c[0])&(peak_nm.depth_range==c[1])])>0:
                table_nm.percent_nm.iloc[i] = len(peak_nm[(peak_nm.area==a)&(peak_nm.tld1_name==c[0])&(peak_nm.depth_range==c[1])&(peak_nm.responsive_nm3)])/float(len(peak_nm[(peak_nm.area==a)&(peak_nm.tld1_name==c[0])&(peak_nm.depth_range==c[1])]))



    exp = pd.DataFrame(columns=('experiment_container_id','cre','area','depth','number_cells','responsive_cells','percent'), index=range(424))
    for i,e in enumerate(peak_dg.experiment_container_id.unique()):
        subset = peak_dg[peak_dg.experiment_container_id==e]
        exp.area.loc[i] = subset.area.iloc[0]
        exp.cre.loc[i] = subset.tld1_name.iloc[0]
        exp.depth.loc[i] = subset.depth_range.iloc[0]
        exp.number_cells.loc[i] = len(subset)
        exp.responsive_cells.loc[i] = len(subset[subset.responsive_dg])
        exp.percent.loc[i] = len(subset[subset.responsive_dg])/float(len(subset))

    exp['cre_depth'] = np.NaN
    for index, row in exp.iterrows():
        exp.cre_depth.loc[index] = (row.cre, row.depth)

    exp_sg = pd.DataFrame(columns=('experiment_container_id','cre','area','depth','cre_depth','number_cells','responsive_cells','percent'), index=range(424))
    for i,e in enumerate(peak_sg.experiment_container_id.unique()):
        subset = peak_sg[peak_sg.experiment_container_id==e]
        exp_sg.area.loc[i] = subset.area.iloc[0]
        exp_sg.cre.loc[i] = subset.tld1_name.iloc[0]
        exp_sg.depth.loc[i] = subset.depth_range.iloc[0]
        exp_sg.number_cells.loc[i] = len(subset)
        exp_sg.responsive_cells.loc[i] = len(subset[subset.responsive_sg])
        exp_sg.percent.loc[i] = len(subset[subset.responsive_sg])/float(len(subset))

    for index, row in exp_sg.iterrows():
        exp_sg.cre_depth.loc[index] = (row.cre, row.depth)


    exp_ns = pd.DataFrame(columns=('experiment_container_id','cre','area','depth','cre_depth','number_cells','responsive_cells','percent'), index=range(424))
    for i,e in enumerate(peak_ns.experiment_container_id.unique()):
        subset = peak_ns[peak_ns.experiment_container_id==e]
        exp_ns.area.loc[i] = subset.area.iloc[0]
        exp_ns.cre.loc[i] = subset.tld1_name.iloc[0]
        exp_ns.depth.loc[i] = subset.depth_range.iloc[0]
        exp_ns.number_cells.loc[i] = len(subset)
        exp_ns.responsive_cells.loc[i] = len(subset[subset.responsive_ns])
        exp_ns.percent.loc[i] = len(subset[subset.responsive_ns])/float(len(subset))

    for index, row in exp_ns.iterrows():
        exp_ns.cre_depth.loc[index] = (row.cre, row.depth)


    exp_nm = pd.DataFrame(columns=('experiment_container_id','cre','area','depth','cre_depth','number_cells','responsive_cells','percent'), index=range(424))
    for i,e in enumerate(peak_nm.experiment_container_id.unique()):
        subset = peak_nm[peak_nm.experiment_container_id==e]
        exp_nm.area.loc[i] = subset.area.iloc[0]
        exp_nm.cre.loc[i] = subset.tld1_name.iloc[0]
        exp_nm.depth.loc[i] = subset.depth_range.iloc[0]
        exp_nm.number_cells.loc[i] = len(subset)
        exp_nm.responsive_cells.loc[i] = len(subset[subset.responsive_nm3])
        exp_nm.percent.loc[i] = len(subset[subset.responsive_nm3])/float(len(subset))

    for index, row in exp_nm.iterrows():
        exp_nm.cre_depth.loc[index] = (row.cre, row.depth)

    sns.boxplot(x='cre_depth',y='percent',data=exp_visp,color='lightgray', order = cre_depth_list)


def save_figure(fig, fname, formats=['.png','.pdf'], transparent=False, dpi=300, facecolor=None, **kwargs):
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42

    if 'size' in kwargs.keys():
        fig.set_size_inches(kwargs['size'])

    elif 'figsize' in kwargs.keys():
        fig.set_size_inches(kwargs['figsize'])
    else:
        fig.set_size_inches(fig.get_figwidth(), fig.get_figheight())
        # fig.set_size_inches(11, 8.5)
    for f in formats:
        fig.savefig(
            fname + f,
            transparent=transparent,
            orientation='landscape',
            dpi=dpi
        )


def plot_strip_plot(data_input, area, plot_key, x_label, fig_base_dir='/allen/aibs/mat/gkocker/bob_platform_plots', figname='dg_decoding_performance', Nticks=10, xlim=None, box=False, bar=False, point=False):


    cre_depth_list = [('Emx1-IRES-Cre',100), ('Slc17a7-IRES2-Cre', 100),('Cux2-CreERT2',100),('Vip-IRES-Cre',100),
                  ('Emx1-IRES-Cre',200),('Slc17a7-IRES2-Cre', 200),('Cux2-CreERT2',200),
                  ('Rorb-IRES2-Cre',200),('Scnn1a-Tg3-Cre',200),('Nr5a1-Cre',200),('Sst-IRES-Cre',200),('Vip-IRES-Cre',200),
                  ('Emx1-IRES-Cre',300),('Slc17a7-IRES2-Cre', 300),
                  ('Rbp4-Cre_KL100',300),('Fezf2-CreER',300),('Tlx3-Cre_PL56',300),('Sst-IRES-Cre',300),
                  ('Ntsr1-Cre_GN220',500)]

    cre_color_dict = core.get_cre_colors()
    cre_depth_colors = []
    cre_list = []
    for c in cre_depth_list:
        cre_list.append(c[0].split('-')[0])
        cre_depth_colors.append(cre_color_dict[c[0]])
    cre_depth_palette = sns.color_palette(cre_depth_colors)

    exp_area = data_input[data_input.area==area]
    with sns.axes_style('whitegrid'):
        fig = plt.figure(figsize=(6,16))
        ax = fig.add_subplot(111)

        ax = sns.stripplot(y='cre_depth',x=plot_key,data=exp_area,palette=cre_depth_palette,
                      order=cre_depth_list, size=10)

        if box:
            ax = sns.boxplot(y='cre_depth', x=plot_key, data=exp_area, palette=cre_depth_palette, order=cre_depth_list, whis=np.inf)
        if bar:
            ax = sns.barplot(y='cre_depth', x=plot_key, data=exp_area, palette=cre_depth_palette, order=cre_depth_list, ci=None, alpha=0.5)
        if point:
            ax = sns.pointplot(y='cre_depth', x=plot_key, data=exp_area, palette=cre_depth_palette, order=cre_depth_list,
                             ci=None, markers='|', markersize=30, join=False)


        # if np.amax(exp_area[plot_key].values[np.isfinite(exp_area[plot_key])]) <= 1:
        #     plt.xticks([0,0.2,0.4, 0.6,0.8,1],range(0,101,20))
        #     plt.xlim(-0.05, 1.05)
        # else:

        if xlim is None:
            xmax = int(np.ceil(np.amax(exp_area[plot_key].values[np.isfinite(exp_area[plot_key])])))
            xmin = int(np.floor(np.amin(exp_area[plot_key].values[np.isfinite(exp_area[plot_key])])))
            x_spacing = max(int(np.round( (xmax - xmin) / Nticks)), 1)

            xmin -= np.remainder(xmin, x_spacing)

        else:
            xmin, xmax = xlim
            x_spacing = max(int(np.round( (xmax - xmin) / Nticks)), 1)

        plt.xticks(range(xmin, xmax+x_spacing, x_spacing ) )
        plt.xlim(xmin-0.1, xmax+.1)


        if xmax > 0 and xmin < 0:
            plt.axvline(x=0, lw=1, color='k')

        plt.yticks(range(19), cre_list)
        plt.axhline(y=3.5, lw=7, color='w')
        plt.axhline(y=11.5, lw=7, color='w')
        plt.axhline(y=17.5, lw=7, color='w')
        plt.axhspan(-0.5,3.5, color='#DBDBDB', alpha=0.4)
        plt.axhspan(3.5,11.5, color='#BDBDBD', alpha=0.4)
        plt.axhspan(11.5,17.5, color='#DBDBDB', alpha=0.4)
        plt.axhspan(17.5,19, color='#BDBDBD', alpha=0.4)
        plt.ylabel("")
        ax.yaxis.tick_right()

        plt.tick_params(labelsize=20)
        plt.xlabel(x_label, fontsize=24)
        plt.title(area, fontsize=24)
        fig.tight_layout()

        figname = os.path.join(fig_base_dir, figname+'_'+area)
        save_figure(fig, figname)

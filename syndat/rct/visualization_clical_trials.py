import logging
import os
from typing import List, Optional, Dict, Union

import numpy as np
import pandas as pd
from plotnine import (ggplot, aes, geom_point, geom_smooth,
                      geom_abline, geom_ribbon, geom_bar, geom_line,
                      geom_violin, geom_boxplot, geom_jitter,
                      coord_cartesian, scale_x_continuous, scale_y_continuous,
                      labs, scale_x_log10, scale_y_log10, facet_wrap,
                      theme, element_text, geom_text, element_blank,
                      position_dodge, position_nudge, position_jitter,
                      theme_minimal, element_rect)

ggstyle = theme(
    axis_title=element_text(size=18),
    axis_text=element_text(size=18),
    plot_title=element_text(size=18, ha='center'),
    strip_text=element_text(size=18)
)

logger = logging.getLogger(__name__)


def gof_continuous(plt_dt: pd.DataFrame, var_name: str, strat_vars: Optional[List[str]] = None, log_trans: bool = False) -> ggplot:
    """
    Generates a goodness-of-fit (GOF) plot for continuous variables using observed vs. reconstructed values.

    Produces scatter plots with a smoothing line and identity line. Optionally applies log-transformation 
    and stratification by specified variables.

    :param plt_dt: A pandas DataFrame containing columns 'Observed' and 'Reconstructed', and optionally stratification variables.
    :param var_name: Name of the variable to display in the plot title.
    :param strat_vars: Optional list of column names to stratify the plot using facet wrap.
    :param log_trans: Whether to apply a log10 transformation to the axes.
    :return: A ggplot object representing the GOF plot.
    """
    min_val = min(plt_dt['Observed'].min(), plt_dt['Reconstructed'].min()) - 2
    max_val = max(plt_dt['Observed'].max(), plt_dt['Reconstructed'].max()) + 2

    p = (ggplot(plt_dt, aes(x='Observed', y='Reconstructed')) +
         geom_point(color='black', alpha=0.4) +
         geom_smooth(se=False, color='blue') +
         geom_abline(intercept=0, slope=1, color='red') +
         scale_x_continuous(limits=(min_val, max_val)) +
         scale_y_continuous(limits=(min_val, max_val)) +
         labs(x='Observed', y='Reconstructed', title=var_name) +
         ggstyle)

    if log_trans:
        min_val = max(min_val, 1e-5)
        min_val = 10 ** np.floor(np.log10(min_val))
        max_val = 10 ** np.ceil(np.log10(max_val + 2))

        p = p + scale_x_log10(limits=(min_val, max_val))
        p = p + scale_y_log10(limits=(min_val, max_val))
        p = p + labs(
            x='Observed (log scale)',
            y='Reconstructed (log scale)',
            title=f'Log {var_name}')

    if strat_vars:
        facets = '~' + '+'.join(strat_vars)
        p += facet_wrap(facets)

    return p

def gof_continuous_list(
    rp0: dict,
    dt: pd.DataFrame,
    strat_vars: Optional[List[str]] = None,
    static: bool = False,
    log_trans: Optional[bool] = False,
    save_path: Optional[str] = None,
    width: Optional[int] = 8,
    height: Optional[int] = 6,
    dpi: Optional[int] = 300) -> Dict[str, ggplot]:
    """
    Creates a dictionary of goodness-of-fit (GOF) plots for a list of continuous variables.
    Saves or displays each plot depending on whether a path is provided.

    :param rp0: Dictionary with a key 'long_cont' containing a list of continuous variable names.
    :param dt: pd.DataFrame with columns including 'REPI', 'TYPE', 'Variable', 'DV', 'SUBJID', 'TIME' (if static = False),
               and optionally stratification variables.
    :param strat_vars: Optional list of column names to stratify each plot (faceted visualization).
    :param static: If True, metrics for static variables will be calculated.
    :param log_trans: If True, applies log10 transformation to both axes in the plots.
    :param save_path: Optional path to a folder. If provided, saves each plot as a PNG file.
                      If not provided, plots will be shown interactively.
    :param width: Width of the saved plot in inches (used only if save_path is provided).
    :param height: Height of the saved plot in inches (used only if save_path is provided).
    :param dpi: Resolution (dots per inch) of the saved plot (used only if save_path is provided).
    :return: A dictionary where keys are variable names and values are ggplot GOF plots.
    """
    if static:
        col_name = 'static_cont'
        TIME_V = []
    else:
        col_name = 'long_cont'
        TIME_V = ['TIME']

    plot_data = (dt[(dt['REPI'] == 1) &
                    (dt['TYPE'].isin(["Observed", "Reconstructed"])) &
                    (dt['Variable'].isin(rp0[col_name]))]
                    .loc[:, ["Variable", "DV", "SUBJID", "TYPE"] + TIME_V + (strat_vars or [])]
                    .pivot_table(index=["SUBJID", "Variable"] + TIME_V + (strat_vars or []),
                                columns="TYPE", values="DV")
                    .reset_index()
                    .dropna(subset=["Observed"]))
    gof_list = {}
    log_name = "Log" if log_trans else ""
    for var in rp0[col_name]:
        plot = gof_continuous(
            plt_dt=plot_data[plot_data['Variable'] == var],
            var_name=var,
            strat_vars=strat_vars,
            log_trans=log_trans
        )
        gof_list[var] = plot

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filename = os.path.join(save_path, '%s_%sgof_plot.png'%(var,log_name))
            plot.save(filename=filename, width=width, height=height, dpi=dpi)
        else:
            print(plot)
    return gof_list

def gof_binary_list(
    rp0: dict,
    dt: pd.DataFrame,
    strat_vars: Optional[List[str]] = None,
    static: bool = False,
    save_path: Optional[str] = None,
    width: Optional[int] = 8,
    height: Optional[int] = 6,
    dpi: Optional[int] = 300) -> Dict[str, ggplot]:
    """
    Creates goodness-of-fit (calibration) plots for binary variables by comparing
    the proportion of observed vs. reconstructed outcomes over time (in %).

    :param rp0: Dictionary with a key 'long_bin' containing a list of binary variable names.
    :param dt: pd.DataFrame with at least columns 'REPI', 'TYPE', 'Variable', 'DV', 'TIME' (if static = False),
               and optionally stratification variables.
    :param strat_vars: Optional list of column names for stratified (faceted) plots.
    :param static: If True, metrics for static variables will be calculated.
    :param save_path: Optional path to a folder. If provided, saves each plot as a PNG.
                      If not provided, plots will be shown interactively.
    :param width: Width of the saved plot in inches (used only if save_path is provided).
    :param height: Height of the saved plot in inches (used only if save_path is provided).
    :param dpi: Resolution (dots per inch) of the saved plot (used only if save_path is provided).
    :return: Dictionary mapping each variable name to its ggplot object.
    """
    if static:
        col_name = 'static_bin'
        TIME_V = []
    else:
        col_name = 'long_cont'
        TIME_V = ['TIME']

    logger.info("This plot applies only to binary endpoints and illustrates the calibration"
          " of the percentage of subjects who achieved the outcome value 1 (e.g., responders).")
    df = dt[(dt["REPI"] == 1) &
            (dt['TYPE'].isin(["Observed", "Reconstructed"])) &
            (dt['Variable'].isin(rp0[col_name]))]
    observed_keys = df[df['TYPE'] == 'Observed'][['SUBJID', 'Variable'] + TIME_V]
    df = df.merge(observed_keys.drop_duplicates(), on=['SUBJID', 'Variable'] + TIME_V, how='inner')
    plot_data = (df.groupby(TIME_V + ['Variable'] + (strat_vars or []))
                   .agg(Observed=('DV', lambda x: 100 * (x[df['TYPE'] == 'Observed'].sum() / len(x))),
                        Reconstructed=('DV', lambda x: 100 * (x[df['TYPE'] == 'Reconstructed'].sum() / len(x))))
                   .reset_index())

    gof_list = {}
    for var in rp0[col_name]:
        plot = gof_continuous(
            plt_dt=plot_data[plot_data['Variable'] == var],
            var_name=var,
            strat_vars=strat_vars,
            log_trans=False)
        gof_list[var] = plot

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filename = os.path.join(save_path, '%s_gof_bin_plot.png'%(var))
            plot.save(filename=filename, width=width, height=height, dpi=dpi)
        else:
            print(plot)
    return gof_list


def bin_traj_time_list(
    rp0: dict,
    dt: pd.DataFrame,
    mode: str = "Reconstructed",
    dt_cs: Optional[pd.DataFrame] = None,
    strat_vars: Optional[List[str]] = None,
    time_unit: Optional[str] = "Months",
    save_path: Optional[str] = None,
    width: Optional[int] = 8,
    height: Optional[int] = 6,
    dpi: Optional[int] = 300) -> Dict[str, ggplot]:
    """
    Creates trajectories plots of the percentage of subjects who achieved the outcome
    value 1 (e.g., responders).

    :param rp0: Dictionary with a key 'long_bin' containing a list of binary variable names.
    :param dt: pd.DataFrame with at least columns 'REPI', 'TYPE', 'Variable', 'DV', 'TIME',
               and optionally stratification variables.
    :param mode: String, usually "Reconstructed", used for filtering TYPE.
    :param dt_cs: Optional counterfactual DataFrame.
    :param strat_vars: Optional list of column names for stratified (faceted) plots.
    :param time_unit: A string representing the unit of time to display on the x-axis label
                  (e.g., "Months", "Days", "Hours").
    :param save_path: Optional path to a folder. If provided, saves each plot as a PNG.
                      If not provided, plots will be shown interactively.
    :param width: Width of the saved plot in inches (used only if save_path is provided).
    :param height: Height of the saved plot in inches (used only if save_path is provided).
    :param dpi: Resolution (dots per inch) of the saved plot (used only if save_path is provided).
    :return: Dictionary mapping each variable name to its ggplot object.
    """
    if mode not in ["Reconstructed", "Simulations"]:
        raise ValueError(f"`mode` must be either 'Reconstructed' or 'Simulations', got '{mode}'")

    logger.info("This plot applies only to binary endpoints and illustrates the calibration" \
          " of the percentage of subjects who achieved the outcome value 1 (e.g., responders)" \
          " along all time points")

    strat_vars = strat_vars or []
    plot_data = (dt[(dt["REPI"] == 1 if mode == "Reconstructed" else True) &
            (dt['TYPE'].isin(["Observed", mode])) &
            (dt['Variable'].isin(rp0['long_bin']))]
            .groupby(strat_vars + ['Variable', 'TYPE', 'TIME'])
            .agg(Rate=('DV', lambda x: 100 * (x.sum() / len(x))))
            .reset_index())

    if dt_cs is not None:
        plot_data_cs = (dt_cs[(dt_cs["REPI"] == 1 if mode == "Reconstructed" else True) &
                (dt_cs['TYPE'].isin([mode])) &
                (dt_cs['Variable'].isin(rp0['long_bin']))]
                .assign(TYPE="Counterfactual")
                .groupby(strat_vars + ['Variable', 'TYPE', 'TIME'])
                .agg(Rate=('DV', lambda x: 100 * (x.sum() / len(x))))
                .reset_index())
        plot_data = pd.concat([plot_data, plot_data_cs])

    cs_name = "counterfactual_" if dt_cs is not None else ""    
    plot_list = {}
    for var in rp0['long_bin']:
        plot = trajectory_plot(
            plt_dt=plot_data[plot_data['Variable'] == var],
            var_name=var,
            strat_vars=strat_vars,
            time_unit=time_unit,
            achievement_plot=True)
        plot_list[var] = plot

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filename = os.path.join(save_path, '%s_%sbin_time_plot.png'%(var,cs_name))
            plot.save(filename=filename, width=width, height=height, dpi=dpi)
        else:
            print(plot)
    return plot_list


def bar_categorical(
    plt_dt: pd.DataFrame,
    var_name: str,
    type_: str,
    strat_vars: Optional[List[str]] = None) -> ggplot:
    """
    Generates a bar chart for a categorical variable comparing observed vs reconstructed distributions.

    :param plt_dt: DataFrame with one categorical variable to plot.
    :param var_name: Name of the variable to use as title.
    :param type_: "Percentage" or "Subjects" to define the bar heights.
    :param strat_vars: Optional list of variables to use for facetting.
    :return: ggplot object.
    """

    if "TIME" in strat_vars and len(plt_dt.TIME.unique())>15:
        logger.warning("'TIME' is a stratification variable with more than 15 values. The plot may not be optimal.")

    df = (plt_dt.groupby(['DV', 'TYPE'] + (strat_vars or []))
                .size()
                .reset_index(name='N'))
    df['PERC'] = df['N'] / df.groupby(['TYPE'] + (strat_vars or []))['N'].transform('sum') * 100

    if strat_vars:
        group_cols = ['DV', 'TYPE'] + strat_vars
        df_grouped = df.groupby(group_cols, as_index=False).sum()
 
        dv_vals = sorted(df_grouped['DV'].unique())
        type_vals = sorted(df_grouped['TYPE'].unique())
 
        strat_vals = sorted(df[strat_vars[0]].unique())
        full_index = pd.MultiIndex.from_product(
            [dv_vals, type_vals, strat_vals],
            names=['DV', 'TYPE'] + strat_vars
        )
        df = df_grouped.set_index(group_cols).reindex(full_index, fill_value=0).reset_index()
    else:
        all_dv = df['DV'].unique()
        all_types = df['TYPE'].unique()
 
        full_index = pd.MultiIndex.from_product(
            [all_dv, all_types], names=['DV', 'TYPE']
        )
 
        df = df.set_index(['DV', 'TYPE']).reindex(full_index, fill_value=0).reset_index()

    if type_ == "Percentage":
        p = (ggplot(df, aes(x='DV', y='PERC', fill='TYPE')) +
             geom_bar(stat='identity', position=position_dodge(width=1)) +
             geom_text(aes(label='round(PERC, 1)', y='PERC + 5'),
                       position=position_dodge(width=1)) +
             labs(x='', y='Percentage of subjects (%)', title=var_name) +
             ggstyle)
    elif type_ == "Subjects":
        p = (ggplot(df, aes(x='DV', y='N', fill='TYPE')) +
             geom_bar(stat='identity', position=position_dodge(width=1)) +
             geom_text(aes(label='N', y='N + 5'),
                       position=position_dodge(width=1)) +
             labs(x='', y='Number of subjects', title=var_name) +
             ggstyle)

    if strat_vars:
        facets = '~' + '+'.join(strat_vars)
        p += facet_wrap(facets)

    return p

def bar_categorical_list(
    rp0: dict,
    dt: pd.DataFrame,
    type_: str = "Percentage",
    dt_cs: Optional[pd.DataFrame] = None,
    strat_vars: Optional[List[str]] = None,
    static: bool = False,
    save_path: Optional[str] = None,
    width: Optional[int] = 8,
    height: Optional[int] = 6,
    dpi: Optional[int] = 300) -> Dict[str, ggplot]:
    """
    Generates and optionally saves bar plots for all categorical variables listed in rp0.

    :param rp0: Dictionary with a key 'long_cat' containing a list of categorical variable names.
    :param dt: DataFrame with the columns 'REPI', 'TYPE', 'Variable', 'DV', 'SUBJID', 'TIME' (if static = False) and optionally others.
    :param type_: "Percentage" or "Subjects" to define the bar heights.
    :param dt_cs: Optional counterfactual DataFrame.
    :param strat_vars: Optional list of variables to use for facetting.
    :param static: If True, metrics for static variables will be calculated.
    :param save_path: Optional path to folder where plots should be saved. If not provided, plots are shown.
    :param width: Width of the saved plot in inches (used only if save_path is provided).
    :param height: Height of the saved plot in inches (used only if save_path is provided).
    :param dpi: Resolution (dots per inch) of the saved plot (used only if save_path is provided).
    :return: Dictionary of ggplot objects keyed by variable name.
    """
    if static:
        col_name = 'static_cat'
        TIME_V = []
    else:
        col_name = 'long_cat'
        TIME_V = ['TIME']

    if dt_cs is not None:
        if type_ != "Percentage":
            raise ValueError("When 'dt_cs' is provided, 'type_' must be 'Percentage' to allow comparison.")

    df = dt[(dt["REPI"] == 1) &
            (dt['TYPE'].isin(["Observed", "Reconstructed"])) &
            (dt['Variable'].isin(rp0[col_name]))]
    observed_keys = df[df['TYPE'] == 'Observed'][['SUBJID', 'Variable'] + TIME_V]
    df = df.merge(observed_keys.drop_duplicates(), on=['SUBJID', 'Variable']  + TIME_V, how='inner')
    df = df.loc[:, ["Variable", "DV", "SUBJID", "TYPE"] + TIME_V + (strat_vars or [])]

    if dt_cs is not None:
        dt_cs = (dt_cs[(dt_cs["REPI"] == 1) &
                (dt_cs['TYPE'].isin(["Reconstructed"])) &
                (dt_cs['Variable'].isin(rp0[col_name]))]
                .assign(TYPE="Counterfactual")
                .merge(observed_keys.drop(columns=["SUBJID"]).drop_duplicates(), on= TIME_V + ["Variable"], how="inner")
                .reset_index())    
        dt_cs = dt_cs.loc[:, ["Variable", "DV", "SUBJID", "TYPE"] + TIME_V + (strat_vars or [])]
        df = pd.concat([df, dt_cs])

    if "TIME" in strat_vars:
        df = df.loc[:, ~df.columns.duplicated()]

    name_ = "perc" if type_ == "Percentage" else "subj"
    cs_name = "counterfactual_" if dt_cs is not None else ""
    gof_list = {}
    for var in rp0[col_name]:
        plot = bar_categorical(
            plt_dt=df[df['Variable'] == var],
            var_name=var,
            type_=type_,
            strat_vars=strat_vars)
        gof_list[var] = plot

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filename = os.path.join(save_path, '%s_%sbar_cat_%s_plot.png'%(var,cs_name,name_))
            plot.save(filename=filename, width=width, height=height, dpi=dpi)
        else:
            print(plot)
    return gof_list

def assign_visit_absolute(dat_vpc: Union[np.ndarray, pd.Series],
                          Visits: Union[np.ndarray, pd.Series],
                          geometric: bool = False) -> np.ndarray:
    """
    Assigns each time point in `dat_vpc` to a visit bin defined by `Visits`, using linear or geometric spacing.

    :param dat_vpc: Array or Series of time points to bin.
    :param Visits: Array or Series of reference visit times.
    :param geometric: If True, uses geometric (log-space) binning.
    :return: Array of assigned visit values.
    """
    sVisits = np.sort(Visits)
    if geometric:
        sVisits = np.log(sVisits)
        cuts = np.concatenate(([-np.inf], np.exp(sVisits[1:] - np.diff(sVisits) / 2), [np.inf]))
    else:
        cuts = np.concatenate(([-np.inf], sVisits[1:] - np.diff(sVisits) / 2, [np.inf]))
    return sVisits[np.digitize(dat_vpc, cuts, right=True) - 1]

def trajectory_plot(
    plt_dt: pd.DataFrame,
    var_name: str,
    strat_vars: Optional[List[str]] = None,
    time_unit: Optional[str] = "Months",
    achievement_plot: Optional[bool] = False) -> ggplot:
    """
    Creates a ribbon plot of the median and 5th-95th percentiles of a continuous variable over time.

    :param plt_dt: DataFrame with summary statistics ('med', 'p5', 'p95') by Visit and TYPE.
    :param var_name: Name of the variable to use as the plot title.
    :param strat_vars: Optional list of variables to use for facetting.
    :param time_unit: A string representing the unit of time to display on the x-axis label
                  (e.g., "Months", "Days", "Hours").
    :param achievement_plot: If True, plot percentage of subjects achieving the outcome over time;
                             if False, plot median with ribbons.
    :return: ggplot object.
    """

    if achievement_plot:
        p = (ggplot(plt_dt, aes(x='TIME', y='Rate', color='TYPE', group='TYPE')) +
            geom_line(size=1) +
            labs(x=f"Time ({time_unit})", y="Percentage of subjects who achieved outcome",
                 title=var_name, fill=None, color=None) +
            coord_cartesian() +
            theme(legend_title=element_blank(), legend_position="top"))
    else:
        p = (ggplot(plt_dt, aes(x='Visit', y='med', color='TYPE', group='TYPE')) +
            geom_ribbon(aes(ymin='p5', ymax='p95', fill='TYPE'), alpha=0.2) +
            geom_line(size=1) +
            labs(x=f"Time ({time_unit})", y="Value", title=var_name, fill=None, color=None) +
            coord_cartesian() +
            theme(legend_title=element_blank(), legend_position="top"))

    if strat_vars:
        facets = '~' + '+'.join(strat_vars)
        p += facet_wrap(facets)
    
    return p

def trajectory_plot_list(
    rp0: dict,
    dt: pd.DataFrame,
    mode: str = "Reconstructed",
    bins: Optional[np.ndarray] = None,
    dt_cs: Optional[pd.DataFrame] = None,
    strat_vars: Optional[List[str]] = None,
    time_unit: Optional[str] = "Months",
    save_path: Optional[str] = None,
    width: Optional[int] = 8,
    height: Optional[int] = 6,
    dpi: Optional[int] = 300) -> Dict[str, ggplot]:
    """
    Generates and optionally saves ribbon plots for continuous variables across visits.

    :param rp0: Dictionary with key 'long_cont' containing a list of variable names to plot.
    :param dt: Main DataFrame containing data for "Observed" and `mode`.
    :param mode: String, usually "Reconstructed", used for filtering TYPE.
    :param bins: Optional array of visit cutoffs. If None, uses unique TIME values in `dt`.
    :param dt_cs: Optional counterfactual DataFrame.
    :param strat_vars: Optional list of stratification variables for facetting.
    :param time_unit: A string representing the unit of time to display on the x-axis label
                  (e.g., "Months", "Days", "Hours").
    :param save_path: Optional path to save plots. If None, plots are printed to console.
    :param width: Width of the saved plot in inches (used only if save_path is provided).
    :param height: Height of the saved plot in inches (used only if save_path is provided).
    :param dpi: Resolution (dots per inch) of the saved plot (used only if save_path is provided).
    :return: Dictionary of ggplot objects keyed by variable name.
    """
    if mode not in ["Reconstructed", "Simulations"]:
        raise ValueError(f"`mode` must be either 'Reconstructed' or 'Simulations', got '{mode}'")

    if bins is None:
        logger.info("No bins were given. Using all time points")
        bins = dt['TIME'].unique()
    strat_vars = strat_vars or []
    dt['Visit'] = assign_visit_absolute(dt['TIME'], bins)

    plot_data = (dt[(dt["REPI"] == 1 if mode == "Reconstructed" else True) & 
                    (dt['TYPE'].isin(["Observed", mode])) & 
                    (dt['Variable'].isin(rp0['long_cont']))]
                    .groupby(strat_vars + ['Variable', 'TYPE', 'Visit'])
                    .agg(med=('DV', 'median'), p5=('DV', lambda x: np.percentile(x, 5)), p95=('DV', lambda x: np.percentile(x, 95)))
                    .reset_index())

    if dt_cs is not None:
        dt_cs['Visit'] = assign_visit_absolute(dt_cs['TIME'], bins)
        plot_data_cs = (dt_cs[(dt_cs["REPI"] == 1 if mode == "Reconstructed" else True) & 
                    (dt_cs['TYPE'].isin([mode])) & 
                    (dt_cs['Variable'].isin(rp0['long_cont']))]
                    .assign(TYPE="Counterfactual")
                    .groupby(strat_vars + ['Variable', 'TYPE', 'Visit'])
                    .agg(med=('DV', 'median'), p5=('DV', lambda x: np.percentile(x, 5)), p95=('DV', lambda x: np.percentile(x, 95)))
                    .reset_index())
        plot_data = pd.concat([plot_data, plot_data_cs])

    cs_name = "counterfactual_" if dt_cs is not None else ""
    plot_list = {}
    for var in rp0['long_cont']:
        plot = trajectory_plot(
            plt_dt=plot_data[plot_data['Variable'] == var],
            var_name=var,
            strat_vars=strat_vars,
            time_unit=time_unit)
        plot_list[var] = plot

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filename = os.path.join(save_path, '%s_%strajectory_plot.png'%(var,cs_name))
            plot.save(filename=filename, width=width, height=height, dpi=dpi)
        else:
            print(plot)

    return plot_list


def raincloud_plot(
    plt_dt: pd.DataFrame,
    var_name: str,
    strat_vars: Optional[List[str]] = None) -> ggplot:
    """
    Generates a raincloud plot (violin + boxplot + jitter) comparing Observed vs Reconstructed data.

    :param dt: DataFrame with columns 'TYPE', 'DV' and optional stratification vars.
    :param var_name: Name of the variable to use as the plot title.
    :param strat_vars: Optional list of variables to use for facetting.
    :return: ggplot object.
    """
    plt_dt["TYPE"] = pd.Categorical(plt_dt["TYPE"], categories=["Observed", "Reconstructed"])

    p = (
        ggplot(plt_dt, aes(x='TYPE', y='DV', fill='TYPE', color='TYPE')) +
        geom_violin(width=0.6, alpha=0.3, position=position_nudge(x=-0.2)) +  # stat_halfeye alternative
        geom_boxplot(width=0.15, outlier_shape=None, alpha=0.7) +
        geom_jitter(aes(fill='TYPE'), position=position_jitter(width=0.1), alpha=0.2, shape='o') +
        labs(title=var_name, x=None, y=None) +
        theme_minimal() +
        theme(
            legend_title=element_blank(),
            legend_position='top',
            plot_title=element_text(size=18, ha='center'),
            axis_text=element_text(size=14),
            axis_title=element_text(size=16),
            panel_background=element_rect(fill="white", colour="white"),
            plot_background=element_rect(fill="white", colour="white"),
            legend_background=element_rect(fill="white", colour="white"),
            strip_background=element_rect(fill="white", color="white")
        )
    )

    if strat_vars:
        facets = '~' + '+'.join(strat_vars)
        p += facet_wrap(facets)

    return p


def raincloud_continuous_list(
    rp0: dict,
    dt: pd.DataFrame,
    static: bool = False,
    strat_vars: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    width: Optional[int] = 8,
    height: Optional[int] = 6,
    dpi: Optional[int] = 300) -> Dict[str, ggplot]:

    """
    Generates and optionally saves raincloud plots for continuous observed vs reconstructed variables

    :param rp0: Dictionary with a key 'long_cont', 'static_cont' containing a list of continuous variable names.
    :param dt: DataFrame with the columns 'REPI', 'TYPE', 'Variable', 'DV', 'SUBJID', 'TIME' and optionally others.
    :param static: If True, metrics for static variables will be calculated.
    :param strat_vars: Optional list of variables to use for facetting.
    :param save_path: Optional path to folder where plots should be saved. If not provided, plots are shown.
    :param width: Width of the saved plot in inches (used only if save_path is provided).
    :param height: Height of the saved plot in inches (used only if save_path is provided).
    :param dpi: Resolution (dots per inch) of the saved plot (used only if save_path is provided).
    :return: Dictionary of ggplot objects keyed by variable name.
    """

    if static:
        col_name = 'static_cont'
        TIME_V = []
    else:
        col_name = 'long_cont'
        TIME_V = ['TIME']

    plot_list = {}
    plot_data = (
        dt[(dt["REPI"] == 1) & (dt["TYPE"].isin(["Observed", "Reconstructed"])) & (dt["Variable"].isin(rp0[col_name]))]
        .loc[:, ["Variable", "DV", "SUBJID", "TYPE"] + TIME_V + (strat_vars or [])]
        .pivot(index=["SUBJID", "Variable"] + TIME_V + (strat_vars or []), 
                columns="TYPE", values="DV")
        .reset_index()
        .dropna(subset=["Observed"])
    )
    for var in rp0[col_name]:
        plot = raincloud_plot(
            plt_dt=plot_data[plot_data["Variable"] == var]
            .melt(id_vars=["SUBJID", "Variable"] + TIME_V + (strat_vars or []),
                    value_vars=["Observed", "Reconstructed"],
                    var_name="TYPE",
                    value_name="DV"),
            var_name=var,
            strat_vars=strat_vars
        )
        plot_list[var] = plot

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filename = os.path.join(save_path, '%s_raincloud_plot.png'%(var))
            plot.save(filename=filename, width=width, height=height, dpi=dpi)
        else:
            print(plot)

    return plot_list

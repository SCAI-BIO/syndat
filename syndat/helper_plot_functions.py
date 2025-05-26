from plotnine import (ggplot, aes, geom_point, geom_smooth,
                      geom_abline, geom_ribbon, geom_bar, geom_line,
                      geom_violin, geom_boxplot, geom_jitter,
                      coord_cartesian, scale_x_continuous, scale_y_continuous,
                      labs, scale_x_log10, scale_y_log10, facet_wrap,
                      theme, element_text, geom_text, element_blank,
                      position_dodge, position_nudge, position_jitter, 
                      theme_minimal, element_rect)
import numpy as np
import pandas as pd

ggstyle = theme(
    axis_title=element_text(size=18),
    axis_text=element_text(size=18),
    plot_title=element_text(size=18, ha='center'),
    strip_text=element_text(size=18)
)


def gof_con(plt_dt, var_name, strat_vars=None, log_trans=False):
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

        p += (scale_x_log10(limits=(min_val, max_val)) +
              scale_y_log10(limits=(min_val, max_val)) +
              labs(x='Observed (log scale)', y='Reconstructed (log scale)', title=f'Log {var_name}'))

    if strat_vars:
        facets = '~' + '+'.join(strat_vars)
        p += facet_wrap(facets)

    return p

def gof_con_list(rp0,dt,strat_vars=None,log_trans=False):

    plot_data = (dt[(dt['REPI'] == 1) &
                    (dt['TYPE'].isin(["Observed", "Reconstructed"])) &
                    (dt['Variable'].isin(rp0['long_cont']))]
                    .loc[:, ["Variable", "DV", "SUBJID", "TIME", "TYPE"] + (strat_vars or [])]
                    .pivot_table(index=["SUBJID", "TIME", "Variable"] + (strat_vars or []),
                                columns="TYPE", values="DV")
                    .reset_index()
                    .dropna(subset=["Observed"]))

    gof_list = {var: gof_con(plt_dt=plot_data[plot_data['Variable'] == var],
                             var_name=var,
                             strat_vars=strat_vars,
                             log_trans=log_trans)
                for var in rp0['long_cont']}
    return gof_list

def gof_bin_list(rp0, dt, strat_vars=None):
    df = dt[(dt['REPI'] == 1) &
            (dt['TYPE'].isin(["Observed", "Reconstructed"])) &
            (dt['Variable'].isin(rp0['long_bin']))]

    plot_data = (df.groupby(['TIME', 'Variable'] + (strat_vars or []))
                   .agg(Observed=('DV', lambda x: 100 * (x[df['TYPE'] == 'Observed'].sum() / len(x))),
                        Reconstructed=('DV', lambda x: 100 * (x[df['TYPE'] == 'Reconstructed'].sum() / len(x))))
                   .reset_index())

    gof_list = {var: gof_con(plt_dt=plot_data[plot_data['Variable'] == var],
                             var_name=f"% {var}",
                             strat_vars=strat_vars,
                             log_trans=False)
                for var in rp0['long_bin']}
    return gof_list

def gof_cat(plt_dt, var_name, type_, strat_vars=None):
    df = (plt_dt.groupby(['DV', 'TYPE'] + (strat_vars or []))
                .size()
                .reset_index(name='N'))

    df['PERC'] = df['N'] / df.groupby(['TYPE'] + (strat_vars or []))['N'].transform('sum') * 100

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

def gof_cat_list(rp0, dt, type_="Percentage", strat_vars=None):
    df = dt[(dt['REPI'] == 1) &
            (dt['TYPE'].isin(["Observed", "Reconstructed"])) &
            (dt['Variable'].isin(rp0['long_cat']))]

    df = df.loc[:, ["Variable", "DV", "SUBJID", "TIME", "TYPE"] + (strat_vars or [])]

    gof_list = {var: gof_cat(plt_dt=df[df['Variable'] == var],
                             var_name=var,
                             type_=type_,
                             strat_vars=strat_vars)
                for var in rp0['long_cat']}
    return gof_list

def assign_visit_absolute(dat_vpc, Visits, geometric=False):
    sVisits = np.sort(Visits)
    if geometric:
        sVisits = np.log(sVisits)
        cuts = np.concatenate(([-np.inf], np.exp(sVisits[1:] - np.diff(sVisits) / 2), [np.inf]))
    else:
        cuts = np.concatenate(([-np.inf], sVisits[1:] - np.diff(sVisits) / 2, [np.inf]))
    return sVisits[np.digitize(dat_vpc, cuts, right=True) - 1]

def var_plot(plt_dt, var_name, strat_vars=None):
    p = (ggplot(plt_dt, aes(x='Visit', y='med', color='TYPE', group='TYPE')) +
         geom_ribbon(aes(ymin='p5', ymax='p95', fill='TYPE'), alpha=0.2) +
         geom_line(size=1) +
         labs(x="Time (Months)", y="Value", title=var_name, fill=None, color=None) +
         coord_cartesian() +
         theme(legend_title=element_blank(), legend_position="top"))

    if strat_vars:
        facets = '~' + '+'.join(strat_vars)
        p += facet_wrap(facets)
    
    return p

def var_plot_list(rp0, dt, bins=None, dt_cs=None, mode="Reconstructed", strat_vars=None):
    if bins is None:
        bins = dt['TIME'].unique()
    strat_vars = strat_vars or []
    dt['Visit'] = assign_visit_absolute(dt['TIME'], bins)

    plot_data = (dt[(dt['REPI'] == 1) & 
                    (dt['TYPE'].isin(["Observed", mode])) & 
                    (dt['Variable'].isin(rp0['long_cont']))]
                    .groupby(strat_vars + ['Variable', 'TYPE', 'Visit'])
                    .agg(med=('DV', 'median'), p5=('DV', lambda x: np.percentile(x, 5)), p95=('DV', lambda x: np.percentile(x, 95)))
                    .reset_index())

    if dt_cs is not None:
        dt_cs['Visit'] = assign_visit_absolute(dt_cs['TIME'], bins)
        plot_data_cs = (dt_cs[dt_cs['TYPE'] == mode]
                        .assign(TYPE='Counterfactual')
                        .groupby(strat_vars + ['Variable', 'TYPE', 'Visit'])
                        .agg(med=('DV', 'median'), p5=('DV', lambda x: np.percentile(x, 5)), p95=('DV', lambda x: np.percentile(x, 95)))
                        .reset_index())
        plot_data = pd.concat([plot_data, plot_data_cs])

    plot_list = {var: var_plot(plot_data[plot_data['Variable'] == var], var, strat_vars) for var in rp0['long_cont']}

    return plot_list


def raincloud(plt_dt, var_name, strat_vars=None):
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


def raincloud_con_list(rp0, dt, type="longitudinal", strat_vars=None):

    if type == "longitudinal":
        plot_data = (
            dt[(dt["REPI"] == 1) & (dt["TYPE"].isin(["Observed", "Reconstructed"])) & (dt["Variable"].isin(rp0["long_cont"]))]
            .loc[:, ["Variable", "DV", "SUBJID", "TIME", "TYPE"] + (strat_vars or [])]
            .pivot(index=["SUBJID", "TIME", "Variable"] + (strat_vars or []), 
                   columns="TYPE", values="DV")
            .reset_index()
            .dropna(subset=["Observed"])
        )
        gof_list = {
            var: raincloud(
                plt_dt=plot_data[plot_data["Variable"] == var]
                .melt(id_vars=["SUBJID", "TIME", "Variable"] + (strat_vars or []),
                      value_vars=["Observed", "Reconstructed"],
                      var_name="TYPE",
                      value_name="DV"),
                var_name=var,
                strat_vars=strat_vars
            )
            for var in rp0["long_cont"]
        }

    elif type == "static":
        plot_data = (
            dt[(dt["REPI"] == 1) & (dt["TYPE"].isin(["Observed", "Reconstructed"])) & (dt["Variable"].isin(rp0["static_cont"]))]
            .loc[:, ["Variable", "DV", "SUBJID", "TYPE"] + (strat_vars or [])]
            .pivot(index=["SUBJID", "Variable"] + (strat_vars or []), 
                   columns="TYPE", values="DV")
            .reset_index()
            .dropna(subset=["Observed"])
        )
        gof_list = {
            var: raincloud(
                plt_dt=plot_data[plot_data["Variable"] == var]
                .melt(id_vars=["SUBJID", "Variable"] + (strat_vars or []),
                      value_vars=["Observed", "Reconstructed"],
                      var_name="TYPE",
                      value_name="DV"),
                var_name=var,
                strat_vars=strat_vars
            )
            for var in rp0["static_cont"]
        }

    return gof_list

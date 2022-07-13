"""Utilities for interacting with the moveHMM R package

Module providing a Python interface for functions in moveHMM, but plotting
customized with Pandas/Matplotlib.

"""
import numpy as np
import pandas as pd
import scipy.stats as scistats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

# Rpy2 for working with moveHMM
import rpy2.robjects as robjs
import rpy2.robjects.conversion as cv
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr

# Initialize R instance
r_stats = importr("stats")
r_base = importr("base")
moveHMM = importr("moveHMM")


def prep_data(xy_df, id_col=None, coordNames=["longitude", "latitude"],
              **kwargs):
    """Interface to moveHMM prepData

    Parameters
    ----------
    xy_df : pandas.DataFrame
      Input data, with geographic coordinates.
    id_col : str, optional
      Name of the column having the role of identifying individual data
      sets.  Default uses a column having name "ID".
    coordNames : array_like, optional
      Names of the x- and y-coordinates in `xy_df`.

    Returns
    -------
    move_data : pandas.DataFrame
      Output DataFrame as returned by prepData.

    """
    with cv.localconverter(robjs.default_converter + pandas2ri.converter):
        coordNames = pd.Series(coordNames)

        if id_col is not None:
            xy_df = xy_df.copy()
            xy_df.rename(columns={id_col: "ID"}, inplace=True)

        move_data = moveHMM.prepData(xy_df, coordNames=coordNames, **kwargs)

    return move_data


def fit_HMM(xy_md, nbStates, stepPar0, anglePar0, **kwargs):
    """Interface to moveHMM fit_HMM

    Parameters
    ----------
    xy_md : pandas.DataFrame
      DataFrame complying with moveData structure.
    nbStates : int
      Number of states in the model.
    stepPar0 : pandas.Series
      Initial values for step parameter search.
    anglePar0 : pandas.Series
      Initial values for angle parameter search.
    **kwargs : optional keyword arguments
      Arguments passed to R fitHMM.

    Returns
    -------
    md_fit : moveHMM (R class)

    """
    with cv.localconverter(robjs.default_converter + pandas2ri.converter):
        # Set input prepared data as R object as it has be to a data.frame
        # AND moveData R object to be taken by fitHMM
        md = cv.py2rpy(xy_md)
        md.rclass = robjs.StrVector(("data.frame", "moveData"))
        # Convert objects to R
        stepPar0_r = cv.py2rpy(stepPar0)
        anglePar0_r = cv.py2rpy(anglePar0)

    md_fit = moveHMM.fitHMM(md, nbStates=nbStates, stepPar0=stepPar0_r,
                            anglePar0=anglePar0_r, **kwargs)

    return md_fit


def AIC(*args):
    """Interface to moveHMM AIC

    Parameters
    ----------
    *args : moveHMM (R class)


    Returns
    -------
    out : pandas.DataFrame
      AIC values for each model

    """
    with cv.localconverter(robjs.default_converter + pandas2ri.converter):
        AIC = r_stats.AIC(*args)

    return AIC


def parse_fit(x):
    """Parse moveHMM object into a dictionary

    Parameters
    ----------
    x : moveHMM (R class)
      HMM fit

    Returns
    -------
    dict

    """
    fit_dict = dict(zip(x.names, list(x)))

    for key in ["mle", "mod", "conditions"]:
        fit_dict[key] = dict(zip(fit_dict[key].names,
                                 list(fit_dict[key])))
        if key == "mle":
            fit_dict[key]["delta"] = np.array(fit_dict[key]["delta"])
        elif key == "mod":
            fit_dict[key]["minimum"] = fit_dict[key]["minimum"][0]
            for subkey in ["estimate", "gradient", "hessian"]:
                fit_dict[key][subkey] = np.array(fit_dict[key][subkey])
            for subkey in ["code", "iterations"]:
                fit_dict[key][subkey] = fit_dict[key][subkey][0]
        else:
            conditions_keys = fit_dict[key].keys()
            for subkey in [k for k in conditions_keys if k != "formula"]:
                fit_dict[key][subkey] = fit_dict[key][subkey][0]

    with cv.localconverter(robjs.default_converter + pandas2ri.converter):
        fit_dict["data"] = cv.rpy2py(fit_dict["data"])
        fit_dict["rawCovs"] = cv.rpy2py(fit_dict["rawCovs"])
        fit_dict["knownStates"] = cv.rpy2py(fit_dict["knownStates"])
        fit_dict["nlmTime"] = cv.rpy2py(fit_dict["nlmTime"])
        for key in ["stepPar", "anglePar", "beta"]:
            fit_dict["mle"][key] = r_base.as_data_frame(fit_dict["mle"][key])

    return fit_dict


def plot_fit_hist(x, state_labels, **kwargs):
    """Plot histograms and densities of states in moveHMM object

    This is a partial implementation of moveHMM's plotMoveHMM.

    Parameters
    ----------
    x : moveHMM (R class)
      HMM fit
    state_labels : array_like
      Labels for each state in `x`.
    **kwargs : optional keyword arguments
      Passed to `matplotlib.pyplot.subplots`.

    Notes
    -----
    Assumes that the step and angle distributions are the default Gamma and
    Von Mises.

    Returns
    -------
    fig : Figure
    ax : Axes

    """
    fit_dict = parse_fit(x)
    steps = fit_dict["data"]["step"]
    angles = fit_dict["data"]["angle"]
    step_pars = fit_dict["mle"]["stepPar"]
    n_states = step_pars.shape[1]
    angle_pars = fit_dict["mle"]["anglePar"]
    zeroInflation = fit_dict["conditions"]["zeroInflation"]
    if zeroInflation:
        zero_mass = step_pars.iloc[-1]
        step_pars.drop(step_pars.index[-1], inplace=True)

    states = pd.Series(moveHMM.viterbi(x))

    states_w = (states.value_counts(ascending=True).sort_index() /
                states.shape[0])

    steps_eval = np.linspace(0, fit_dict["data"]["step"].max(), num=10000)
    angles_eval = np.linspace(-np.pi, np.pi, num=1000)

    def calc_step_dens(state_pars, steps):
        shape = state_pars[0] ** 2 / state_pars[1] ** 2
        scale = state_pars[1] ** 2 / state_pars[0]
        return scistats.gamma.pdf(steps, shape, scale=scale)

    def calc_angle_dens(state_pars, angles):
        shape = state_pars[1]
        loc = state_pars[0]
        return scistats.vonmises.pdf(angles, shape, loc=loc)

    step_denss = np.apply_along_axis(calc_step_dens, 0, step_pars,
                                     steps=steps_eval)
    # Weighted by the proportion of each state in the Viterbi states
    # sequence
    for state_idx in np.arange(n_states):
        if zeroInflation:
            step_denss[:, state_idx] *= ((1 - zero_mass.iloc[state_idx]) *
                                         states_w.iloc[state_idx])
        else:
            step_denss[:, state_idx] *= states_w.iloc[state_idx]

    angle_denss = np.apply_along_axis(calc_angle_dens, 0, angle_pars,
                                      angles=angles_eval)
    # Weighted by the proportion of each state in the Viterbi states
    # sequence
    for state_idx in np.arange(n_states):
        if zeroInflation:
            angle_denss[:, state_idx] *= ((1 - zero_mass[state_idx]) *
                                          states_w.iloc[state_idx])
        else:
            angle_denss[:, state_idx] *= states_w.iloc[state_idx]

    fig, axs = plt.subplots(2, **kwargs)
    ax_steps, ax_angles = axs
    ax_steps.set_xlabel("Step length [km]")
    ax_steps.set_ylabel("Density")
    ax_angles.set_xlabel("Turning angle [radians]")
    ax_angles.set_ylabel("Density")
    angle_ticks = np.arange(-np.pi, stop=np.pi + np.pi / 4, step=np.pi / 2)
    angle_tick_labels = [r"$-\pi$", r"$-\frac{\pi}{2}$",
                         "$0$", r"$\frac{\pi}{2}$", r"$\pi$"]
    ax_angles.xaxis.set_major_locator(mticker.FixedLocator(angle_ticks))
    ax_angles.xaxis.set_major_formatter(mticker
                                        .FixedFormatter(angle_tick_labels))
    step_hist_vals, _, _ = ax_steps.hist(steps, bins=14, density=True,
                                         alpha=0.6)

    # Plot densities for each state
    for state_idx, state_label in enumerate(state_labels):
        ax_steps.plot(steps_eval, step_denss[:, state_idx],
                      label=state_label)

    ymax = step_hist_vals.max() * 1.2
    ax_steps.set_ylim(0, ymax)
    ax_steps.legend(loc=9, bbox_to_anchor=(0.5, 1.15),
                    ncol=len(state_labels))

    angle_hist_vals, _, _ = ax_angles.hist(angles, bins=14,
                                           density=True, alpha=0.6)
    for state_idx, state_label in enumerate(state_labels):
        ax_angles.plot(angles_eval, angle_denss[:, state_idx],
                       label=state_label)

    ymax = angle_hist_vals.max() * 1.2  # rewriting object
    ax_angles.set_ylim(0, ymax)         # rewriting object

    return fig, axs


def plot_trprobs(x, covariate_name, state_labels=None, xlabel=None,
                 plot_ci=False, q_ci=0.95, **kwargs):
    """Plot transition probabilities for a given covariate in moveHMM object

    This is a partial implementation of moveHMM's plotMoveHMM.

    Parameters
    ----------
    x : moveHMM (R class)
      HMM fit
    covariate_name : str
      Name of the covariate to plot transition probabilities for.
    state_labels : array_like, optional
      Labels for each state in `x`.
    xlabel : str, optional
      Label for the x-axis. Default is `covariate_name`.
    plot_ci : bool, optional
      Whether to plot confidence intervals.
    q_ci : float, optional
      Quantile level to use for confidence intervals.
    **kwargs : optional keyword arguments
      Passed to `matplotlib.pyplot.subplots`.

    Notes
    -----
    Assumptions:

    1. Assumes that the step and angle distributions are the default Gamma
    and Von Mises.

    2. The model has multiple states.

    Returns
    -------
    fig : Figure
    axs : Axes

    """
    fit_dict = parse_fit(x)
    step_pars = fit_dict["mle"]["stepPar"]
    n_states = step_pars.shape[1]
    beta = fit_dict["mle"]["beta"]
    if fit_dict["conditions"]["zeroInflation"]:
        step_pars.drop(step_pars.index[-1], inplace=True)
    if state_labels is None:
        state_labels = list(map(str, np.arange(1, n_states + 1)))

    if beta.shape[0] > 1:
        # raw_covs = fit_dict["rawCovs"]
        tpm = moveHMM.getPlotData(x, type="tpm", format="wide")
        tpm_d = dict(zip(tpm.names, list(tpm)))
        with cv.localconverter(robjs.default_converter + pandas2ri.converter):
            tpm = cv.rpy2py(tpm_d[covariate_name])

        fig = plt.figure(**kwargs)
        axs = list()
        # Loop over columns in transition probability matrix
        states_prod = np.transpose(np.meshgrid(state_labels,
                                               state_labels, indexing="ij"),
                                   (1, 2, 0)).reshape((-1, 2))
        tpm_hat = tpm.iloc[:, 1:(n_states ** 2) + 1]
        for ax_idx, (probs_lab, probs) in enumerate(tpm_hat.items()):
            ax = fig.add_subplot(n_states, n_states, ax_idx + 1)
            ax.plot(tpm[covariate_name], probs)
            ax.set_ylim(-0.01, 1.01)
            ax.set_ylabel("P({} -> {})".format(*states_prod[ax_idx]))
            if xlabel is None:
                ax.set_xlabel(covariate_name)
            else:
                ax.set_xlabel(xlabel)

            if plot_ci:
                lci = tpm.iloc[:, 1 + ax_idx + n_states ** 2]
                uci = tpm.iloc[:, 1 + ax_idx + n_states ** 2 * 2]
                yerr = np.vstack((probs - lci, uci - probs))
                ax.errorbar(tpm[covariate_name], probs, yerr=yerr,
                            fmt="none", linewidth=1, ecolor="gray")

            axs.append(ax)

        axs = np.array(axs).reshape((n_states, n_states))
        # Share x and y axes
        for ax in axs[0]:
            ax.get_shared_x_axes().join(*axs.flatten())
            ax.set_xticklabels([])
            ax.set_xlabel("")
        for ax in axs[:, 1:].flatten():
            ax.get_shared_y_axes().join(*axs.flatten())
            ax.set_yticklabels([])

        return fig, axs


def plot_states(x, ID, state_labels=None,
                is_datetime_index=True, **kwargs):
    """Plot states and state probabilities.

    Parameters
    ----------
    x : moveHMM (R class)
      HMM fit
    ids : str
      ID in `x` to select for plotting.
    state_labels : array_like, optional
      Labels for each state in `x`.
    is_datetime_index : bool, optional
      Whether index of data is a datetime object.
    **kwargs : optional keyword arguments
      Passed to `matplotlib.pyplot.subplots`.

    Returns
    -------
    fig : Figure
    ax : Axes

    """
    fit_dict = parse_fit(x)
    step_pars = fit_dict["mle"]["stepPar"]
    ids_all = fit_dict["data"]["ID"]
    n_states = step_pars.shape[1]

    if state_labels is None:
        state_labels = list(map(str, np.arange(1, n_states + 1)))

    states = viterbi(x)
    probs = state_probs(x)

    df_all = pd.DataFrame(np.hstack((states[:, np.newaxis], probs)),
                          index=(pd.MultiIndex
                                 .from_frame(ids_all.reset_index())),
                          columns=["state"] + state_labels)
    df_id = df_all[state_labels].xs(ID, level="ID")

    if is_datetime_index:
        df_id.index = pd.DatetimeIndex(df_id.index)

    # Set up plot
    fig, axs = plt.subplots(n_states, 1, sharex=True, sharey=True,
                            **kwargs)
    dlocator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    dformatter = mdates.ConciseDateFormatter(dlocator)

    for ax_idx, ax in enumerate(axs):
        df_id[state_labels[ax_idx]].plot(ax=ax, rot=0)
        ax.xaxis.set_major_locator(dlocator)
        ax.xaxis.set_major_formatter(dformatter)
        ax.set_ylabel("P(state={})".format(state_labels[ax_idx]))
        ax.set_xlabel("")
        ax.autoscale(enable=True, axis="x", tight=True)

    return fig, axs


def viterbi(x):
    """Viterbi algorithm for reconstructing most probable state sequence

    Parameters
    ----------
    x : moveHMM (R class)
      HMM fit

    Returns
    -------
    ndarray
      Integer sequence of states

    """
    with cv.localconverter(robjs.default_converter + numpy2ri.converter):
        states = np.array(moveHMM.viterbi(x), int)

    return states


def state_probs(x):
    """Probabilities of the HMM process being in each state

    Parameters
    ----------
    x : moveHMM (R class)
      HMM fit

    Returns
    -------
    ndarray

    """
    with cv.localconverter(robjs.default_converter + numpy2ri.converter):
        states = np.array(moveHMM.stateProbs(x))

    return states


def sim_movement(n_ids=1, n_states=2, step_distr="gamma", angle_distr="vm",
                 step_pars=None, angle_pars=None, beta=None,
                 covariates=None, n_covariates=None,
                 is_zero_inflated=False, n_per_id=None, model=None,
                 return_states=False):
    """Simulate movement as Hidden Markov Model

    Parameters
    ----------
    n_ids : int, optional
      Number of individuals (IDs) to simulate.
    n_states : int, optional
      Number of states to simulate.
    step_distr : str, optional
      Name of the distribution for the step lengths.  Supported
      distributions are: "gamma", "weibull", "lnorm", and "exp".
    angle_distr : str, optional
      Name of the distribution for the turning angles.  Supported
      distributions are: "vm", "wrpcauchy", "lnorm", and "exp".  Set to
      None if the angle distribution should not be estimated.
    step_pars : array_like, optional
      Parameters for the step length distribution.
    angle_pars : array_like, optional
      Parameters for the turning angle distribution.
    beta : array_like, optional
      Matrix of regression parameters for the transition probabilities
    covariates : DataFrame, optional
      Covariate values to include in the model.  Covariates can also be
      simulated as a standard normal distribution, by setting
      `covariates=None`, and `n_covariates > 0`.
    n_covariates : int, optional
      Number of covariates to simulate; does not need to be specified if
      `covariates` is not None.
    is_zero_inflated : bool, optional
      Whether step length distribution is inflated at zero.
      If `TRUE`, values for the zero-mass parameters should be
      included in `stepPar`.
    n_per_id : int or array_like, optional
      Either the number of the number of observations per animal (integer
      scalar), or the bounds of the number of observations per animal
      (length 2 array or list). In the latter case,  the numbers of
      obervations generated for each animal are uniformly picked from
      this interval.
    model : moveHMM, optional
      Simulation based on a fitted model.  If this argument is specified,
      most other arguments will be ignored, except for n_ids, n_per_id,
      covariates (if covariates are different from those in the data), and
      states.
    return_states : bool, optional
      Whether the simulated states should be returned.

    Returns
    -------
    DataFrame

    """
    none_converter = cv.Converter("None converter")
    none_converter.py2rpy.register(type(None), _none2null)

    with cv.localconverter(robjs.default_converter + pandas2ri.converter +
                           none_converter):
        sim_df = moveHMM.simData(nbAnimals=n_ids, nbStates=n_states,
                                 stepDist=step_distr, angleDist=angle_distr,
                                 stepPar=step_pars, anglePar=angle_pars,
                                 beta=beta, covs=covariates,
                                 nbCovs=n_covariates,
                                 zeroInflation=is_zero_inflated,
                                 obsPerAnimal=n_per_id, model=model,
                                 states=return_states)

    return sim_df


def _none2null(none_obj):
    return robjs.r("NULL")


if __name__ == '__main__':
    # For animation purposes
    from matplotlib.lines import Line2D
    from matplotlib.animation import FuncAnimation

    # Initial parameters for simulation with 3 states
    idx1names = ["Resting", "Searching", "Travelling"]
    idx0names = ["mean0", "sd0", "zeroMass0"]
    stepMean0 = pd.Series([0.1, 5, 23], index=idx1names)
    stepSD0 = pd.Series([0.5, 5, 1], index=idx1names)
    zeroMass0 = pd.Series([0.9, 0.15, 0], index=idx1names)
    stepPar0 = pd.concat([stepMean0, stepSD0, zeroMass0], keys=idx0names,
                         names=["parameter", "state"])

    idx0names = ["mean0", "conc0"]
    angleMean0 = pd.Series([0, 0, 0], index=idx1names)
    angleConc0 = pd.Series([0.1, 0.1, 2], index=idx1names)
    anglePar0 = pd.concat([angleMean0, angleConc0], keys=idx0names,
                          names=["parameter", "state"])

    # Simulate movement
    sim_df = sim_movement(n_ids=1, n_states=3, step_pars=stepPar0,
                          angle_pars=anglePar0, is_zero_inflated=True,
                          n_covariates=1, n_per_id=1000)

    # Fit model with 3 states, no covariates
    fit0 = fit_HMM(sim_df, nbStates=3, stepPar0=stepPar0,
                   anglePar0=anglePar0)
    print(fit0)
    # Fit another model with the covariate
    fmla = robjs.Formula("~cov1")
    fit1 = fit_HMM(sim_df, nbStates=3, formula=fmla, stepPar0=stepPar0,
                   anglePar0=anglePar0)
    print(fit1)

    fig, ax = plot_fit_hist(fit1, state_labels=idx1names, figsize=(9, 8))

    plot_states(fit1, 1, state_labels=idx1names, is_datetime_index=False)

    # Plot
    states = viterbi(fit1)
    col_pal = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
               "#0072B2", "#D55E00", "#CC79A7"]
    colors = [col_pal[i] for i in np.unique(states) - 1]
    color_map = dict(zip(colors, idx1names))

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(sim_df["x"].min(), sim_df["x"].max())
    ax.set_ylim(sim_df["y"].min(), sim_df["y"].max())
    line, = ax.plot([], [], color="y", linewidth=1)
    markers = ax.scatter([], [])
    # Custom legend from scratch only once
    leg_elements = [Line2D([0], [0], color=colors[0], marker="o",
                           linestyle="None", label=color_map[colors[0]]),
                    Line2D([0], [0], color=colors[1], marker="o",
                           linestyle="None", label=color_map[colors[1]]),
                    Line2D([0], [0], color=colors[2], marker="o",
                           linestyle="None", label=color_map[colors[2]])]
    states_leg = ax.legend(handles=leg_elements, loc=8, frameon=False,
                           bbox_to_anchor=(0.5, -0.18), title="states",
                           ncol=3, numpoints=1)

    def update_anim(i):
        i += 1
        ti = sim_df.index[i]
        ax.set_title("step: {}".format(ti), fontdict=dict(fontsize=10))
        x, y = sim_df[["x", "y"]][:i].to_numpy().T
        # Update line
        line.set_data(x, y)
        # Update markers
        markers.set_offsets(np.column_stack((x, y)))
        ms = ((np.geomspace(1e-3, 1e4, num=len(x)) / 1e4) *
              plt.rcParams["lines.markersize"] ** 2)
        markers.set_sizes(ms)
        colrs = [colors[k] for k in states[:i] - 1]
        markers.set_color(colrs)
        return line, markers

    anim = FuncAnimation(fig, update_anim, frames=sim_df.shape[0] - 1,
                         interval=10, blit=False)

    fig.tight_layout()
    fig.show()

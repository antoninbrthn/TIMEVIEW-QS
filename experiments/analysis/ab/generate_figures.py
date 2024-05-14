"""
Created on 09/05/2024
@author: Antonin Berthon

Generates figures for Qualitative Similarity measure report.
"""
import math
import os

import dice_ml
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from tslearn.clustering import KernelKMeans
from tslearn.clustering import TimeSeriesKMeans

from experiments.analysis.analysis_utils import find_results
from experiments.benchmark import load_column_transformer
from experiments.datasets import load_dataset
from timeview.ab_utils.qs_model import TTS_QS
from timeview.ab_utils.similarity import (
    load_data,
    dcw,
    get_similarities,
)
from timeview.ab_utils.transitions import T_BINARY_TREND
from timeview.lit_module import load_model


def format_df(df_test, df_cf):
    newdf = df_cf.copy().values.tolist()
    org = df_test.copy().values.tolist()[0]
    for ix in range(df_cf.shape[0]):
        for jx in range(len(org)):
            if not isinstance(newdf[ix][jx], str):
                if math.isclose(newdf[ix][jx], org[jx], rel_tol=abs(org[jx] / 10000)):
                    newdf[ix][jx] = "-"
                else:
                    newdf[ix][jx] = str(newdf[ix][jx])
            else:
                if newdf[ix][jx] == org[jx]:
                    newdf[ix][jx] = "-"
                else:
                    newdf[ix][jx] = str(newdf[ix][jx])
    return pd.DataFrame(newdf, columns=df_cf.columns, index=df_cf.index)


def format_fig(axis_fs=20, tick_fs=18, legend_fs=18, figsize=None):
    fig = plt.gcf()
    for ax in fig.get_axes():
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.xaxis.label.set_size(axis_fs)
        ax.yaxis.label.set_size(axis_fs)
        ax.title.set_size(axis_fs)
    plt.legend(fontsize=legend_fs)
    if figsize is not None:
        fig.set_size_inches(figsize)
    return fig


def plot_composition(t, y, s, tr):
    plt.plot(t, y)
    ylim = plt.gca().get_ylim()
    # annotate s in between trs
    # place in middle
    for i in range(len(s)):
        plt.text(
            (tr[i] + tr[i + 1]) / 2,
            ylim[0] + 0.5 * (ylim[1] - ylim[0]),
            s[i],
            horizontalalignment="center",
            fontsize=15,
        )
    [plt.axvline(tr_i, color="red", linestyle="--") for tr_i in tr]


def plot_clusters(ts, ys, clus, title="Clusters"):
    n_c = len(np.unique(clus))
    fig, ax = plt.subplots(1, n_c, figsize=(4 * n_c, 5))
    for c in range(n_c):
        ax[c].plot(ts[clus == c].T, ys[clus == c].T, lw=0.2, c="C0")
        ax[c].set_title(f"Cluster {c}")
    plt.suptitle(title)
    plt.tight_layout()


def plot_all_clusters(export_dir):
    model_name = "TTS"
    # dataset_name = "beta_900_20"
    dataset_name = "beta-sin_3000_20"
    # dataset_name = "synthetic_tumor_wilkerson_1"
    max_n = 1000
    load_data_bool = True

    if load_data_bool:
        results = find_results(dataset_name, model_name, results_dir="benchmarks")
        timestamp = results[-1]
        litmodel, dataset, column_transformer, compositions, preds = load_data(
            dataset_name,
            model_name,
            timestamp,
            seed=1826701614,
        )
        X, ts, ys = dataset.get_X_ts_ys()
        ts = np.array(ts)
        ys = np.array(ys)
        feature_names = dataset.get_feature_names()
        if max_n:
            sub_inds = np.random.choice(range(X.shape[0]), max_n, replace=False)
            X = X.iloc[sub_inds]
            ts = ts[sub_inds]
            ys = ys[sub_inds]
            compositions = [compositions[i] for i in sub_inds]
            preds = preds[sub_inds]

    ###### Standard Kmeans
    for n_clus in [4, 6, 8]:
        for metric in ["euclidean", "dtw"]:
            print(f"Running Kmeans with {n_clus} clusters and metric {metric}")
            km = TimeSeriesKMeans(
                n_clusters=n_clus, metric=metric, max_iter=5, random_state=0
            ).fit(preds)
            clus = km.fit_predict(preds)

            plot_clusters(
                ts, ys, clus, title=f"TimeSeriesKMeans dist {metric}- {n_clus} clusters"
            )
            fn = f"{dataset_name}_tskmeans_{metric}_{n_clus}.png"
            plt.savefig(os.path.join(export_dir, fn))
            plt.close()

    ###### Kernel Kmeans
    for n_clus in [4, 6, 8]:
        for kernel in ["gak", "cosine"]:
            print(f"Running KernelKmeans with {n_clus} clusters and kernel {kernel}")
            gak_km = KernelKMeans(n_clusters=n_clus, kernel="gak", random_state=0).fit(
                preds
            )
            clus = gak_km.fit_predict(preds)

            plot_clusters(
                ts, ys, clus, title=f"KernelKMeans kernel {kernel} - {n_clus} clusters"
            )
            fn = f"{dataset_name}_kernelkmeans_{kernel}_{n_clus}.png"
            plt.savefig(os.path.join(export_dir, fn))
            plt.close()

    ##### Hierarchical clustering
    # Example data and custom distance function
    for metric in ["euclidean", "cosine"]:
        print(f"Running hierarchical clustering with metric {metric}")
        distance_matrix = pdist(X, metric=metric)
        linkage_matrix = linkage(distance_matrix, method="average")
        for n_clus in [4, 6, 8]:
            clus = (
                fcluster(linkage_matrix, n_clus, criterion="maxclust") - 1
            )  # to start at 0
            plot_clusters(
                ts,
                ys,
                clus,
                title=f"Hierarchical clustering metric {metric} - {n_clus} clusters",
            )
            fn = f"{dataset_name}_hierarchical_{metric}_{n_clus}.png"
            plt.savefig(os.path.join(export_dir, fn))
            plt.close()

    ####### hierarchical clustering with qs metric
    transition = T_BINARY_TREND
    distance_matrix = []
    for i in range(X.shape[0]):
        similarities = get_similarities(
            select_i=0,
            compositions=compositions[i:],
            preds=preds[i:],
            sim_func="simple_distance_motifs_transition",
            kwargs={"transition": transition},
        )
        similarities = similarities[1:]  # shift to only compute the upper triangle
        distance_matrix = np.concatenate([distance_matrix, similarities])

    linkage_matrix = linkage(
        distance_matrix, method="average"
    )  # Can use 'single', 'complete', 'average', 'ward'

    # plot linkage
    plt.figure(figsize=(25, 10))
    dendrogram(linkage_matrix, labels=range(X.shape[0]))
    plt.title("Hierarchical clustering QS-transition - Dendrogram")
    plt.show()

    # cut n_clus clusters
    for n_clus in [4, 6, 8]:
        clus = (
            fcluster(linkage_matrix, n_clus, criterion="maxclust") - 1
        )  # to start at 0

        plot_clusters(
            ts,
            preds,
            clus,
            title=f"Hierarchical clustering QS-transition - {n_clus} clusters",
        )
        fn = f"{dataset_name}_hierarchical_qs_binary_transition_{n_clus}.png"
        plt.savefig(os.path.join(export_dir, fn))
        plt.close()

    ####### hierarchical clustering with qs metric
    transition = T_BINARY_TREND
    distance_matrix = []
    for i in range(X.shape[0]):
        similarities = get_similarities(
            select_i=0,
            # 0 relative to the index of composition which is shifted to only compute the upper triangle
            compositions=compositions[i:],
            preds=preds[i:],
            sim_func="dcw_lamb0",
            kwargs={"transition": transition},
        )
        similarities = similarities[1:]  # shift to only compute the upper triangle
        distance_matrix = np.concatenate([distance_matrix, similarities])

    linkage_matrix = linkage(
        distance_matrix, method="average"
    )  # Can use 'single', 'complete', 'average', 'ward'

    # plot linkage
    plt.figure(figsize=(25, 10))
    dendrogram(linkage_matrix, labels=range(X.shape[0]))
    plt.title("Hierarchical clustering QS-transition - Dendrogram")
    plt.show()

    # cut n_clus clusters
    for n_clus in [4, 6, 8]:
        clus = (
            fcluster(linkage_matrix, n_clus, criterion="maxclust") - 1
        )  # to start at 0

        plot_clusters(
            ts,
            preds,
            clus,
            title=f"Hierarchical clustering QS-transition - {n_clus} clusters",
        )
        fn = f"{dataset_name}_hierarchical_qs_dcw_l0_{n_clus}.png"
        plt.savefig(os.path.join(export_dir, fn))
        plt.close()


def compare_clusters(export_dir):
    model_name = "TTS"
    dataset_name = "beta-sin_3000_20"
    max_n = 1000  # 1000 instances only
    np.random.seed(123)
    load_data_bool = True

    if load_data_bool:
        results = find_results(dataset_name, model_name, results_dir="benchmarks")
        timestamp = results[-1]
        litmodel, dataset, column_transformer, compositions, preds = load_data(
            dataset_name,
            model_name,
            timestamp,
        )
        X, ts, ys = dataset.get_X_ts_ys()
        ts = np.array(ts)
        ys = np.array(ys)
        feature_names = dataset.get_feature_names()
        if max_n:
            sub_inds = np.random.choice(range(X.shape[0]), max_n, replace=False)
            X = X.iloc[sub_inds]
            ts = ts[sub_inds]
            ys = ys[sub_inds]
            compositions = [compositions[i] for i in sub_inds]
            preds = preds[sub_inds]

    n_clus = 4

    # HC - Euclidean distance
    metric = "euclidean"
    print(f"Running hierarchical clustering with metric {metric}")
    distance_matrix = pdist(X, metric=metric)
    linkage_matrix = linkage(distance_matrix, method="average")
    clus1 = fcluster(linkage_matrix, n_clus, criterion="maxclust") - 1  # to start at 0

    # Kmeans - DTW
    metric = "dtw"
    print(f"Running Kmeans with {n_clus} clusters and metric {metric}")
    km = TimeSeriesKMeans(
        n_clusters=n_clus, metric=metric, max_iter=5, random_state=0
    ).fit(preds)
    clus2 = km.fit_predict(preds)

    # HC with DCW lambda=0
    transition = T_BINARY_TREND
    # compute distance manually
    distance_matrix = []
    for i in range(X.shape[0]):
        similarities = get_similarities(
            select_i=0,
            compositions=compositions[i:],
            preds=preds[i:],
            sim_func="dcw_lamb0",
            kwargs={"transition": transition},
        )
        similarities = similarities[1:]
        distance_matrix = np.concatenate([distance_matrix, similarities])
    linkage_matrix = linkage(distance_matrix, method="average")
    clus3 = fcluster(linkage_matrix, n_clus, criterion="maxclust") - 1  # to start at 0

    # plot clusters
    fig, axes = plt.subplots(3, n_clus, figsize=(14, 8), sharey=True, sharex=True)
    names = ["HC\nEuclidean", "Kmeans\nDTW", "HC\nDCW l=0"]
    for i, clus in enumerate([clus1, clus2, clus3]):
        for c in range(n_clus):
            ax = axes[i, c]
            ax.plot(ts[clus == c].T, ys[clus == c].T, lw=0.2, c="C0")
            if i == 0:
                ax.set_title(f"Cluster {c}")
            if c == 0:
                ax.set_ylabel(names[i])
    for ax in axes[-1, :]:
        ax.set_xlabel("t")
    axes[-1, 0].set_ylabel(names[-1], fontweight="bold")
    axes[0, 0].set_ylim(-2.5, 2.5)
    format_fig()
    plt.tight_layout()
    plt.savefig(os.path.join(export_dir, f"{dataset_name}_clusters_comparison.png"))
    plt.close()
    # plt.show()


def cf_on_tumor(export_dir):
    dataset_name = "synthetic_tumor_wilkerson_1"
    model_name = "TTS"

    results = find_results(dataset_name, model_name, results_dir="benchmarks")
    timestamp = results[-1]
    litmodel = load_model(timestamp, seed=None, benchmarks_folder="benchmarks")
    dataset = load_dataset(
        dataset_name, dataset_description_path="dataset_descriptions"
    )
    column_transformer = load_column_transformer(timestamp, benchmarks_dir="benchmarks")
    X = dataset.X.copy()
    ts = dataset.ts
    t = ts[0]
    X[
        "outcome"
    ] = -1  # no need to this to be a valid value since we use the model's prediction
    x_test = X.drop("outcome", axis=1)
    features_names = dataset.get_feature_names()

    d = dice_ml.Data(
        dataframe=X,
        continuous_features=features_names,
        outcome_name="outcome",
    )

    target_composition = ([5], [0, 1])
    model_str = "dcw_l0"
    lamb = 0.0

    sim_func = dcw
    transition = T_BINARY_TREND
    x_i = 100
    n_cf = 5
    # transition = T_ID
    tts_qs = TTS_QS(
        litmodel.model,
        target_composition,
        similarity=sim_func,
        column_transformer=column_transformer,
        transition=transition,
        lamb=lamb,
    )

    backend = "PYT"  # needs pytorch installed
    m = dice_ml.Model(model=tts_qs, backend=backend, model_type="regressor")
    # DiCE explanation instance
    exp = dice_ml.Dice(d, m, method="random")

    # generate counterfactuals
    query_instance = x_test[x_i : x_i + 1]
    dice_exp = exp.generate_counterfactuals(
        query_instance,
        total_CFs=n_cf,
        desired_range=[0.0, 0.1],
    )
    # highlight only the changes
    dice_exp.visualize_as_dataframe(show_only_changes=True)
    # get cf as dataframe
    cfs_df = dice_exp.cf_examples_list[0].final_cfs_df
    # get initial points
    initial_points = dice_exp.cf_examples_list[0].test_instance_df

    coeffs = litmodel.model.predict_latent_variables(
        column_transformer.transform(initial_points)
    )
    init_template, init_tr = tts_qs.bspline.get_template_from_coeffs(coeffs[0, :])
    init_pred = litmodel.model.forecast_trajectory(
        column_transformer.transform(initial_points)[0, :], t
    )

    cfs_compos = []
    cfs_preds = []
    for cf_i in range(len(cfs_df)):
        cfs = cfs_df.iloc[[cf_i]]
        coeffs = litmodel.model.predict_latent_variables(
            column_transformer.transform(cfs)
        )
        cf_template, cf_tr = tts_qs.bspline.get_template_from_coeffs(coeffs[0, :])
        cfs_pred = litmodel.model.forecast_trajectory(
            column_transformer.transform(cfs)[0, :], t
        )
        cfs_compos.append((cf_template, cf_tr))
        cfs_preds.append(cfs_pred)

    os.makedirs(export_dir, exist_ok=True)
    fn = f"{dataset_name}_x{x_i}_{model_str}_target{'-'.join([str(s) for s in target_composition[0]])}_n{n_cf}"

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 2, 1)
    plt.title(f"Initial instance")
    plt.plot(t, init_pred)
    x_annot = "\n".join(
        [f"{k}: {v}" for k, v in initial_points.round(2).iloc[0].items()]
    )  # x vals
    x_annot = f"Composition:\n{init_template},\n{[round(t,2) for t in init_tr]}"
    plt.annotate(
        x_annot,
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=12,
    )
    plt.subplot(1, 2, 2)
    plt.title(f"Counterfactual explanations")
    plt.annotate(
        f"Objective: {target_composition}",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=12,
    )
    for i, ((cf_template, cf_tr), cf_preds) in enumerate(zip(cfs_compos, cfs_preds)):
        plt.plot(t, cf_preds, label="CF " + str(i + 1))
    axes = plt.gcf().get_axes()
    axes[0].set_ylabel("Value")
    for ax in axes:
        ax.set_xlabel("Time")
    format_fig(legend_fs=12)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(export_dir, fn + ".png"), dpi=250)
    plt.close()
    # plt.show()

    # export csv
    dt_init = dice_exp.cf_examples_list[0].test_instance_df.round(2)
    dt_init.columns = [c.replace("_", "") for c in dt_init.columns]
    dt_init.to_csv(
        path_or_buf=os.path.join(export_dir, fn + "_initial.csv"),
        index=True,
        # 2 decimal
    )
    dt_cfs = format_df(
        dice_exp.cf_examples_list[0].test_instance_df.round(2),
        dice_exp.cf_examples_list[0].final_cfs_df.round(2),
    )
    dt_cfs.columns = [c.replace("_", "") for c in dt_cfs.columns]
    dt_cfs.to_csv(path_or_buf=os.path.join(export_dir, fn + "_cfs.csv"), index=True)


def plot_lime_exp(lime_exp, label=1):
    exp = lime_exp.as_list(label=label)
    vals = [x[1] for x in exp]
    names = [x[0] for x in exp]
    vals.reverse()
    names.reverse()
    colors = ["green" if x > 0 else "red" for x in vals]
    pos = np.arange(len(exp)) + 0.5
    plt.barh(pos, vals, align="center", color=colors)
    plt.yticks(pos, names)


def local_explanations(export_dir):
    # Load data
    dataset_name = "synthetic_tumor_wilkerson_1"
    model_name = "TTS"
    results = find_results(dataset_name, model_name, results_dir="benchmarks")
    timestamp = results[-1]
    litmodel = load_model(timestamp, seed=None, benchmarks_folder="benchmarks")
    dataset = load_dataset(
        dataset_name, dataset_description_path="dataset_descriptions"
    )
    column_transformer = load_column_transformer(timestamp, benchmarks_dir="benchmarks")
    X = dataset.X.copy()
    ts = dataset.ts
    t = ts[0]
    X[
        "outcome"
    ] = -1  # no need to this to be a valid value since we use the model's prediction
    x_test = X.drop("outcome", axis=1)
    features_names = dataset.get_feature_names()

    # Instance to generate local explanation for
    x_i = 100
    query_instance = x_test[x_i : x_i + 1]
    init_pred = litmodel.model.forecast_trajectory(
        column_transformer.transform(query_instance)[0, :], t
    )
    n_cf = 20

    # Set model
    model_str = "dcw_l0"
    lamb = 0.0
    sim_func = dcw
    transition = T_BINARY_TREND
    target_compos = [([5], [0, 1]), ([3], [0, 1])]
    dice_imps = []
    lime_imps = []
    for target_composition in target_compos:
        # TTS model equipped with a QS measure to a given template composition
        tts_qs = TTS_QS(
            litmodel.model,
            target_composition,
            similarity=sim_func,
            column_transformer=column_transformer,
            transition=transition,
            lamb=lamb,
        )

        # DiCE
        d = dice_ml.Data(
            dataframe=X,
            continuous_features=features_names,
            outcome_name="outcome",
        )
        m = dice_ml.Model(model=tts_qs, backend="PYT", model_type="regressor")
        exp = dice_ml.Dice(d, m, method="random")
        dice_exp = exp.generate_counterfactuals(
            query_instance,
            total_CFs=n_cf,
            desired_range=[0.0, 0.1],  # lower means closer to template
        )
        dice_imp = exp.local_feature_importance(
            query_instance,
            total_CFs=n_cf,
            cf_examples_list=dice_exp.cf_examples_list,
            desired_range=[0.0, 0.1],
        )
        dice_imps.append(dice_imp.local_importance[0])

        # LIME
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            x_test.values,
            feature_names=features_names,
            class_names=["outcome"],
            verbose=True,
            mode="regression",
        )
        lime_exp = lime_explainer.explain_instance(
            query_instance.values[0],
            lambda x: 1
            - tts_qs(torch.tensor(x)).numpy(),  # higher output means closer to template
            num_features=len(features_names),
        )

        lime_imps.append(lime_exp)

    for imp in dice_imps:  # rename for plotting
        imp["init"] = imp.pop("initial_tumor_volume")

    order = ["age", "weight", "init", "dosage"]
    dts_local = [pd.DataFrame([imp])[order].melt() for imp in dice_imps]

    fig, axes = plt.subplot_mosaic(
        """
        AABC
        AADE
        """,
    )
    plt.sca(axes["A"])
    plt.title("Initial instance")
    plt.plot(t, init_pred, lw=2)
    plt.ylabel("Value")
    plt.xlabel("Time")
    plt.sca(axes["B"])
    plt.title("Template")
    # plot a basic convex function
    plt.plot(t, -(t**2), lw=3, c="C1")
    plt.annotate(
        f"Composition:\n{target_compos[0]}",
        xy=(0.05, 0.05),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=15,
    )
    plt.yticks([])
    plt.xticks([])
    plt.sca(axes["C"])
    plt.title("DiCE")
    plt.barh(
        y=dts_local[0]["variable"],
        width=dts_local[0]["value"],
    )
    plt.sca(axes["D"])
    # plot a basic convex function
    plt.plot(t, (t**2), lw=3, c="C2")
    plt.annotate(
        f"Composition:\n{target_compos[1]}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=15,
    )
    plt.xlabel("Time")
    plt.yticks([])
    plt.sca(axes["E"])
    plt.barh(
        y=dts_local[1]["variable"],
        width=dts_local[1]["value"],
    )
    plt.xlabel("Score")
    fig.set_size_inches(12, 4)
    format_fig()
    plt.tight_layout()
    fn = f"{dataset_name}_DICE_local_imp_{model_str}_target{'-'.join([str(s) for s in target_composition[0]])}_n{n_cf}"
    plt.savefig(os.path.join(export_dir, fn + ".png"), dpi=250)
    plt.show()

    fig, axes = plt.subplots(2, 1, figsize=(6, 4))
    plt.sca(axes[0])
    plt.title("LIME")
    plot_lime_exp(lime_imps[0])
    plt.sca(axes[1])
    plot_lime_exp(lime_imps[1])
    plt.xlabel("Score")
    format_fig(axis_fs=18)
    plt.tight_layout()
    fn = f"{dataset_name}_LIME_local_imp_{model_str}_target{'-'.join([str(s) for s in target_composition[0]])}_n{n_cf}"
    plt.savefig(os.path.join(export_dir, fn + ".png"), dpi=250)
    plt.show()


def global_explanations(export_dir):
    # Load data
    dataset_name = "synthetic_tumor_wilkerson_1"
    model_name = "TTS"
    results = find_results(dataset_name, model_name, results_dir="benchmarks")
    timestamp = results[-1]
    litmodel = load_model(timestamp, seed=None, benchmarks_folder="benchmarks")
    dataset = load_dataset(
        dataset_name, dataset_description_path="dataset_descriptions"
    )
    column_transformer = load_column_transformer(timestamp, benchmarks_dir="benchmarks")
    X = dataset.X.copy()
    ts = dataset.ts
    t = ts[0]
    X[
        "outcome"
    ] = -1  # no need to this to be a valid value since we use the model's prediction
    x_test = X.drop("outcome", axis=1)
    features_names = dataset.get_feature_names()

    lamb = 0.0
    sim_func = dcw
    transition = T_BINARY_TREND
    target_composition = ([5], [0, 1])  # decreasing trend
    model_str = "dcw_l0"
    # TTS model equipped with a QS measure to a given template composition
    tts_qs = TTS_QS(
        litmodel.model,
        target_composition,
        similarity=sim_func,
        column_transformer=column_transformer,
        transition=transition,
        lamb=lamb,
    )

    d = dice_ml.Data(
        dataframe=X,
        continuous_features=features_names,
        outcome_name="outcome",
    )
    m = dice_ml.Model(model=tts_qs, backend="PYT", model_type="regressor")
    exp = dice_ml.Dice(d, m, method="random")

    # bootstrap
    n_groups = 10
    n_cfs_global = 20
    ni_per_groups = 20
    res = []
    for groups_i in range(n_groups):
        inds = np.random.choice(range(x_test.shape[0]), ni_per_groups, replace=False)
        query_instances = x_test.iloc[inds]
        cobj = exp.global_feature_importance(
            query_instances,
            total_CFs=n_cfs_global,
            posthoc_sparsity_param=None,
            desired_range=[0.0, 0.1],
        )
        res.append(cobj.summary_importance)
    dt_global = pd.DataFrame(res)

    # box plot of feature importance
    sns.boxplot(data=dt_global.melt(), x="variable", y="value")
    plt.title(f"{dataset_name} - Global feature importance with DiCE")
    annot_str = f"QS:{model_str}\nTarget:{target_composition}\nBoostrap:{n_groups} x {ni_per_groups} instances\nn_cfs:{n_cfs_global}"
    plt.annotate(
        annot_str,
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=15,
    )
    plt.xlabel("Feature")
    plt.ylabel("Global feature attribution score")
    format_fig(figsize=(12, 6))
    plt.tight_layout()
    fn = f"{dataset_name}_DICE_global_imp_{model_str}_target{'-'.join([str(s) for s in target_composition[0]])}"
    plt.savefig(os.path.join(export_dir, fn + ".png"), dpi=250)
    plt.close()
    # plt.show()

    # SHAP
    explainer = shap.Explainer(tts_qs, x_test)
    shap_values = explainer(x_test)

    fn = f"{dataset_name}_SHAP_global_imp_{model_str}_target{'-'.join([str(s) for s in target_composition[0]])}"
    out_file = os.path.join(export_dir, fn)
    with open(out_file, "wb") as f:
        explainer.save(f)

    annot_str = f"QS:{model_str}, Target:{target_composition}"
    plt.title(f"{dataset_name} - Global feature importance with SHAP\n{annot_str}")
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(export_dir, fn + ".png"), dpi=250)
    plt.close()
    # plt.show()


if __name__ == "__main__":
    ROOT_DIR = "/Users/antonin/Dropbox/Antonin/Code/TIMEVIEW"
    os.chdir(os.path.join(ROOT_DIR, "experiments"))  # set working dir
    export_dir = os.path.join(ROOT_DIR, "figs")
    os.makedirs(export_dir, exist_ok=True)

    # Clustering
    compare_clusters(export_dir)
    # Counterfactual explanations
    cf_on_tumor(export_dir)
    # Local feature attribution
    local_explanations(export_dir)
    # Global feature attribution
    global_explanations(export_dir)

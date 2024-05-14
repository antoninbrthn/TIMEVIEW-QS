"""
Created on 08/05/2024
@author: Antonin Berthon

Utilities to compute similarities between compositions.
"""
import numpy as np
from fastdtw import fastdtw

from experiments.benchmark import load_column_transformer
from experiments.datasets import load_dataset
from timeview.basis import BSplineBasis
from timeview.lit_module import load_model
from timeview.ab_utils.transitions import T_ID, T_BINARY_TREND
from timeview.ab_utils.dcw_utils import (
    prepare_compositions,
    compute_dcw,
    interval_euclidean_distance,
    min_interval_size,
)


# --- Similarity measures ---
def simple_distance(**kwargs):
    """
    Compute the sum of the simple motif and time distances between two compositions.

    Arguments: s1, s2, t1, t2
    """
    s1, s2 = kwargs.get("s1", []), kwargs.get("s2", [])
    t1, t2 = kwargs.get("t1", []), kwargs.get("t2", [])
    return simple_distance_motifs(s1=s1, s2=s2) + simple_distance_times(t1=t1, t2=t2)


def simple_distance_motifs(**kwargs):
    """
    Compute the distance between two compositions by directly comparing the motifs in a 1-1 fashion.
    If one composition is longer than the other, it is cropped to the length of the shortest one.

    Arguments: s1, s2
    """
    s1, s2 = kwargs.get("s1", []), kwargs.get("s2", [])
    m = min(len(s1), len(s2))
    s1, s2 = np.array(s1[:m]), np.array(s2[:m])
    return 1 - np.sum(s1 == s2) / m


def simple_distance_times(**kwargs):
    """
    Compute the distance between two compositions by directly comparing the transition times in a 1-1 fashion.
    If one composition is longer than the other, it is cropped to the length of the shortest one.

    Arguments: t1, t2
    """
    t1, t2 = kwargs.get("t1", []), kwargs.get("t2", [])
    m = min(len(t1), len(t2))
    t1, t2 = np.array(t1[:m]), np.array(t2[:m])
    return np.linalg.norm(t1 - t2) / m


def simple_distance_motifs_transition(**kwargs):
    """
    Compute the distance between two compositions by directly comparing the motifs according to a transition matrix.

    Arguments: s1, s2, transition_matrix
    """
    s1, s2 = kwargs.get("s1", []), kwargs.get("s2", [])
    transition = kwargs.get("transition_matrix", T_ID)
    m = min(len(s1), len(s2))
    s1, s2 = s1[:m], s2[:m]
    return sum(transition[a][b] for a, b in zip(s1, s2)) / m


def transition_distance(transition=T_ID):
    return lambda a, b: transition[a][b]


def simple_distance_motifs_transition_bt(**kwargs):
    transition = T_BINARY_TREND
    return simple_distance_motifs_transition(transition_matrix=transition, **kwargs)


def local_extrema_distance(**kwargs):
    """
    Compute the distance between two compositions by comparing the times of their local extrema.

    Arguments: s1, s2, t1, t2
    """
    s1, s2 = kwargs.get("s1", []), kwargs.get("s2", [])
    t1, t2 = kwargs.get("t1", []), kwargs.get("t2", [])
    extrema = kwargs.get("kind", "max")  # default to local maxima
    trend1 = "increasing" if extrema == "max" else "decreasing"
    trend2 = "decreasing" if extrema == "max" else "increasing"

    trend1_indices = [
        i for i, state in enumerate(BSplineBasis.STATES) if trend1 in state
    ]
    trend2_indices = [
        i for i, state in enumerate(BSplineBasis.STATES) if trend2 in state
    ]

    def get_extrema_inds(s):
        ext_indices = []
        if s[0] in trend2_indices:
            ext_indices.append(0)
        for i in range(1, len(s)):
            if s[i - 1] in trend1_indices and s[i] in trend2_indices:
                ext_indices.append(i)
        if s[-1] in trend1_indices:
            ext_indices.append(len(s))
        return ext_indices

    ext1 = get_extrema_inds(s1)
    ext2 = get_extrema_inds(s2)
    return simple_distance_times(t1=np.array(t1)[ext1], t2=np.array(t2)[ext2])


def local_max_distance(**kwargs):
    return local_extrema_distance(kind="max", **kwargs)


def local_min_distance(**kwargs):
    return local_extrema_distance(kind="min", **kwargs)


def dcw(**kwargs):
    """Dynamic Composition Warping:
    - distance_func: binary trend transition matrix between motifs
    - weight_func: min of the interval sizes
    - penalty_func: Euclidean distance between intervals (penalise big shifts)
    - lamb: weight of the time shift penalty. Defaults to 0 (only motifs are considered).

    Arguments: s1, s2, t1, t2, lamb, transition
    """
    s1, s2 = kwargs.get("s1", []), kwargs.get("s2", [])
    t1, t2 = kwargs.get("t1", []), kwargs.get("t2", [])
    lamb = kwargs.get("lamb", 0)
    transition = kwargs.get("transition", T_BINARY_TREND)
    s1_extended, s2_extended, common_t = prepare_compositions(s1, t1, s2, t2)

    weight_func = min_interval_size()
    penalty_func = interval_euclidean_distance()
    distance_func = lambda a, b: transition_distance(transition=transition)(a, b)

    return compute_dcw(
        s1_extended,
        common_t,
        s2_extended,
        common_t,
        distance_func=distance_func,
        weight_func=weight_func,
        penalty_func=penalty_func,
        lamb=lamb,
    )


def dcw_lamb0(**kwargs):
    return dcw(lamb=0, **kwargs)


def dcw_lamb1(**kwargs):
    return dcw(lamb=1, **kwargs)


def dcw_lamb0p5(**kwargs):
    return dcw(lamb=0.5, **kwargs)


# --- Benchmarks ---
def mape(y_hat, y_hat_2):
    return np.mean(np.abs((y_hat - y_hat_2) / y_hat))


def l_2(y_hat, y_hat_2):
    return np.linalg.norm(y_hat - y_hat_2)


def dtw(y_hat, y_hat_2):
    distance, _ = fastdtw(y_hat, y_hat_2)
    return distance


# --- Utils ---
def get_similarities(
    select_i, compositions, preds, sim_func="simple_distance", **kwargs
):
    """
    Compute the similarities between a selected instance and all the others.
    Depending on the similarity function, the similarities can be computed on the compositions or the predictions.
    Args:
        select_i: instance index
        compositions: list of all the compositions
        preds: list of all the predictions
        sim_func: similarity function to use
    Returns: np.array of similarities across the dataset
    """
    if sim_func in [
        "simple_distance",
        "simple_distance_motifs",
        "simple_distance_times",
        "local_max_distance",
        "local_min_distance",
        "simple_distance_motifs_transition",
        "simple_distance_motifs_transition_bt",
        "dcw_lamb0",
        "dcw_lamb1",
        "dcw_lamb0p5",
    ]:
        return get_similarities_tts(select_i, compositions, sim_func=sim_func, **kwargs)
    elif sim_func in ["mape", "l_2", "dtw"]:
        return get_similarities_benchmarks(select_i, preds, sim_func=sim_func)
    else:
        raise ValueError(f"Unknown similarity function {sim_func}")


def get_similarities_tts(select_i, compositions, sim_func="simple_distance", **kwargs):
    s, t = compositions[select_i].values()
    similarities = []
    for i in range(len(compositions)):
        s2, t2 = compositions[i].values()
        similarities.append(eval(sim_func)(s1=s, t1=t, s2=s2, t2=t2, **kwargs))
    return np.array(similarities)


def get_similarities_benchmarks(select_i, preds, sim_func="l_1"):
    y_hat = preds[select_i]
    similarities = []
    for i in range(len(preds)):
        y_hat_2 = preds[i]
        similarities.append(eval(sim_func)(y_hat, y_hat_2))
    return np.array(similarities)


def get_compositions(model, X_transformed):
    coeffs = model.predict_latent_variables(X_transformed)
    bspline = BSplineBasis(
        model.config.n_basis,
        (0, model.config.T),
        internal_knots=model.config.internal_knots,
    )
    compositions = []
    for c in coeffs:
        template, transition_points = bspline.get_template_from_coeffs(c)
        compositions.append(
            {"template": template, "transition_points": transition_points}
        )
    return compositions


def get_predictions(model, X_transformed, ts):
    # Running into broadcasting issues in TTS.forecast_trajectories() so gathering predictions one by one
    preds = np.array(
        [
            model.forecast_trajectory(X_transformed[i, :], ts[i])
            for i in range(X_transformed.shape[0])
        ]
    )
    return preds


def load_data(dataset_name, model_name, timestamp, path_prefix="", seed=None):
    print(f"Loading dataset {dataset_name}, model {model_name} {timestamp}", end="...")
    # load model
    litmodel = load_model(
        timestamp, seed=seed, benchmarks_folder=path_prefix + "benchmarks"
    )
    # load dataset
    dataset = load_dataset(
        dataset_name, dataset_description_path=path_prefix + "dataset_descriptions"
    )
    column_transformer = load_column_transformer(
        timestamp, benchmarks_dir=path_prefix + "benchmarks"
    )
    X_transformed = column_transformer.transform(dataset.X.values)
    compositions = get_compositions(litmodel.model, X_transformed)
    preds = get_predictions(litmodel.model, X_transformed, dataset.ts)
    print("Done")
    return litmodel, dataset, column_transformer, compositions, preds

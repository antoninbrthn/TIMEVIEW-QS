"""
Created on 08/05/2024
@author: Antonin Berthon

Interactive visualisation tool for visualising different similarity measures.
"""
import pandas as pd
import plotly.graph_objects as go

import plotly.io as pio

pio.renderers.default = "browser"
from dash.exceptions import PreventUpdate
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from experiments.analysis.analysis_utils import find_results
from timeview.ab_utils.similarity import get_similarities, load_data

import dash
from dash import dcc, html, Input, Output, State
from dash import dash_table
from dash_table.Format import Format, Scheme
import plotly.graph_objs as go
import numpy as np

import os

ROOT_DIR = "/Users/antonin/Dropbox/Antonin/Code/TIMEVIEW"
os.chdir(os.path.join(ROOT_DIR, "experiments/"))  # set working dir

# Controls
datasets = [
    "beta_900_20",
    "synthetic_tumor_wilkerson_1",
    "beta-sin_3000_20",
    "stress-strain-lot-max-0.2",
    "flchain_1000",
    "airfoil_log",
]
similarity_measures = [
    # QS
    "simple_distance",
    "simple_distance_motifs",
    "simple_distance_times",
    "local_max_distance",
    "local_min_distance",
    "simple_distance_motifs_transition_bt",
    "dcw_lamb0",
    "dcw_lamb1",
    "dcw_lamb0p5",
    # benchmarks
    "mape",
    "l_2",
    "dtw",
]
control_keys = ["dataset_name", "similarity_measure", "dim_reduction", "select_i"]
DEFAULT_CONTROLS = {
    "dataset_name": "beta_900_20",
    "similarity_measure": "simple_distance",
    "dim_reduction": "PCA",
}
model_name = "TTS"
# limit nb of points to show
max_n = 5000

# Initial selected index and top indices for plot
DEFAULT_SELECT_I = 42
TOP_N = 5


def get_name(feature_names):
    return (
        lambda x: f'i={x.name}<br>{"<br>".join([f"{f}: {x[f]}" for f in feature_names])}'
    )


# Initialize the Dash app
app = dash.Dash(__name__)

# Create initial figures for four plots
fig_scatter = go.Figure()
fig_selected = go.Figure()
fig_closest = go.Figure()
fig_hover = go.Figure()


# init scatter plot with embeddings
def init_scatter(X_embed, X, feature_names):
    fig_scatter = go.Figure()
    fig_scatter.add_trace(
        go.Scatter(
            x=X_embed[:, 0],
            y=X_embed[:, 1],
            mode="markers",
            name="Training data",
            hovertext=X.apply(get_name(feature_names), axis=1),
        )
    )
    format_figures(which="selected")
    return fig_scatter


def check_controls_changed(stored_data, previous_data):
    """Checks whether the controls have changed."""
    for key in control_keys:
        if stored_data.get(key, None) != previous_data.get(key, None):
            return True
    return False


def update_previous_values(stored_data):
    return {key: stored_data[key] for key in control_keys if key in stored_data.keys()}


def set_legend_top(fig):
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12),
        )
    )
    return fig


def format_figures(which="all"):
    global fig_scatter, fig_selected, fig_closest, fig_hover
    if which == "all" or which == "selected":
        fig_selected.update_layout(
            title="Selected Point Predicted Trajectory",
            xaxis_title="Time",
            yaxis_title="Value",
            margin=dict(l=20, r=20, t=40, b=20),
            overwrite=True,
        )
    if which == "all" or which == "closest":
        fig_closest.update_layout(
            title="Closest Points Predicted Trajectories",
            xaxis_title="Time",
            yaxis_title="Value",
            margin=dict(l=20, r=20, t=40, b=20),
            overwrite=True,
        )
    if which == "all" or which == "hover":
        fig_hover.update_layout(
            title="Hover Point Predicted Trajectory",
            xaxis_title="Time",
            yaxis_title="Value",
            margin=dict(l=20, r=20, t=40, b=20),
            overwrite=True,
        )
    if which == "all" or which == "scatter":
        fig_scatter.update_layout(
            title="Input Space Embedding",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            margin=dict(l=20, r=20, t=40, b=20),
            overwrite=True,
        )


# fig_scatter = init_scatter(X_embed, X, feature_names)  # init scatter plot
format_figures()

app.layout = html.Div(
    [
        dcc.Store(id="intermediate-data"),  # Store intermediate data
        dcc.Store(id="previous-values", data={}),  # Top bar for controls
        html.Div(
            [
                html.H3("Controls", style={"flex-grow": "1"}),
                dcc.Dropdown(
                    id="dataset-dropdown",
                    options=[{"label": i, "value": i} for i in datasets],
                    value=DEFAULT_CONTROLS["dataset_name"],  # default value
                    style={"flex-grow": "3"},
                ),
                dcc.Dropdown(
                    id="similarity-dropdown",
                    options=[{"label": i, "value": i} for i in similarity_measures],
                    value=DEFAULT_CONTROLS["similarity_measure"],  # default value
                    style={"flex-grow": "3"},
                ),
                dcc.RadioItems(
                    id="dimension-reduction-radio",
                    options=[
                        {"label": "PCA", "value": "PCA"},
                        {"label": "t-SNE", "value": "tSNE"},
                    ],
                    value=DEFAULT_CONTROLS["dim_reduction"],  # default value
                    style={"flex-grow": "3"},
                ),
            ],
            style={
                "display": "flex",
                "width": "100%",
                "padding": "20px",
                "boxSizing": "border-box",
                "alignItems": "center",
            },
        ),
        html.Div(
            [
                dcc.Graph(id="scatter-plot", figure=fig_scatter),
                dcc.Graph(id="selected-plot", figure=fig_selected),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-around",
                "padding": "2px",
            },
        ),
        html.Div(
            [
                dcc.Graph(id="closest-plot", figure=fig_closest),
                dcc.Graph(id="hover-plot", figure=fig_hover),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-around",
                "padding": "2px",
            },
        ),
        html.Div(
            [
                dash_table.DataTable(
                    id="points-table",
                    columns=[],
                    data=[],
                    style_table={"height": "300px", "overflowY": "auto"},
                    style_cell={"textAlign": "center", "padding": "10px"},
                    style_data_conditional=[
                        {
                            "if": {"column_type": "numeric"},
                            "type": "numeric",
                            "format": Format(precision=3, scheme=Scheme.fixed),
                        }
                    ],
                )
            ],
            style={
                "width": "100%",
                "height": "100vh",
                "display": "flex",
                "flexDirection": "column",
            },
        ),
    ]
)


def make_json_serializable(data):
    if isinstance(data, (np.ndarray, np.generic)):
        return data.tolist()
    elif isinstance(data, pd.DataFrame):
        return data.to_dict(orient="list")
    elif isinstance(data, dict):
        return {k: make_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_json_serializable(item) for item in data]
    else:
        return data


# Callback for updating data based on controls
@app.callback(
    Output("intermediate-data", "data"),
    Output("previous-values", "data"),
    [
        Input("dataset-dropdown", "value"),
        Input("similarity-dropdown", "value"),
        Input("dimension-reduction-radio", "value"),
    ],
    [State("previous-values", "data")],
    [State("intermediate-data", "data")],
)
def update_data(
    dataset_name, similarity_measure, dim_reduction, previous_data, stored_data
):
    # check if dataset_name has changed
    dataset_name_changed = dataset_name != previous_data.get("dataset_name", None)
    dim_reduction_changed = dim_reduction != previous_data.get("dim_reduction", None)
    previous_data.update(
        {
            "dataset_name": dataset_name,
            "similarity_measure": similarity_measure,
            "dim_reduction": dim_reduction,
        }
    )

    if dataset_name_changed:
        print("Loading dataset")
        results = find_results(dataset_name, model_name, results_dir="benchmarks")
        timestamp = results[-1]
        litmodel, dataset, column_transformer, compositions, preds = load_data(
            dataset_name, model_name, timestamp
        )
        X, ts, ys = dataset.get_X_ts_ys()
        feature_names = dataset.get_feature_names()
        ts = np.array(ts)
        ys = np.array(ys)

        # lim number of points to debug
        X = X[:max_n]
        ts = ts[:max_n]
        ys = ys[:max_n]
        preds = preds[:max_n]
        compositions = compositions[:max_n]
    else:
        X = np.array(stored_data["X"])
        ts = np.array(stored_data["ts"])
        ys = np.array(stored_data["ys"])
        feature_names = stored_data["feature_names"]
        preds = np.array(stored_data["preds"])
        compositions = stored_data["compositions"]

    if dataset_name_changed or dim_reduction_changed:
        print("Loading embeddings")
        bypass_emb = dataset_name in ["stress-strain-lot-max-0.2"]
        X_oh = pd.get_dummies(X)  # handle categorical columns
        if bypass_emb:
            X_embed = X.values
            assert X_embed.shape[1] <= 2
        # Embed points
        elif dim_reduction == "tSNE":
            tsne = TSNE(n_components=2, random_state=0)
            X_embed = tsne.fit_transform(X_oh)
        else:
            pca = PCA(n_components=2)
            X_embed = pca.fit_transform(X_oh)
    else:
        X_embed = stored_data["X_embed"]

    return (
        make_json_serializable(
            {
                "X": X,
                "X_embed": X_embed,
                "ts": ts,
                "ys": ys,
                "preds": preds,
                "compositions": compositions,
                "similarity_measure": similarity_measure,
                "feature_names": feature_names,
                "dataset_name": dataset_name,
                "dim_reduction": dim_reduction,
            },
        ),
        previous_data,
    )


# Callbacks to update the line plots based on click and hover
@app.callback(
    [
        Output("scatter-plot", "figure"),
        Output("selected-plot", "figure"),
        Output("closest-plot", "figure"),
        Output("hover-plot", "figure"),
    ],
    [
        Input("intermediate-data", "data"),
        Input("scatter-plot", "clickData"),
        Input("scatter-plot", "hoverData"),
    ],
    [State("previous-values", "data")],
)
def update_line_plots(stored_data, clickData, hoverData, previous_data):
    global fig_selected, fig_closest, fig_hover, fig_scatter

    if not stored_data:
        raise PreventUpdate  # Prevent callback execution if no data available
    else:
        # check if controls have changed
        controls_changed = check_controls_changed(stored_data, previous_data)
        previous_data.update(update_previous_values(stored_data))

    if stored_data:
        # unpack stored data
        X = pd.DataFrame(stored_data["X"])
        X_embed = np.array(stored_data["X_embed"])
        ts = np.array(stored_data["ts"])
        ys = np.array(stored_data["ys"])
        feature_names = stored_data["feature_names"]
        preds = np.array(stored_data["preds"])
        compositions = stored_data["compositions"]
        similarity_measure = stored_data["similarity_measure"]
    else:
        return fig_scatter, fig_selected, fig_closest, fig_hover

    if controls_changed or (len(fig_scatter.data) == 0):
        # Update selected plot
        fig_scatter.data = []
        fig_scatter = init_scatter(X_embed, X, feature_names)

        # Empy other plots
        fig_selected.data = []
        fig_closest.data = []
        fig_hover.data = []
        format_figures()

    select_i = clickData["points"][0]["pointIndex"] if clickData else None
    if select_i and (select_i >= len(compositions)):
        select_i = DEFAULT_SELECT_I
    select_i_changed = select_i != previous_data.get("select_i", None)
    previous_data.update({"select_i": select_i})

    if clickData and select_i_changed:
        similarities = get_similarities(
            select_i, compositions, preds, sim_func=similarity_measure
        )
        top_inds = np.argsort(similarities)[:TOP_N]

        # Update selected plot
        y_x = preds[select_i]
        fig_selected = go.Figure()
        fig_selected.add_trace(
            go.Scatter(x=ts[select_i], y=y_x, mode="lines", name="Predicted Trajectory")
        )

        # Predicted trajectories for closest points
        top_inds = [i for i in top_inds if i != select_i]
        fig_closest = go.Figure()
        fig_closest.add_trace(
            go.Scatter(
                x=ts[select_i],
                y=preds[select_i],
                mode="lines",
                name=f"Selected, i={select_i}",
                line=dict(width=3),
            )
        )
        y_top_n = preds[top_inds]
        for i in range(len(top_inds)):
            d_str = (
                f"{similarities[top_inds[i]]:.1e}"
                if similarities[top_inds[i]] < 1e-2
                else f"{similarities[top_inds[i]]:.2f}"
            )
            fig_closest.add_trace(
                go.Scatter(
                    x=ts[top_inds[i]],
                    y=y_top_n[i],
                    mode="lines",
                    name=f"i={top_inds[i]} (d={d_str})",
                )
            )

        # Update scatter plot with selected and closest points
        fig_scatter = init_scatter(X_embed, X, feature_names)
        fig_scatter.update_traces(marker=dict(size=5, color="blue"))
        fig_scatter.add_trace(
            go.Scatter(
                x=[X_embed[select_i, 0]],
                y=[X_embed[select_i, 1]],
                mode="markers",
                marker=dict(size=12, color="red"),
                name="Selected",
                hoverinfo="skip",
            )
        )
        fig_scatter.add_trace(
            go.Scatter(
                x=X_embed[top_inds, 0],
                y=X_embed[top_inds, 1],
                mode="markers",
                marker=dict(size=10, color="green"),
                name="Closest",
                hoverinfo="skip",
            )
        )
        format_figures()
        fig_closest = set_legend_top(fig_closest)
        fig_scatter = set_legend_top(fig_scatter)

    if hoverData:
        hover_i = hoverData["points"][0]["pointIndex"]
        y_hover = preds[hover_i]
        fig_hover = go.Figure()
        fig_hover.add_trace(
            go.Scatter(
                x=ts[hover_i], y=y_hover, mode="lines", name="Hovered Point Trajectory"
            )
        )

        format_figures("hover")

    return fig_scatter, fig_selected, fig_closest, fig_hover


@app.callback(
    [Output("points-table", "columns"), Output("points-table", "data")],
    [
        Input("intermediate-data", "data"),
        Input("scatter-plot", "clickData"),
    ],
)
def update_table(stored_data, clickData):
    if not stored_data:
        raise PreventUpdate  # Prevent callback execution if no data available
    else:
        # unpack stored data
        X = pd.DataFrame(stored_data["X"])
        X_embed = np.array(stored_data["X_embed"])
        feature_names = stored_data["feature_names"]
        compositions = stored_data["compositions"]
        preds = np.array(stored_data["preds"])
        similarity_measure = stored_data["similarity_measure"]

    columns = [{"name": col, "id": col} for col in feature_names]

    if clickData:  # when clicking on a scatter plot point
        select_i = clickData["points"][0]["pointIndex"]
        similarities = get_similarities(
            select_i, compositions, preds, sim_func=similarity_measure
        )

        top_inds = np.argsort(similarities)[:TOP_N]
        top_inds = np.concatenate(([select_i], [i for i in top_inds if i != select_i]))
        score_similarities = similarities[top_inds]

        data = X.iloc[top_inds]
        data = data.reset_index()
        for i, sim in enumerate(score_similarities):
            data.loc[i, "distance"] = sim
        data = data.to_dict("records")
        columns = [
            {"name": col, "id": col} for col in ["index", "distance"] + feature_names
        ]
    else:
        # No point selected
        data = []

    return columns, data


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=True, port=8050)

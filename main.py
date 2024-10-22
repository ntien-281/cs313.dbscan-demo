import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import time

st.set_page_config(page_title="DBSCAN Demo", layout="centered")

global_fig = go.Figure()

# DBSCAN Parameters
st.sidebar.title("Parameters")
dataset_choice = st.sidebar.selectbox("Select Dataset", ("Moons", 'Circles', "Blobs - 3", "Blobs - 5"))
eps = st.sidebar.slider("Epsilon (Neighborhood Radius)", 0.05, 0.3, 0.15)
min_samples = st.sidebar.slider("Minimum Samples", 1, 10, 5)
plot_size = st.sidebar.slider("Plot Size", 400, 1000, 700)
step_time = st.sidebar.slider("Step time", 0.01, 0.5, 0.05)
run_button = st.sidebar.button("Run/Rerun")

plot_placeholder = st.empty()

# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'labels' not in st.session_state:
    st.session_state.labels = None
if 'paused' not in st.session_state:
    st.session_state.paused = False

# Function to generate dataset
def create_data():
    if dataset_choice == "Moons":
        X, _ = make_moons(n_samples=200, noise=0.1)
    if dataset_choice == "Circles":
        X, _ = make_circles(n_samples=200, noise=0.1)
    elif dataset_choice == "Blobs - 3":
        X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.8)
    elif dataset_choice == "Blobs - 5":
        X, _ = make_blobs(n_samples=200, centers=5, cluster_std=0.8)
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)

# Function to compute neighbors within epsilon
def region_query(X, point_idx, eps):
    neighbors = []
    for i in range(len(X)):
        if np.linalg.norm(X[point_idx] - X[i]) <= eps:
            neighbors.append(i)
    return neighbors

# Expand cluster starting from a core point
def expand_cluster(X, labels, point_idx, neighbors, cluster_id, eps, min_samples, core_point_idx):
    labels[point_idx] = cluster_id
    i = 0
    plot_dbscan(X, labels, core_point_idx=core_point_idx)
    while i < len(neighbors):
        neighbor_idx = neighbors[i]
        if labels[neighbor_idx] == -1:  # If it's noise, reassign to current cluster and highlight
            labels[neighbor_idx] = cluster_id
            plot_dbscan(X, labels, core_point_idx=neighbor_idx)
        elif labels[neighbor_idx] == 0:  # If unclassified, assign to the cluster
            labels[neighbor_idx] = cluster_id
            new_neighbors = region_query(X, neighbor_idx, eps)
            if len(new_neighbors) >= min_samples:
                neighbors += new_neighbors  # Expand if it's a core point
            plot_dbscan(X, labels, core_point_idx=neighbor_idx)
        i += 1

# DBSCAN algorithm implementation from scratch
def dbscan(X, eps, min_samples):
    labels = np.zeros(len(X))  # 0: unclassified, -1: noise
    cluster_id = 0
    for point_idx in range(len(X)):
        if labels[point_idx] != 0:  # Skip if already classified
            continue
        neighbors = region_query(X, point_idx, eps)
        if len(neighbors) < min_samples:
            labels[point_idx] = -1  # Mark as noise
        else:
            cluster_id += 1  # Start new cluster
            expand_cluster(X, labels, point_idx, neighbors, cluster_id, eps, min_samples, core_point_idx=point_idx)
        # plot_dbscan(X, labels, core_point_idx=point_idx)
        time.sleep(step_time)
    return labels

# Plot dataset function
def plot_dataset(data):
    global global_fig
    global_fig = go.Figure()
    plot_placeholder.empty()
    df = pd.DataFrame(data, columns=["Feature 1", "Feature 2"])
    
    global_fig.add_trace(go.Scatter(
        x=df["Feature 1"],
        y=df["Feature 2"],
        mode='markers',
        marker=dict(size=10, line=dict(width=1, color='Black')),  # Hollow markers
        marker_color='white'
    ))

    x_min, x_max = data[:, 0].min() - 0.2, data[:, 0].max() + 0.2
    y_min, y_max = data[:, 1].min() - 0.2, data[:, 1].max() + 0.2

    # Hide axes and gridlines
    global_fig.update_layout(
        title="Data Preview",
        width=plot_size,
        height=plot_size,
        xaxis=dict(showline=False, showgrid=False, zeroline=False, visible=False, range=[x_min, x_max]),
        yaxis=dict(showline=False, showgrid=False, zeroline=False, visible=False, range=[y_min, y_max], scaleanchor="x", scaleratio=1),
        showlegend=False
    )
    plot_placeholder.plotly_chart(global_fig, config={'displayModeBar': False}, use_container_width=True)

# Plot DBSCAN function
def plot_dbscan(X, labels, core_point_idx=None, final=False):

    time.sleep(step_time)
    global global_fig
    if global_fig is None:
        global_fig = go.Figure()
    global_fig.data = []

    df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
    df['Labels'] = labels
    color_map = {0: 'white', -1: 'white'}  # Noise color
    unique_labels = np.unique(labels)

    for label in unique_labels:
        label_int = int(label)
        if label != -1:  # Don't include noise in color map
            color_map[label_int] = px.colors.qualitative.Plotly[label_int % len(px.colors.qualitative.Plotly)]

    # df['Color'] = df['Labels'].map(color_map)

    for label in unique_labels:
        label_int = int(label)
        color = color_map[label_int] if label_int != -1 else 'white'
        marker_color = color_map[label_int] if label_int != -1 else 'white'

        if label_int == 0: # Unvisited points have black outline, noises are all white
            color = 'black'
            marker_color = 'white'

        cluster_points = df[df['Labels'] == label]
        if len(cluster_points) > 0:
            global_fig.add_trace(go.Scatter(
                x=cluster_points["Feature 1"],
                y=cluster_points["Feature 2"],
                mode='markers',
                marker=dict(size=10, line=dict(width=1, color=color)),
                marker_color=marker_color,
                name=f'Cluster {label_int}' if label_int != -1 else 'Noise'
            ))
            # Underlay cluster area
            if label != 0 and label != -1:
                global_fig.add_trace(go.Scatter(
                    x=cluster_points["Feature 1"],
                    y=cluster_points["Feature 2"],
                    mode='markers',
                    marker=dict(size=eps * 1000, color=color, opacity=0.04, line=dict(width=0)),
                    marker_color=marker_color,
                    name=None,

                ))

    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2

    if core_point_idx is not None and eps is not None:
        circle_color = color_map[labels[core_point_idx]]

        theta = np.linspace(0, 2 * np.pi, 100)
        x_circle = X[core_point_idx, 0] + eps * np.cos(theta)
        y_circle = X[core_point_idx, 1] + eps * np.sin(theta)
        global_fig.add_trace(go.Scatter(
            x=x_circle,
            y=y_circle,
            mode='lines',
            line=dict(color=circle_color, width=2),
            showlegend=False
        ))

    # Hide axes and gridlines
    global_fig.update_layout(
        title="Result",
        width=plot_size,
        height=plot_size,
        xaxis=dict(showline=False, showgrid=False, zeroline=False, visible=False, range=[x_min, x_max]),
        yaxis=dict(showline=False, showgrid=False, zeroline=False, visible=False, range=[y_min, y_max], scaleanchor="x", scaleratio=1),
        showlegend=False
    )
    if final:
        plot_placeholder.plotly_chart(global_fig, key="final", config={'displayModeBar': False}, use_container_width=True)
    else:
        plot_placeholder.plotly_chart(global_fig, config={'displayModeBar': False}, use_container_width=True)
        
# Update the dataset in the session state when a new dataset is selected
if st.session_state.dataset != dataset_choice:
    st.session_state.dataset = dataset_choice
    st.session_state.data = create_data()  # Generate new data when dataset changes
    st.session_state.labels = np.zeros(st.session_state.data.shape[0])  # Reset labels

    # Plot the dataset
    plot_dataset(st.session_state.data)
else:
    # Keep the data same until the user clicks "Run DBSCAN"
    X = st.session_state.data
    plot_dataset(X)

# Run DBSCAN
if run_button:
    st.session_state.labels = dbscan(X, eps, min_samples)
    plot_dbscan(X, st.session_state.labels, final=True)
import os
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import json
import statistics
import sklearn
import sklearn.neighbors
import sklearn.preprocessing
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import networkx as nx
from networkx.algorithms import bipartite

import torch

import torch.optim.lr_scheduler as lr_scheduler
import torch_geometric
import torch_geometric.transforms as T
import torch_geometric.transforms
import torch_geometric.datasets
import torch_geometric.nn
from torch_geometric.utils import to_networkx

from env_variables import *

###########################################################
################ Data IO ##################################
###########################################################

def pred_df_to_csv(edges_df_bipartite, nodes_df):
    data_list = []
    
    pred_edges_df = edges_df_bipartite[edges_df_bipartite["edge_label"]==1].reset_index(drop=True)

    for idx, row in pred_edges_df.iterrows():
        nuclei_golgi_pair = {}

        source_id = row["source"]
        node = nodes_df[nodes_df["ID"]==source_id]
        node_type = reversed_nodes_type_int_encodings[node["node_type"].iloc[0]]
        nuclei_golgi_pair[node_type] = node

        target_id = row["target"]
        node = nodes_df[nodes_df["ID"]==target_id]
        node_type = reversed_nodes_type_int_encodings[node["node_type"].iloc[0]]
        nuclei_golgi_pair[node_type] = node

        nuclei = nuclei_golgi_pair["nuclei"]
        golgi = nuclei_golgi_pair["golgi"]

        xn, yn, zn = nuclei["X"].iloc[0], nuclei["Y"].iloc[0], nuclei["Z"].iloc[0]
        xg, yg, zg = golgi["X"].iloc[0], golgi["Y"].iloc[0], golgi["Z"].iloc[0]

        data_list.append([xn, yn, zn, xg, yg, zg])

    data_list = np.array(data_list)
    
    return data_list

class CustomEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for data types that are not JSON serializable.
    See here for more info: https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
    Usage: 
    
    with open("out.json", "w") as outfile:
        json.dump(content, outfile, cls = CustomEncoder)
        
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy()
        return super(CustomEncoder, self).default(obj)

#########################################
####### Data Preprocessing ##############
#########################################

def load_df_from_csv(gt_vectors_csv: str, column_names=['YN', 'XN', 'YG', 'XG', 'ZN', 'ZG'],
                     nuclei_columns=['YN', 'XN', 'ZN'],
                     golgi_columns=['YG', 'XG', 'ZG'],
                     nodes_type_int_encodings=nodes_type_int_encodings,
                    edges_type_int_encodings = edges_type_int_encodings):
    if("automatic" not in gt_vectors_csv):
        df = pd.read_csv(gt_vectors_csv, delimiter=",", header=None)
        df.columns = column_names

        # Split the DataFrame into two
        nuclei_df = df[nuclei_columns].copy().reset_index(drop=True)
        nuclei_df["node_type"] = nodes_type_int_encodings['nuclei']
        nuclei_df.rename(columns={'YN': 'Y', 'XN': 'X', 'ZN': 'Z'}, inplace=True)  # Rename nuclei columns

        golgi_df = df[golgi_columns].copy().reset_index(drop=True)
        golgi_df["node_type"] = nodes_type_int_encodings['golgi']
        golgi_df.rename(columns={'YG': 'Y', 'XG': 'X', 'ZG': 'Z'}, inplace=True)  # Rename golgi columns

        # Set the 'ID'
        golgi_df["ID"] = range(len(nuclei_df), len(nuclei_df) + len(golgi_df))
        nuclei_df["ID"] = range(len(nuclei_df))

        # Concatenate the two DataFrames into 'nodes_df'
        nodes_df = pd.concat([nuclei_df, golgi_df]).reset_index(drop=True)
        nodes_df["X"] = nodes_df["X"].apply(lambda x: x - 1)
        nodes_df["Y"] = nodes_df["Y"].apply(lambda y: y - 1)
        nodes_df["Z"] = nodes_df["Z"].apply(lambda z: z - 1)

        # Create edges between nuclei and golgi based on nodes in the same line
        edges_df = pd.DataFrame({'source': nuclei_df['ID'], 'target': golgi_df['ID']})
    else:
        df = pd.read_csv(gt_vectors_csv, delimiter=",")
        df["node_type"] = df["node_type"].apply(lambda x : nodes_type_int_encodings[x])
        nodes_df = df.copy()
        edges_df = pd.DataFrame(columns=['source', 'target', "edge_type"])#empty
        
    return df, nodes_df, edges_df

def scale_df_features(df, file_name):
    
    scale_x, scale_y, scale_z = 1, 1, 1

    scales_dict = {"Crop1.csv":{"x":0.333,"y":0.333,"z":0.270},
                  "Crop2.csv":{"x":0.333,"y":0.333,"z":0.270},
                  "Crop3.csv":{"x":0.333,"y":0.333,"z":0.270},
                  "Crop4.csv":{"x":0.333,"y":0.333,"z":0.270},
                  "Crop5_BC.csv":{"x":0.333,"y":0.333,"z":0.270},
                  "Crop6_BC.csv":{"x":0.333,"y":0.333,"z":0.270},
                  "Crop7_BC.csv":{"x":0.333,"y":0.333,"z":0.400},
                  "Crop8_BC.csv":{"x":0.333,"y":0.333,"z":0.400}}

    if(file_name in scales_dict):
        scale_x, scale_y, scale_z = scales_dict[file_name]["x"], scales_dict[file_name]["y"], scales_dict[file_name]["z"]
    
    df["X"] = df["X"]*scale_x
    df["Y"] = df["Y"]*scale_y
    df["Z"] = df["Z"]*scale_z
    return df
        
def cart2sph(x,y,z):
    """ x, y, z :  ndarray coordinates"""
    #also called phi
    azimuth = np.arctan2(y,x)
    xy2 = x**2 + y**2
    #also called theta
    elevation = np.arctan2(z, np.sqrt(xy2))
    r = np.sqrt(xy2 + z**2)
    return azimuth.item(), elevation.item(), r

def get_edge_features(edges_df, nodes_df,  
                            reversed_nodes_type_int_encodings = reversed_nodes_type_int_encodings, 
                        edges_type_int_encodings = edges_type_int_encodings):
    # Initialize lists to store calculated metrics
    x1_values, y1_values, z1_values, node_type1_values = [], [], [], []
    x2_values, y2_values, z2_values, node_type2_values = [], [], [], []
    delta_x_values = []
    delta_y_values = []
    delta_z_values = []
    weight_values = []
    angle_orientation_theta_values = []
    angle_orientation_phi_values = []

    edge_type_values = []
    
    # Calculate metrics for each edge and store in lists
    for index, row in edges_df.iterrows():
        source_id = row['source']
        target_id = row['target']

        source_node = nodes_df.loc[nodes_df['ID'] == source_id].to_dict('records')[0]
        target_node = nodes_df.loc[nodes_df['ID'] == target_id].to_dict('records')[0]

        delta_x = source_node['X'] - target_node['X']
        delta_y = source_node['Y'] - target_node['Y']
        delta_z = source_node['Z'] - target_node['Z']

        # Calculate Euclidean distance as the 'weight'
        weight = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

        # Append calculated values to respective lists
        x1_values.append(source_node['X'])
        y1_values.append(source_node['Y'])
        z1_values.append(source_node['Z'])
        node_type1_values.append(source_node['node_type'])

        x2_values.append(target_node['X'])
        y2_values.append(target_node['Y'])
        z2_values.append(target_node['Z'])
        node_type2_values.append(target_node['node_type'])

        delta_x_values.append(delta_x)
        delta_y_values.append(delta_y)
        delta_z_values.append(delta_z)
        weight_values.append(weight)
        
        phi_rads, theta_rads, r = cart2sph(delta_x, delta_y, delta_z)
        
        angle_orientation_theta_values.append(theta_rads)
        angle_orientation_phi_values.append(phi_rads)
        
        edge_type_string = reversed_nodes_type_int_encodings[source_node["node_type"]]+"-"+reversed_nodes_type_int_encodings[target_node["node_type"]] 
        edge_type_values.append(edges_type_int_encodings[edge_type_string])

    # Add the lists as new columns to edges_df
    edges_df['x1'] = x1_values
    edges_df['y1'] = y1_values
    edges_df['z1'] = z1_values
    edges_df['node_type1'] = node_type1_values

    edges_df['x2'] = x2_values
    edges_df['y2'] = y2_values
    edges_df['z2'] = z2_values
    edges_df['node_type2'] = node_type2_values

    edges_df['delta_x'] = delta_x_values
    edges_df['delta_y'] = delta_y_values
    edges_df['delta_z'] = delta_z_values
    edges_df['weight'] = weight_values
    edges_df["edge_type"] = edge_type_values
    
    
    edges_df["angle_orientation_theta"] = angle_orientation_theta_values
    edges_df["angle_orientation_phi"] = angle_orientation_phi_values
    
    
    return edges_df

def get_edges_knn(nodes_df: pd.DataFrame, edge_types: Dict = {"nuclei-nuclei": 0, "nuclei-golgi": 0,"golgi-golgi": 0, "golgi-nuclei": 0},
                                            reversed_nodes_type_int_encodings=reversed_nodes_type_int_encodings):
    """
    Builds an initial edge DataFrame based on K-nearest neighbors (KNN) algorithm using nodes from the nodes_df DataFrame.

    :param nodes_df: DataFrame containing node information, including coordinates and node types.
    :param edge_types: Dictionary specifying the maximum number of edges for each edge type.
    :return: DataFrame of edges with columns ['source', 'target'].
    """
    # Extract node information       
    coordinates = nodes_df[['Y', 'X', 'Z']].values
    node_types = nodes_df['node_type'].values
    node_ids = nodes_df["ID"].values
    
    # Initialize KNN model
    #n_neighbors = max(list(edge_types.values()))
    #n_neighbors = int(n_neighbors*1.5)
    #n_neighbors= min(n_neighbors,len(nodes_df))
    n_neighbors = len(nodes_df)
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors= n_neighbors, 
                                              algorithm='ball_tree').fit(coordinates)

    # Compute K-nearest neighbors
    distances, indexes = nbrs.kneighbors(coordinates)

    # Initialize the edge list
    edge_list = []
    
    for node_i_index in range(len(indexes)):
        
        node_i_id = node_ids[node_i_index]
        node_i_type = node_types[node_i_index]
        node_i_coordinates = coordinates[node_i_index]
        
        node_i_type_string = reversed_nodes_type_int_encodings[node_i_type]
        # Initialize variables to keep track of edge counts
        edge_counts = {"nuclei-nuclei": 0, "nuclei-golgi": 0, "golgi-golgi": 0, "golgi-nuclei": 0}
           
        node_i_indexes = indexes[node_i_index]
        for node_j_index in node_i_indexes:
            node_j_id = node_ids[node_j_index]
            node_j_type = node_types[node_j_index]
            node_j_coordinates = coordinates[node_j_index]
            
            if(node_j_id!=node_i_id):
                #raise ValueError("Node cannot be neigbour of itself!")
                
                node_j_type_string = reversed_nodes_type_int_encodings[node_j_type]

                edge_type_string = node_i_type_string+"-"+node_j_type_string
                
                if(edge_type_string in edge_types):
                    if(edge_counts[edge_type_string]<edge_types[edge_type_string]):
                        edge_info = {"source":node_i_id, "target":node_j_id}
                        edge_counts[edge_type_string]+=1
                        edge_list.append(edge_info)
                else:
                    raise ValueError("Edge type",edge_type_string,"not supported!")
    
    edges_df = pd.DataFrame(edge_list)
    
    return edges_df

def apply_edges_df_label(edges_df, edges_df_knn):
    edges_info_dict = {}
    for idx, row in edges_df.iterrows():
        src = row["source"]
        tgt = row["target"]
        if(src not in edges_info_dict):
            edges_info_dict[src] = set()
        if(tgt not in edges_info_dict):
            edges_info_dict[tgt] = set()
        edges_info_dict[src].add(tgt)
        edges_info_dict[tgt].add(src)
    
    merged_edges_df = edges_df_knn.copy()
    merged_edges_df["edge_label"] = 0
    
    for idx, row in merged_edges_df.iterrows():
        src = row["source"]
        tgt = row["target"]

        # Check if src and tgt are in edges_info_dict
        if src in edges_info_dict:
            if tgt in edges_info_dict[src]:
                merged_edges_df.at[idx, "edge_label"] = 1  # Update the edge_label in the original DataFrame  
    
    return merged_edges_df

def apply_concat_nodes_edges_df(nodes_df, edges_df):
    # Merge nodes_df with edges_df for source nodes
    merged_df = pd.merge(edges_df, nodes_df, left_on='source', right_on='ID', how='left')
    merged_df.rename(columns={'X': 'source_X', 'Y': 'source_Y', 'Z': 'source_Z', 'node_type': 'source_node_type', 'ID':'source_ID'}, inplace=True)

    # Merge nodes_df with edges_df for target nodes
    merged_df = pd.merge(merged_df, nodes_df, left_on='target', right_on='ID', how='left')
    merged_df.rename(columns={'X': 'target_X', 'Y': 'target_Y', 'Z': 'target_Z', 'node_type': 'target_node_type', 'ID':'target_ID'}, inplace=True)
    
    merged_df.drop(['source', 'target', 'source_ID', 'target_ID'], axis=1, inplace=True)
    merged_df.reset_index(drop=True, inplace = True)
    return merged_df

def load_dfs(data_type, gt_vector_csv_path, scale_feats):
    raw_df, nodes_df, edges_df = load_df_from_csv(gt_vector_csv_path)
    
    nodes_df_scaled = nodes_df.copy()
    if(scale_feats):
        nodes_df_scaled = scale_df_features(nodes_df_scaled, os.path.basename(gt_vector_csv_path))
    
    edges_df["edge_label"] = 1
    
    return raw_df, nodes_df, edges_df, nodes_df_scaled

def load_knn_df(nodes_df, edges_df, k_inter, k_intra, k_inter_max):
    
    if(k_inter=="min"):
        k_inter_curr = 1
        k_intra_curr = k_intra
        while(k_inter_curr<=k_inter_max):
            edge_types = {"nuclei-nuclei": k_intra_curr, "nuclei-golgi": k_inter_curr,
                     "golgi-golgi": k_intra_curr, "golgi-nuclei": 0}
            edges_df_knn = get_edges_knn(nodes_df, edge_types = edge_types)    
            nx_G_knn = nx_build_graph(nodes_df, edges_df_knn)
            is_connected = nx.is_connected(nx_G_knn)
            if(is_connected):
                break
            else:
                k_inter_curr+=1
    else:
        k_inter_curr = k_inter
        k_intra_curr= k_intra
        edge_types = {"nuclei-nuclei": k_intra_curr, "nuclei-golgi": k_inter_curr,
                     "golgi-golgi": k_intra_curr, "golgi-nuclei": 0}
        #apply the true labels to the edges_df_knn
        edges_df_knn = get_edges_knn(nodes_df, edge_types = edge_types)
    
    edges_df_knn["edge_label"] = 0
    if(edges_df.shape[0]>0):
        edges_df_knn = apply_edges_df_label(edges_df, edges_df_knn)
    
    return edges_df_knn, k_inter_curr, k_intra_curr

def eval_edges_df(edges_df_true, edges_df_pred):
    
    edges_info_dict = {}
    for idx, row in edges_df_true.iterrows():
        src = row["source"]
        tgt = row["target"]
        if(src not in edges_info_dict):
            edges_info_dict[src] = set()
        if(tgt not in edges_info_dict):
            edges_info_dict[tgt] = set()
        edges_info_dict[src].add(tgt)
        edges_info_dict[tgt].add(src)
    
    edge_labels_string = []
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    tp_total_count = len(edges_df_true)
    
    for idx, row in edges_df_pred.iterrows():
        pred_label = row["edge_label"]
        src = row["source"]
        tgt = row["target"]
        
        is_true_edge = (src in edges_info_dict) and (tgt in edges_info_dict[src])
        
        if(pred_label==1):
            if(is_true_edge):
                tp+=1
                edge_labels_string.append("tp")
            else:
                fp+=1
                edge_labels_string.append("fp")
        elif(pred_label==0):
            if(not is_true_edge):
                tn+=1
                edge_labels_string.append("tn")
            else:
                fn+=1   
                edge_labels_string.append("fn")
        else:
            raise ValueError("Wrong Label ! ->",pred_label)
    
    metrics = {}
    
    metrics["tp"] = tp
    metrics["fp"] = fp
    metrics["tn"] = tn
    metrics["fn"] = fn
    
    metrics["acc"] = round(calculate_acc(tp, fp, tn, fn) ,3)
    metrics["precision"] = round(calculate_precision(tp, fp) ,3)
    metrics["recall"] = round(calculate_recall(tp, fn) ,3)
    metrics["f1_score"] = round(calculate_f1_score(metrics["precision"], metrics["recall"]),3)
    metrics["tp_percent"] = round(tp/tp_total_count,3) if tp_total_count > 0 else 0
    metrics["tp_total_count"] = round(tp_total_count,3)
    
    return metrics, edge_labels_string

def df_make_plot(nodes_df, edges_df, edge_labels, title, plot_styles = {
                "nuclei":{"marker":"o","color":"red", "alpha":0.3},
                "golgi":{"marker":"o","color":"green", "alpha":0.3},
                "tp": {"color": "black",  "dashed": False, "alpha":1},
                "fp": {"color": "yellow", "dashed": False, "alpha":1},
                "tn": None,
                "fn": {"color": "blue",  "dashed": True, "alpha":1}
            }):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    node_list = nx_convert_nodes_df_to_list(nodes_df)
    edge_list = nx_convert_edges_df_to_list(edges_df)

    GraphInfo.plot_graph_nx_matplotlib(node_list, edge_list, edge_labels, ax, dims = 3, plot_styles = plot_styles, reversed_nodes_type_int_encodings = reversed_nodes_type_int_encodings)
    plt.title(title)
    
    #Legends
    
    # Create custom legend handles and labels based on plot_styles
    legend_handles = []
    legend_labels = []

    #Add Edge Legends
    for label, style in plot_styles.items():
        if isinstance(style, dict) and "dashed" in style:
            color = style["color"]
            dashed = style.get("dashed", False)
            alpha = style["alpha"]
            linestyle = "--" if dashed else "-"
            legend_handles.append(matplotlib.lines.Line2D([0], [0], color=color, linewidth=2, linestyle=linestyle, label=label, alpha = alpha))
            legend_labels.append(label.upper() + " Edge")

    # Add node legends
    legend_handles.append(matplotlib.lines.Line2D([0], [0], marker=plot_styles["nuclei"]["marker"], color="w", alpha = plot_styles["nuclei"]["alpha"],
                                    label="Nuclei", markerfacecolor=plot_styles["nuclei"]["color"], markersize=10))
    legend_labels.append("Nuclei")
    legend_handles.append(matplotlib.lines.Line2D([0], [0], marker=plot_styles["golgi"]["marker"], color="w", alpha = plot_styles["golgi"]["alpha"],
                                    label="Golgi", markerfacecolor=plot_styles["golgi"]["color"], markersize=10))
    legend_labels.append("Golgi")

    legend = fig.legend(legend_handles, legend_labels, loc="upper right", fontsize="small", ncol=2)
    
    return fig

####################################################
## Networkx           ##############################
####################################################

def nx_convert_dict_to_edges_df(edges_dict):
    edges_list = []
    for src, tgt in edges_dict.items():
        edges_list.append({"source": src, "target":tgt})

    edges_df = pd.DataFrame(edges_list)
    return edges_df

def nx_convert_edges_df_knn_to_pred(edges_df_knn, edges_df_pred):
    merged_edges_df = pd.merge(edges_df_knn, edges_df_pred, on=["source", "target"], how="right")
    return merged_edges_df

def nx_convert_edges_df_knn_to_pred(edges_df_knn, edges_df_bipartite):
    merged_edges_df = edges_df_knn.copy()
    
    merged_edges_df = pd.merge(edges_df_knn[['source', 'target', 'edge_label']], edges_df_bipartite, on=["source", "target"], how="right")
    
    return merged_edges_df

def nx_convert_edges_df_to_list(edges_df):
    """
    Converts a Pandas DataFrame of edges to a list of edges.

    Args:
    edges_df: A Pandas DataFrame of edges.

    Returns:
    A list of edges.
    """

    edges_list = []
    for i in range(len(edges_df)):
        edge = [edges_df.loc[i, "source"], edges_df.loc[i, "target"]]
        edge_attrs = {}
        for col in edges_df.columns:
            if col not in ["source", "target"]:
                edge_attrs[col] = edges_df.loc[i, col] 
        edge.append(edge_attrs)
        edges_list.append(edge)
    return edges_list

def nx_convert_nodes_df_to_list(nodes_df):
    nodes_list = []
    for index, row in nodes_df.iterrows():
        node_info = (row["ID"],{
            "Y": row["Y"],
            "X": row["X"],
            "Z": row["Z"],
            "node_type": row["node_type"]
        })
        nodes_list.append(node_info)
    return nodes_list

def nx_build_graph(nodes_df , edges_df, nodes_type_int_encodings=nodes_type_int_encodings):
    """
    Builds a bipartite networkx graph from a list with node information for nuclei (nuclei_list),
    a list with node information for golgi (golgi_lost) and another list with edge information.

    :param node_list: list of dictionaries with node information of nuclei
    :param golgi_list: list of dictionaries with node information of golgi
    :param edge_list: list of dictionaries with edge information
    :return: networkx Graph Bipartite instance
    """ 
    nuclei_list = nx_convert_nodes_df_to_list(nodes_df[nodes_df["node_type"]==nodes_type_int_encodings["nuclei"]].reset_index(drop=True))
    golgi_list = nx_convert_nodes_df_to_list(nodes_df[nodes_df["node_type"]==nodes_type_int_encodings["golgi"]].reset_index(drop=True))
    edge_list = nx_convert_edges_df_to_list(edges_df)

    G = nx.Graph(is_directed = True)
    
    # Add nodes with the node attribute "bipartite"
    G.add_nodes_from(nuclei_list, bipartite=0)
    G.add_nodes_from(golgi_list, bipartite=1)
    # Add edges only between nodes of opposite node set
    G.add_edges_from(edge_list)
    
    return G

#####################################
#### Pytorch Geometric          #####
####################################

class pyg_IdentityEncoder(object):
    """
    Encoder for Pytorch Geometric. Converts DataFrame to torch tensor.
    """ 
    def __init__(self, dtype = torch.float):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(np.array(df.values)).view(-1, 1).to(self.dtype)
    
def pyg_load_node_df(df : pd.DataFrame, encoder = pyg_IdentityEncoder(), node_feats_cols_to_remove = ["ID"], **kwargs):
    """
    Convert a node DataFrame to pytorch tensor.

    :param df: Pandas DataFrame with the nodes information
    :param encoder: Encoder object to encode the dataframe values and convert to pytorch tensor.
    :param  node_feats_cols_to_remove : columns from the dataframe to not count as node features
    :return: x (torch tensor with node attributes), y (torch tensor with node labels), 
                mapping (mapping dictionary for node ids)
    """ 
    
    xl = [encoder(df[col]) for col in df.columns if col not in node_feats_cols_to_remove]
    
    #if(len(xl)>0):
    x = torch.cat(xl, dim=-1)
    y = torch.squeeze(encoder(df["node_type"])).to(torch.int64)
    #else:
    #x= torch.tensor([])
    #y = torch.tensor([]).to(torch.int64)
    return x, y

def pyg_load_edge_df(df : pd.DataFrame, src_index_col : str, dst_index_col : str,
                  encoder = pyg_IdentityEncoder(), edge_feats_cols_to_remove = ["source","target","edge_label"], **kwargs):
    """
    Convert an edges DataFrame to pytorch tensor.

    :param df: Pandas DataFrame with the edges information
    :param src_index_col: name of the source column
    :param dst_index_col: name of the destination column
    :param encoder: Encoder object to encode the dataframe values and convert to pytorch tensor.
    :param  edge_feats_cols_to_remove : columns from the dataframe to not count as edge features
    :return: edge_index (torch tensor with dimensions (number_edges,2)), edge_attr (torch tensor with)
    """ 
    src = np.array(df[src_index_col].values)
    dst = np.array(df[dst_index_col].values)
    
    edge_index, edge_attr, edge_label, edge_label_index, edge_weight = None, None, None, None, None
    if(len(src)>0 and len(dst)>0):
        # Create a tensor from the concatenated NumPy ndarray
        edge_index = torch.tensor(np.stack((src, dst), axis=0), dtype=torch.int64)

        edge_attrs = [encoder(df[col]) for col in df.columns if col not in edge_feats_cols_to_remove]
        edge_attr = torch.cat(edge_attrs, dim=-1).to(torch.float32)

        edge_label = torch.squeeze(encoder(df["edge_label"])).to(torch.float32)
        edge_label_index = torch.tensor(np.stack((src, dst), axis=0), dtype=torch.int64)
        
        if "weight" in df.columns:
            edge_weight = torch.squeeze(encoder(df["weight"])).to(torch.float32)
        else:
            edge_weight = None
    return edge_index, edge_attr, edge_label, edge_label_index, edge_weight

def pyg_load_graph_from_df(nodes_df, edges_df, to_undirected = True, graph_type = "homo",
                         node_feats_cols_to_remove = ["ID"], edge_feats_cols_to_remove = ["source","target","edge_label"],
                        encoder = pyg_IdentityEncoder()):
    """
    Convert an edges DataFrame to pytorch geometric graph.

    :param nodes_df: Pandas DataFrame with the nodes information
    :param edges_df: Pandas DataFrame with the edges information
    :param to_undirected: boolean flag to convert graph to undirected or not. if undirected the number of edges in edge_index will be double
    :param graph_type: type of graph homo or hetero
    :return: Pytorch Geometric graph
    """     
    node_x, node_y = pyg_load_node_df(nodes_df, node_feats_cols_to_remove = node_feats_cols_to_remove)
    edge_index, edge_attr, edge_label, edge_label_index, edge_weight = pyg_load_edge_df(
        edges_df,
        src_index_col='source',
        dst_index_col='target',
        encoder = encoder,
        edge_feats_cols_to_remove = edge_feats_cols_to_remove
    )
    
    
    if(graph_type=="homo"):
        data =  torch_geometric.data.Data()
        data.x = node_x
        data.y = node_y
        #edges used for message passing
        data.edge_index = edge_index
        data.edge_label = edge_label
        #edges used for supervision
        #data.edge_label_index = edge_label_index
        data.edge_weight = edge_weight
        data.edge_attr = edge_attr
        data.edge_types = torch.tensor(list(edges_df["edge_type"]))
        
    else:
        raise ValueError("Only implemented Homogeneous Graphs! Please specify graph_type==\"homo\"")
        
    # We also need to make sure to add the reverse edges
    # in order to let a GNN be able to pass messages in both directions.
    # We can leverage the `T.ToUndirected()` transform for this from PyG:
    if(to_undirected):
        data = torch_geometric.transforms.ToUndirected()(data)
    
    return data

##########################
## Data Loading  #########
##########################

def distribute_elements_to_lists(elements, num_lists):
    #Used for the cross-validation to distribute elements evenly by num_lists
    num_elements = len(elements)
    assert(num_elements >= num_lists)
    elements_per_list = num_elements // num_lists  # Number of elements per list (floor division)

    # Initialize the lists
    lists = [[] for _ in range(num_lists)]

    # Distribute elements to each list
    for i, element in enumerate(elements):
        list_index = i % num_lists  # Cycle through the lists
        lists[list_index].append(element)

    return lists

def get_graph_list(data_type, k_inter, k_intra, k_inter_max, scale_feats = True, normalize = False, 
                                           shuffle = False, graph_id_filter = None,
                                            node_feats = "all", edge_feats = "all"):
                
    if data_type=="Real":
        vectors_dir = r'../data/vectors'
    elif data_type=="Real_automatic":
        vectors_dir = r'../data/vectors_automatic_csv'
    else:
        vectors_dir = data_type
    
    #Read the Data
    gt_vectors_csvs = os.listdir(vectors_dir)
    
    dfs_info_list = []
    
    for i in range(len(gt_vectors_csvs)):

        gt_vector_csv = gt_vectors_csvs[i]
        gt_vector_csv_path = os.path.join(vectors_dir, gt_vector_csv)
        graph_id = gt_vector_csv
        
        raw_df, nodes_df, edges_df, nodes_df_scaled = load_dfs(data_type, gt_vector_csv_path, scale_feats)
        
        dfs_info = {"raw_df":raw_df, "nodes_df":nodes_df_scaled, "edges_df":edges_df, 
                    "graph_id":gt_vector_csv, "nodes_df_original":nodes_df}
        dfs_info_list.append(dfs_info)
    
    for i ,dfs_info in enumerate(dfs_info_list):
        
        nodes_df, edges_df = dfs_info["nodes_df"], dfs_info["edges_df"]
        
        #Get KNN Graph
        edges_df_knn, k_inter_curr, k_intra_curr =  load_knn_df(nodes_df, edges_df, 
                                                                k_inter, k_intra, k_inter_max)
        #Apply Features
        edges_df = get_edge_features(edges_df, nodes_df)
        edges_df_knn = get_edge_features(edges_df_knn, nodes_df)

        #shuffle
        if(shuffle):
            nodes_df = nodes_df.sample(frac=1).reset_index(drop=True)
            edges_df = edges_df.sample(frac=1).reset_index(drop=True)
            edges_df_knn = edges_df_knn.sample(frac=1).reset_index(drop=True)
            
        dfs_info_list[i] = {**dfs_info,
                    "nodes_df":nodes_df, "edges_df":edges_df, "edges_df_knn":edges_df_knn, 
                    "k_inter":k_inter_curr, "k_intra":k_intra_curr, "node_feats":node_feats, "edge_feats":edge_feats}
    
    #Convert to Graph Info
    graph_list = []
    for dfs_info in dfs_info_list:
        raw_df, nodes_df, edges_df, \
        edges_df_knn, graph_id, k_inter, k_intra = dfs_info["raw_df"], dfs_info["nodes_df"], dfs_info["edges_df"],  \
                                dfs_info["edges_df_knn"], dfs_info["graph_id"],  dfs_info["k_inter"],  dfs_info["k_intra"]  
        
        graph_info = GraphInfo(raw_df, dfs_info["nodes_df_original"], nodes_df, edges_df, edges_df_knn, graph_id = graph_id, k_intra = k_intra, k_inter = k_inter,
                              node_feats = dfs_info["node_feats"], edge_feats = dfs_info["edge_feats"], normalize = normalize)
        graph_list.append(graph_info)
        
    return graph_list

def normalize_df(df):
    
    global_normalization = {"x":200, "y":150, "z":10, "delta":9}

    global_normalization["norm"] = np.sqrt(3*global_normalization["delta"]**2)

    data_columns = {"X":"x","Y":"y","Z":"z", 
                    "x1":"x", "x2":"x", "y1":"y", "y2":"y", "z1":"z", "z2":"z",
                    "weight":"norm",
                    "XN":"x","YN":"y","ZN":"z", "XG":"x","YG":"y","ZG":"z",
                     "delta_x":"delta","delta_y":"delta","delta_z":"delta"}
    
    for col in df.columns:
        if col in data_columns:
            min_value = 0#df[col].min()
            max_value = global_normalization[data_columns[col]]#df[col].max()
            df[col] = (df[col] - min_value) / (max_value - min_value)
    return df

def normalize_df_angles(df):
    
    phi_columns = ["angle_orientation_phi", "angle_rotation_phi"]
    theta_columns = ["angle_orientation_theta", "angle_rotation_theta"]
    angle_columns = phi_columns+theta_columns
    for col in df.columns:
        #theta range is from 0 to pi
        if col in theta_columns:
            df[col] = df[col]/np.pi
        #phi range is from 0 to 2pi, so divide by 2pi    
        elif col in phi_columns:
            df[col] = df[col]/2/np.pi
    return df

class GraphInfo:
    def __init__(self, 
                 raw_df, nodes_df_original,
                 nodes_df, edges_df, edges_df_knn, normalize = False, 
                 graph_id = None, k_inter = None, k_intra = None, node_feats = "all",
                edge_feats = "all", sort = False):
        
        self.graph_id = graph_id
        self.k_inter = k_inter
        self.k_intra = k_intra
        self.normalize = normalize
        
        self.raw_df = raw_df
        
        self.nodes_df_original = nodes_df_original
        self.nodes_df = nodes_df
        self.edges_df = edges_df
        self.edges_df_knn = edges_df_knn
        
        if(sort):
            self.nodes_df, self.edges_df, self.edges_df_knn = self.nodes_df.sample(frac=1).reset_index(drop=True), self.edges_df.sample(frac=1).reset_index(drop=True), self.edges_df_knn.sample(frac=1).reset_index(drop=True)
        
        if(self.normalize):
            self.nodes_df  = normalize_df(self.nodes_df)
            self.edges_df = normalize_df(self.edges_df)
            self.edges_df_knn = normalize_df(self.edges_df_knn)

        self.nodes_df = normalize_df_angles(self.nodes_df)
        self.edges_df = normalize_df_angles(self.edges_df)
        self.edges_df_knn = normalize_df_angles(self.edges_df_knn)

        
        node_feats_cols_to_remove = [col for col in list(self.nodes_df.columns) if col not in node_feats]
        node_feats_cols_to_remove = list(set(["ID"]+node_feats_cols_to_remove))
        
        edge_feats_cols_to_remove = [col for col in list(self.edges_df.columns) if col not in edge_feats]
        edge_feats_cols_to_remove = list(set(["source","target","edge_label"]+edge_feats_cols_to_remove))
        
        self.pyg_graph = pyg_load_graph_from_df(self.nodes_df, self.edges_df_knn, to_undirected = False, graph_type = "homo",
                             node_feats_cols_to_remove = node_feats_cols_to_remove, edge_feats_cols_to_remove = edge_feats_cols_to_remove,
                            encoder = pyg_IdentityEncoder())
            
        self.pyg_graph_edge_list, self.pyg_graph_true_labels, self.edge_list, self.edge_list_knn = [], np.array([]), [], []
        self.pyg_graph_edge_list = self.edge_index_to_edge_list(self.pyg_graph.edge_index)
        self.pyg_graph_true_labels = self.pyg_graph.edge_label.detach().cpu().numpy()
        self.edge_list = nx_convert_edges_df_to_list(self.edges_df)
        self.edge_list_knn = nx_convert_edges_df_to_list(self.edges_df_knn)
        self.node_list = nx_convert_nodes_df_to_list(self.nodes_df)   
        
        self.edge_x, self.edge_y = GraphInfo.convert_pygraph_to_numpy(self.pyg_graph)
        
        if node_feats != "all":
            self.nodes_df = self.nodes_df[node_feats]
        if edge_feats != "all":
            self.edges_df, self.edges_df_knn = self.edges_df[edge_feats] , self.edges_df_knn[edge_feats]
    
    @staticmethod
    def convert_pygraph_to_numpy(pygraph):
        # Initialize edge_x and edge_y
        edge_x = []
        edge_y = []

        # Iterate through each edge in edge_index
        for i in range(pygraph.edge_index.shape[1]):
            src_node = pygraph.edge_index[0, i]  # Source node index
            dst_node = pygraph.edge_index[1, i]  # Destination node index

            # Temporary list to store edge and node attributes
            temp = []

            # Add edge_attr for the edge to temp
            temp.extend(pygraph.edge_attr[i].tolist())

            # Append node attributes for source and destination nodes to temp
            src_attr = pygraph.x[src_node].tolist()
            if src_attr != None:
                temp.extend(src_attr)
            dst_attr = pygraph.x[dst_node].tolist()
            if dst_attr != None:
                temp.extend(dst_attr)
            # Append temp to edge_x
            edge_x.append(temp)
            # Append label from edge_label to edge_y
            edge_y.append(pygraph.edge_label[i])

        # Convert lists to numpy arrays
        edge_x = np.array(edge_x)
        edge_y = np.array(edge_y)

        return edge_x, edge_y
            
    @staticmethod
    def edge_index_to_edge_list(edge_index):
        """
        Convert PyTorch Geometric edge_index tensor to NetworkX edge_list format.

        Args:
            edge_index (torch.Tensor): Edge index tensor (2 x num_edges) in PyTorch Geometric format.
            num_nodes (int): Number of nodes in the graph.

        Returns:
            list: List of edges in NetworkX edge_list format.
        """
        edge_list = edge_index.t().tolist()  # Transpose edge_index and convert to list
        return edge_list
    
    @staticmethod
    def edge_list_to_edge_df(edge_list):
        data_list = []
        for edge in edge_list:
            data_list.append({"source":edge[0], "target":edge[1], "edge_label":1})
        edges_df = pd.DataFrame(data_list) 
        return edges_df
    
    @staticmethod
    def convert_edge_pred_to_label(true, pred):
        label_mapping = {
            (1, 1): "tp",
            (1, 0): "fn",
            (0, 1): "fp",
            (0, 0): "tn"
        }
        return label_mapping[(true, pred)]

    @staticmethod
    def convert_edge_preds_to_labels(true_labels, pred_labels):
        """
        Converts edge predicted output (0 or 1) to tp,fp,tn,fn

        Args:
            pred_labels (list): List of predicted edge labels (0 or 1) corresponding to each edge.
            true_labels (list): List of true edge labels (0 or 1) corresponding to each edge.

        Returns:
            pred_labels_list (list) : List of strings with "tp", "fp", "tn", "fn"
        """
        pred_labels_list = []
        
        for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
            
            label = GraphInfo.convert_edge_pred_to_label(true, pred)
            pred_labels_list.append(label)
        
        return pred_labels_list
    
    @staticmethod
    def plot_graph_nx_matplotlib(node_list, edge_list, edge_labels, ax, dims = 2, 
        plot_edges = True, 
        plot_styles = {
            "nuclei":{"marker":"o","color":"red", "alpha":0.3},
            "golgi":{"marker":"o","color":"green", "alpha":0.3},
            "tp": {"color": "black",  "dashed": False, "alpha":1},
            "fp": {"color": "yellow", "dashed": False, "alpha":1},
            "tn": None,
            "fn": {"color": "blue",  "dashed": True, "alpha":1}
        }, reversed_nodes_type_int_encodings = reversed_nodes_type_int_encodings):
        
        assert (len(edge_list)==len(edge_labels))
        dims_allowed_values = [2,3]
        if dims not in dims_allowed_values:
            raise ValueError("Wrong dims! Allowed values", str(dims_allowed_values))
        
        node_pos = {}
        # Draw nodes
        for node in node_list:
            node_id = node[0]
            node_info = node[1]
            node_type = reversed_nodes_type_int_encodings[node_info["node_type"]]
            node_color = plot_styles[node_type]["color"]#node_info["color"]
            node_alpha = plot_styles[node_type]["alpha"]
            marker = plot_styles[node_type]["marker"]
            node_size = 30#node_info["size"]
            pos = (node_info["X"], node_info["Y"], node_info["Z"])#node_info["coordinates"]#node_pos_list[node_id]
            node_pos[node_id] = pos
            
            if(dims ==2):
                ax.scatter(pos[0], pos[1], s=node_size, c=node_color, marker = marker, alpha = node_alpha)
            else:
                ax.scatter3D(pos[0], pos[1], pos[2], s=node_size, c=node_color, marker = marker, alpha = node_alpha)
        
        if plot_edges:
            linewidth = 1
            alpha = 1
            for i in range(len(edge_list)):
                u = edge_list[i][0]
                v = edge_list[i][1]
                edge_label = edge_labels[i]
                edge_style = plot_styles[edge_label]


                if(edge_style!=None):
                    color = edge_style["color"]
                    dashed = edge_style["dashed"]
                    alpha = edge_style["alpha"]

                    # Draw arrows
                    if dashed:
                        line_style = "--"
                    else:
                        line_style = "-"

                    if dims == 2:
                        line_kwargs = {
                            "linewidth": linewidth,
                            "color": color,
                            "alpha": alpha,
                            "linestyle": line_style                    
                        }
                        ax.plot([node_pos[u][0], node_pos[v][0]],
                                [node_pos[u][1], node_pos[v][1]], **line_kwargs)
                    else:
                        line_kwargs = {
                            "linewidth": linewidth,
                            "color": color,
                            "alpha": alpha,
                            "linestyle": line_style
                        }
                        ax.plot([node_pos[u][0], node_pos[v][0]],
                                [node_pos[u][1], node_pos[v][1]],
                                [node_pos[u][2], node_pos[v][2]], **line_kwargs)


        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        if(dims==3):
            ax.set_zlabel("Z")
        return ax
    
    @staticmethod
    def make_graph_plot(node_list, edge_list, edge_labels, plot_styles, dims = 2,
                        title = "", figax = None, figsize = (6,4), plot_edges = True):

        #True Graph
        graph_fig = figax
        if(not graph_fig):
            graph_fig = plt.figure(figsize=figsize)
        plt.title(title)
        
        #G_nx = build_graph_nx(node_list, edge_list)
        
        graph_fig = GraphInfo.plot_graph_nx_matplotlib(node_list, edge_list, edge_labels,  
                                                       graph_fig, dims = dims, plot_styles = plot_styles, 
                                                       plot_edges = plot_edges)
        
        return graph_fig
    
    @staticmethod
    def visualize_nodes_df(df, plot_styles = {
            "nuclei":{"marker":"o","color":"red", "alpha":0.3},
            "golgi":{"marker":"o","color":"green", "alpha":0.3},
            "tp": {"color": "black",  "dashed": False, "alpha":1},
            "fp": {"color": "yellow", "dashed": False, "alpha":1},
            "tn": None,
            "fn": {"color": "blue",  "dashed": True, "alpha":1}
        }, figsize = (6,4)):
        
        # Plot
        fig = plt.figure(figsize=figsize,dpi=250)
        ax = fig.add_subplot(111, projection="3d")
        ax.grid(False)
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    
        for idx, row in df.iterrows():
            x, y, z = row['X'], row['Y'], row['Z']
            node_type = reversed_nodes_type_int_encodings[row["node_type"]]
            node_color = plot_styles[node_type]["color"]
            node_alpha = plot_styles[node_type]["alpha"]
            marker = plot_styles[node_type]["marker"]
            ax.scatter3D(x, y, z, s=30, c=node_color, marker = marker, alpha = node_alpha)
        
        return ax 


########################
### Plots ##############
########################
    

def get_nested_dict(d, keys):
    nested_dict = None
    nested_dict = d.copy()
    for key in keys:
        nested_dict = nested_dict[key]
    return nested_dict

def plot_table(results_list, metrics_dict_entries = [["@best","metrics"],["@best","@constraints","metrics"], ["@constraints", "metrics"]],
              make_print = False):
    
    df_entries = []
    
    # Print the formatted values in the desired order
    for result in results_list:
        
        metrics = result["aggregated_metrics"]
        
        dataset_params = result["job_parameters"]
        data_type_train = dataset_params["data_type_train"]
        data_type_test = dataset_params.get("data_type_test",data_type_train)
        #angle_features = dataset_params["angle_features"]
        node_feats = dataset_params["node_feats"]
        edge_feats = dataset_params["edge_feats"]
        scale_features = dataset_params["scale_features"]
        normalize = dataset_params["normalize"]
        model_type = result["job_parameters"]["model_type"]
    
        k_intra = dataset_params["knn_intra_nodes"]
        k_inter = dataset_params["knn_inter_nodes"] 

        rouc_auc_score = metrics.get("rouc_auc_score","NA")
        
        count = 0
        for metrics_dict_entry in metrics_dict_entries:
            
            metrics_dict = None
            if metrics_dict_entry is None:
                metrics_dict = metrics
            else:
                metrics_dict = get_nested_dict(metrics, metrics_dict_entry)
                
            constraints = "NA" 
            if metrics_dict_entry is not None:
                constraints = "False"
                if "@constraints" in metrics_dict_entry:
                    if "@best" in metrics_dict_entry:
                        constraints = "Greedy w/ Threshold"
                    else:
                        constraints = "Greedy w/o Threshold"
                elif "@constraints_opt" in metrics_dict_entry:
                    constraints = "Optimization"
            
            acc =  metrics_dict["acc"]
            precision = metrics_dict["precision"]
            TPR = metrics_dict["TPR"]
            FPR = metrics_dict["FPR"]
            f1_score = metrics_dict["f1_score"]
            fn = metrics_dict["fn"]
            tp = metrics_dict["tp"]
            fp = metrics_dict["fp"]
            tn = metrics_dict["tn"]
            tp_percent = metrics_dict["tp_percent"]
            tp_total_count = metrics_dict["tp_total_count"]

            # Check and format values
            formatted_k_intra = f"{k_intra:.3f}" if isinstance(k_intra, float) else k_intra
            formatted_k_inter = f"{k_inter:.3f}" if isinstance(k_inter, float) else k_inter
            formatted_rouc_auc_score = f"{rouc_auc_score:.3f}" if isinstance(rouc_auc_score, float) else rouc_auc_score
            formatted_acc= f"{acc:.3f}" if isinstance(acc, float) else acc
            formatted_precision = f"{precision:.3f}" if isinstance(precision, float) else precision
            formatted_TPR = f"{TPR:.3f}" if isinstance(TPR, float) else TPR
            formated_fpr = f"{FPR:.3f}" if isinstance(FPR, float) else FPR
            formatted_f1_score = f"{f1_score:.3f}" if isinstance(f1_score, float) else f1_score
            formatted_tp = f"{tp:.3f}" if isinstance(tp, float) else tp
            formatted_fn = f"{fn:.3f}" if isinstance(fn, float) else fn
            formatted_tn = f"{tn:.3f}" if isinstance(tn, float) else tn
            formatted_fp = f"{fp:.3f}" if isinstance(fp, float) else fp
            formatted_tp_percent = f"{tp_percent:.3f}" if isinstance(tp_percent, float) else tp_percent
            formatted_tp_total_count = f"{tp_total_count:.3f}" if isinstance(tp_total_count, float) else tp_total_count

            formatted_values_dict = {
                "Algorithm": model_type, "Data Train": data_type_train, "Data Test":data_type_test, 
                "Constraints": constraints, "Normalize" :normalize, 
                "Scale": scale_features, "Node Feat.": node_feats, "Edge Feat.": edge_feats,
                "K Intra": formatted_k_intra, "K Inter": formatted_k_inter, 
                "ROC AUC Score": formatted_rouc_auc_score, "Accuracy": formatted_acc, 
                "Precision": formatted_precision, "TPR": formatted_TPR, "FPR": formated_fpr, "F1-Score": formatted_f1_score,
                "TP Percent": formatted_tp_percent, "TP Total Count": formatted_tp_total_count, 
                "TP":formatted_tp, "FP":formatted_fp, "TN":formatted_tn, "FN":formatted_fn
            }
            df_entries.append(formatted_values_dict)
            
    df = pd.DataFrame(df_entries).reset_index(drop=True)
    return df

def plot_df_to_latex(df, columns_to_drop = ['TP Total Count'], columns_min = ['FP', 'FN'], columns_max = ['ROC AUC Score', 
                            'TP Percent', 'F1-Score','Accuracy','Precision',
                             'Recall', "TP", "TN"]):
    #https://github.com/pandas-dev/pandas/issues/38328
    columns_to_drop = [col for col in columns_to_drop if col in  df.columns.tolist()]
    df = df.drop(columns=columns_to_drop)

    columns_min = [col for col in columns_min if col in  df.columns.tolist()]
    columns_max = [col for col in columns_max if col in  df.columns.tolist()]

    plot_str = df.style.highlight_min(subset= columns_min, props='textbf:--rwrap;')\
      .highlight_max(subset= columns_max, props='textbf:--rwrap').to_latex(hrules=True)
    
    return plot_str

########################
### Metrics ############
#########################

def calculate_acc(tp, fp, tn, fn):
    denominator = (tp + fp + tn + fn)
    if(denominator!=0):
        return (tp+tn)/denominator
    else:
        return 0
    
def calculate_precision(tp, fp):
    denominator = (tp+fp)
    if(denominator!=0):
        return tp/denominator
    else:
        return 0
    
def calculate_recall(tp, fn):
    denominator = (tp+fn)
    if(denominator!=0):
        return tp/denominator
    else:
        return 0
    
def calculate_fpr(fp, tn):
    denominator = (fp+tn)
    if(denominator!=0):
        return fp/denominator
    else:
        return 0
    
def calculate_f1_score(precision, recall):
    denominator = (precision+recall)
    if(denominator==0):
        return 0
    else:
        return (2*precision*recall)/denominator

@torch.no_grad()
def compute_confusion_matrix(true_labels, pred_labels, positive_label=1):
    """
    Compute the confusion matrix.

    :param true_labels: List of true labels (0 or 1)
    :param pred_labels: List of predicted labels (0 or 1)
    :param positive_label: The positive class label (default is 1)
    :return: Dictionary containing TP, FP, TN, FN.
    """
    assert len(true_labels) == len(pred_labels)

    TP, FP, TN, FN = 0, 0, 0, 0

    for true, pred in zip(true_labels, pred_labels):
        if true == positive_label:
            if pred == positive_label:
                TP += 1
            else:
                FN += 1
        else:
            if pred == positive_label:
                FP += 1
            else:
                TN += 1

    confusion_matrix = {
        "tp": TP,
        "fp": FP,
        "tn": TN,
        "fn": FN
    }

    return confusion_matrix

@torch.no_grad()
def eval_metrics(true : List, pred_labels : List, tp_total_count : int):
    """
    Function that calculates metrics using sklearn for predicted and true labels.

    :param true: list of true labels
    :param pred_labels: list of predicted labels
    :return: dictionary with metrics
    """ 
    assert (len(true) == len(pred_labels))
    
    metrics = {}
    confusion_matrix = compute_confusion_matrix(true,pred_labels)
    tp = confusion_matrix["tp"]
    fp = confusion_matrix["fp"]
    tn = confusion_matrix["tn"]
    fn = confusion_matrix["fn"]
    
    metrics["tp"] = tp
    metrics["fp"] = fp
    metrics["tn"] = tn
    metrics["fn"] = fn
    
    #metrics["acc"] = round(sklearn.metrics.accuracy_score(true, pred_labels),3)
    #metrics["precision"] = round(sklearn.metrics.precision_score(true,pred_labels, zero_division=0),3)
    #metrics["recall"] = round(sklearn.metrics.recall_score(true,pred_labels, zero_division=0),3)
    
    metrics["acc"] = round(calculate_acc(tp, fp, tn, fn) ,3)
    metrics["precision"] = round(calculate_precision(tp, fp) ,3)
    metrics["TPR"] = round(calculate_recall(tp, fn) ,3)
    metrics["FPR"] = round(calculate_fpr(fp, tn),3)
    metrics["f1_score"] = round(calculate_f1_score(metrics["precision"], metrics["TPR"]),3)
    
    metrics["tp_percent"] = round(tp/tp_total_count,3) if tp_total_count>0 else 0
    metrics["tp_total_count"] = tp_total_count
    
    return metrics

@torch.no_grad()
def aggregate_metrics(measurements):
    
    #computed metrics -> "acc", "precision", "recall", "tp", fp", "tn", "fn"
    aggregation_dict = {"acc":"mean", "precision":"mean", "TPR":"mean", "FPR":"mean", "f1_score":"mean", 
                        "tp":"sum", "fp":"sum", "tn":"sum", "fn":"sum",
                        "optimal_threshold":"mean", "tp_percent":"mean", "tp_total_count":"sum"}
    
    #initialization
    aggregated_metrics = {}
    for metric in aggregation_dict:
        if metric in measurements[0]:
            aggregated_metrics[metric] = []
    
    for measure in measurements:
        for metric in measure:
            if(metric not in ("@constraints")):
                aggregated_metrics[metric].append(measure[metric])
            
    for metric in aggregated_metrics:
        if metric in aggregation_dict:
            if(aggregation_dict[metric]=="mean"):
                aggregated_metrics[metric] = statistics.mean(aggregated_metrics[metric])
            elif(aggregation_dict[metric]=="sum"):
                aggregated_metrics[metric] = sum(aggregated_metrics[metric])
        else:
            raise ValueError("Metric not in metrics to be aggregated!")
        
    return aggregated_metrics

############################################################################
## Pytorch Early Stopping algorithms  ######################################
############################################################################

class EarlyStopper:
    """
    Early stopping that appends losses to lists and stops if there was no decrease in loss
    for the last epochs. The number of epochs where this can ocur before the stop is 
    equal to the patience parameter.
    """ 
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.losses = []

    def early_stop(self, validation_loss : List):
        last_losses = self.losses[-self.patience:]
        count = 0
        for loss in last_losses:
            if validation_loss > (loss + self.min_delta):
                    count+=1
                    if count>=self.patience:
                        self.losses.append(validation_loss)
                        return True
        self.losses.append(validation_loss)
        return False
    
#https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopperMinMax:
    """
    Early stopping that appends losses to lists and stops if the minimum loss did not 
    decrease for the last patience epochs. This is different from the previous class
    because it uses the global minimum instead of the minimum from the last patience epochs
    to calculate the stop.
    """ 
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss : List):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
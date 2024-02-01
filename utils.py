#############
##  MLP    ##
#############
#mlp.py

from typing import Dict, Union
from itertools import product

from typing import List, Any, Iterable

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, fc_dims: Iterable[int], nonlinearity: nn.Module,
                 dropout_p: float = 0, use_batchnorm: bool = False, last_output_free: bool = False):
        super().__init__()
        assert isinstance(fc_dims, (list, tuple)
                          ), f"fc_dims must be a list or a tuple, but got {type(fc_dims)}"

        self.input_dim = input_dim
        self.fc_dims = fc_dims
        self.nonlinearity = nonlinearity
        # if dropout_p is None:
        #     dropout_p = 0
        # if use_batchnorm is None:
        # use_batchnorm = False
        self.dropout_p = dropout_p
        self.use_batchnorm = use_batchnorm

        layers: List[nn.Module] = []
        for layer_i, dim in enumerate(fc_dims):
            layers.append(nn.Linear(input_dim, dim))
            if last_output_free and layer_i == len(fc_dims) - 1:
                continue

            layers.append(nonlinearity)
            if dim != 1:
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(dim))
                if dropout_p > 0:
                    layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_layers(x)

    @property
    def output_dim(self) -> int:
        return self.fc_dims[-1]
    
#############
## Utils   ##
#############

from typing import Iterable, Tuple, Mapping, Dict

import torch
from torch_scatter import scatter
from torch_geometric.utils import softmax as edge_softmax

#from models.mlp import MLP


def aggregate_features(features: torch.Tensor, agg_index: torch.Tensor, num_nodes: int, agg_mode: str, attention_model=None, edge_attr=None, **kwargs):
    if "attention" not in agg_mode:
        return scatter(src=features, index=agg_index, reduce=agg_mode, dim=0, dim_size=num_nodes)

    if "classifier" in agg_mode:
        assert edge_attr is not None
        coeffs = attention_model(edge_attr)
    else:
        coeffs = attention_model(features)
    assert coeffs.shape == (len(features), 1), f"features {features.shape}, coeffs {coeffs.shape}"
    # normalize coefficients for each node to sum to 1 (forward edges only)
    if "softmax" in agg_mode:
        coeffs = edge_softmax(coeffs, index=agg_index, num_nodes=num_nodes)
    elif "normalized" in agg_mode:
        coeffs = normalize_edge_coefficients(coeffs, index=agg_index, num_nodes=num_nodes)
    weighted_features = features * coeffs  # multiply by their attention coefficients and sum up
    return scatter(src=weighted_features, index=agg_index, reduce="sum", dim=0, dim_size=num_nodes)


def normalize_edge_coefficients(coeffs: torch.Tensor, index: torch.Tensor, num_nodes: int):
    coeffs_sum = scatter(coeffs, index=index, reduce="sum", dim=0, dim_size=num_nodes)
    coeffs_sum_selected = coeffs_sum.index_select(index=index, dim=0)
    return coeffs / (coeffs_sum_selected + 1e-16)


def dims_from_multipliers(output_dim: int, multipliers: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(output_dim * mult) for mult in multipliers)


def model_params_from_params(params: Mapping, model_prefix: str, param_names: Iterable[str]):
    return {param_name: params[f"{model_prefix}_{param_name}"] for param_name in param_names}
    # return {
    #     "input_dim":        params.get(f"{model_prefix}_input_dim"),
    #     "fc_dims":          params.get(f"{model_prefix}_fc_dims"),
    #     "nonlinearity":     params.get(f"{model_prefix}_nonlinearity"),
    #     "dropout_p":        params.get(f"{model_prefix}_dropout_p"),
    #     "use_batchnorm":    params.get(f"{model_prefix}_use_batchnorm"),
    # }

###########
## MPN ####
############
#mpn.py

from typing import List, Optional

import torch
from torch import nn
#import pytorch_lightning as pl
#from PolarMOT.models.mlp import MLP
from torch_geometric.data import Data

# TODO: rework these models to define only a single layer/pass and stack them in Sequential or something


class MessagePassingNetworkNonRecurrent(nn.Module):
    def __init__(self, edge_models: List[nn.Module], node_models: List[nn.Module], steps: int, use_same_frame: bool=False):
        """
        Args:
            edge_models: a list/tuple of callable Edge Update models
            node_models: a list/tuple of callable Node Update models
        """
        super().__init__()
        assert len(edge_models) == steps, f"steps={steps} not equal edge models {len(edge_models)}"
        assert len(node_models) == steps - 1, f"steps={steps} -1 not equal node models {len(node_models)}"
        self.edge_models = nn.ModuleList(edge_models)
        self.node_models = nn.ModuleList(node_models)
        self.steps = steps
        self.use_same_frame = use_same_frame

    def forward(self, x, edge_index, edge_attr, num_nodes: int):
        """
        Does a single node and edge feature vectors update.
        Args:
            x: node features matrix
            edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
            graph adjacency (i.e. edges)
            edge_attr: edge features matrix (ordered by edge_index)
        Returns: Updated Node and Edge Feature matrices
        """
        edge_embeddings = []
        for step, (edge_model, node_model) in enumerate(zip(self.edge_models, self.node_models.append(None))):
            # Edge Update
            edge_attr_mpn = edge_model(x, edge_index, edge_attr)
            edge_embeddings.append(edge_attr_mpn)

            if step == self.steps - 1:
                continue  # do not process nodes in the last step - only edge features are used for classification
            # Node Update
            x = node_model(x, edge_index, edge_attr_mpn)
        assert len(
            edge_embeddings) == self.steps, f"Collected {len(edge_embeddings)} edge embeddings for {self.steps} steps"
        return x, edge_embeddings


class MessagePassingNetworkRecurrent(nn.Module):
    def __init__(self, edge_model: nn.Module, node_model: nn.Module, steps: int,
                 use_same_frame: bool = False, same_frame_edge_model: Optional[nn.Module] = None):
        """
        Args:
            edge_model: an Edge Update model
            node_model: an Node Update model
        """
        super().__init__()
        self.edge_model = edge_model
        self.same_frame_edge_model = same_frame_edge_model
        self.node_model = node_model
        self.steps = steps
        self.use_same_frame = use_same_frame

    def forward(self, x, edge_index, edge_attr, num_nodes: int, same_frame_edge_index=None, same_frame_edge_attr=None):
        """
        Does a single node and edge feature vectors update.
        Args:
            x: node features matrix
            edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
            graph adjacency (i.e. edges)
            edge_attr: edge features matrix (ordered by edge_index)
        Returns: Updated Node and Edge Feature matrices
        """
        for step in range(self.steps):
            # Edge Update
            edge_attr_mpn = self.edge_model(x, edge_index, edge_attr)
            if self.use_same_frame:
                if self.same_frame_edge_model is not None:
                    same_frame_edge_attr_mpn = self.same_frame_edge_model(x, same_frame_edge_index, same_frame_edge_attr)
                else:
                    same_frame_edge_attr_mpn = self.edge_model(x, same_frame_edge_index, same_frame_edge_attr)
            else:
                same_frame_edge_attr_mpn = None

            if step == self.steps - 1:
                continue  # do not process nodes in the last step - only edge features are used for classification
            # Node Update
            x = self.node_model(x, edge_index, edge_attr_mpn,
                                same_frame_edge_index=same_frame_edge_index,
                                same_frame_edge_attr=same_frame_edge_attr_mpn)
        return x, edge_attr_mpn


class MessagePassingNetworkRecurrentNodeEdge(nn.Module):
    def __init__(self, edge_model: nn.Module, node_model: nn.Module, steps: int,
                 use_same_frame: bool = False, same_frame_edge_model: Optional[nn.Module] = None):
        """
        Args:
            edge_model: an Edge Update model
            node_model: an Node Update model
        """
        super().__init__()
        self.edge_model = edge_model
        self.same_frame_edge_model = same_frame_edge_model
        self.node_model = node_model
        self.steps = steps
        self.use_same_frame = use_same_frame

    def forward(self, x, edge_index, edge_attr, num_nodes: int, same_frame_edge_index=None, same_frame_edge_attr=None):
        """
        Does a single node and edge feature vectors update.
        Args:
            x: node features matrix
            edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
            graph adjacency (i.e. edges)
            edge_attr: edge features matrix (ordered by edge_index)
        Returns: Updated Node and Edge Feature matrices
        """
        edge_attr_mpn = edge_attr
        same_frame_edge_attr_mpn = same_frame_edge_attr
        for step in range(self.steps):
            # Node Update
            x = self.node_model(x, edge_index, edge_attr_mpn,
                                same_frame_edge_index=same_frame_edge_index,
                                same_frame_edge_attr=same_frame_edge_attr_mpn)

            # Edge Update
            edge_attr_mpn = self.edge_model(x, edge_index, edge_attr)
            if self.use_same_frame:
                if self.same_frame_edge_model is not None:
                    same_frame_edge_attr_mpn = self.same_frame_edge_model(x, same_frame_edge_index, same_frame_edge_attr)
                else:
                    same_frame_edge_attr_mpn = self.edge_model(x, same_frame_edge_index, same_frame_edge_attr)
            else:
                same_frame_edge_attr_mpn = None

        return x, edge_attr_mpn
    
####################
## Edge Models #####
####################
#edge_models.py

from typing import List, Any

import torch
from torch import nn
#from PolarMOT.models.mlp import MLP


class BasicEdgeModel(nn.Module):
    """ Class used to peform an edge update during neural message passing """

    def __init__(self, edge_mlp):
        super(BasicEdgeModel, self).__init__()
        self.edge_mlp = edge_mlp

    def forward(self, x, edge_index, edge_attr):
        source_nodes, target_nodes = edge_index
        # assert len(source_nodes) == len(target_nodes) == len(
            # edge_attr), f"Different lengths {len(source_nodes)}, {len(target_nodes)}, {len(edge_attr)} "
        merged_features = torch.cat([x[source_nodes], x[target_nodes], edge_attr], dim=1)
        # print(f"merged_features, {merged_features.shape}")
        assert len(merged_features) == len(source_nodes), f"Merged input has wrong length {merged_features.shape} != {edge_attr.shape}"
        return self.edge_mlp(merged_features)
    
####################
## Node Models #####
####################
#node_models.py

from tkinter import E
from typing import List, Any

import torch
from torch import nn
from torch_scatter import scatter
from torch_geometric.utils import softmax as edge_softmax

#from models.mlp import MLP
#from models.utils import aggregate_features


class TimeAwareNodeModel(nn.Module):
    """ Class used to peform a node update during neural message passing """

    def __init__(self, flow_forward_model: MLP, flow_backward_model: MLP, node_mlp: MLP, node_agg_mode: str):
        super().__init__()
        assert (flow_forward_model.output_dim + flow_backward_model.output_dim ==
                node_mlp.input_dim), f"Flow models have incompatible output/input dims"
        self.flow_forward_model = flow_forward_model
        self.flow_backward_model = flow_backward_model
        self.node_mlp = node_mlp
        self.node_agg_mode = node_agg_mode

    def forward(self, x, edge_index, edge_attr, **kwargs):
        past_nodes, future_nodes = edge_index

        """
        x[past_nodes]  # features of nodes emitting messages
        edge_attr       # emitted messages
        x[future_nodes] # features of nodes receiving messages
        """
        # Collect and process forward-directed edge messages (past->present)
        # print(f"x[past_nodes], {x[past_nodes].shape}")
        # print(f"edge_attr, {edge_attr.shape}")

        # TODO: Try actually sharing edges: use a single flow MLP and then scatter twice - with past/future_nodes as index

        flow_forward_input = torch.cat([x[future_nodes], x[past_nodes], edge_attr], dim=1)  # [n_edges x edge_feature_count]
        # print(f"flow_forward_input, {flow_forward_input.shape}")
        assert len(flow_forward_input) == len(edge_attr)

        flow_forward = self.flow_forward_model(flow_forward_input)
        # print(f"flow_forward, {flow_forward.shape}")
        # print(f"flow_forward {flow_forward}")
        # print(f"flow_forward {flow_forward.grad_fn}")

        flow_forward_aggregated = scatter(src=flow_forward, index=future_nodes,
                                          reduce=self.node_agg_mode, dim=0, dim_size=len(x))
        # print(f"flow_forward_aggregated {flow_forward_aggregated}")
        # print(f"flow_forward_aggregated {flow_forward_aggregated.grad_fn}")

        # Collect and process backward-directed edge messages (present->past)
        flow_backward_input = torch.cat([x[past_nodes], x[future_nodes], edge_attr], dim=1)  # [n_edges x edge_feature_count]
        flow_backward = self.flow_backward_model(flow_backward_input)
        # print(f"flow_backward {flow_backward}")
        flow_backward_aggregated = scatter(src=flow_backward, index=past_nodes,
                                           reduce=self.node_agg_mode, dim=0, dim_size=len(x))
        # print(f"flow_backward_aggregated {flow_backward_aggregated}")
        # print(f"flow_backward_aggregated {flow_backward_aggregated.grad_fn}")

        # Concat both flows for each node
        flow_total = torch.cat((flow_forward_aggregated, flow_backward_aggregated), dim=1)
        return self.node_mlp(flow_total)


class ContextualNodeModel(nn.Module):
    """ Class used to peform a node update during neural message passing """

    def __init__(self, flow_forward_model: MLP, flow_frame_model: MLP, flow_backward_model: MLP,
                 total_flow_model: MLP, node_agg_mode: str, attention_model: nn.Module = None, node_aggr_sections: int = 3):
        super().__init__()
        assert (flow_forward_model.output_dim + flow_frame_model.output_dim + flow_backward_model.output_dim ==
                total_flow_model.input_dim), f"Flow models have incompatible output/input dims"
        self.flow_forward_model = flow_forward_model
        self.flow_frame_model = flow_frame_model
        self.flow_backward_model = flow_backward_model
        self.node_agg_mode = node_agg_mode
        self.attention_model = attention_model
        self.total_flow_model = total_flow_model
        self.node_aggr_sections = node_aggr_sections

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                same_frame_edge_index: torch.Tensor, same_frame_edge_attr: torch.Tensor, **kwargs):
        past_nodes, future_nodes = edge_index
        """
        x[past_nodes]  # features of nodes emitting messages
        edge_attr       # emitted messages
        x[future_nodes] # features of nodes receiving messages
        """
        early_frame_nodes, later_frame_nodes = same_frame_edge_index

        # Collect and process forward-directed edge messages (past->present)
        flow_forward_input = torch.cat([x[future_nodes], x[past_nodes], edge_attr], dim=1)  # [n_edges x edge_feature_count]
        flow_forward = self.flow_forward_model(flow_forward_input)

        # Collect and process backward-directed edge messages (present->past)
        # [n_edges x edge_feature_count]
        flow_backward_input = torch.cat([x[past_nodes], x[future_nodes], edge_attr], dim=1)
        flow_backward = self.flow_backward_model(flow_backward_input)

        # Collect and process same frame edge messages
        flow_frame_input = torch.cat([x[early_frame_nodes],
                                      x[later_frame_nodes],
                                      same_frame_edge_attr], dim=1)  # [n_edges x edge_feature_count]
        flow_frame = self.flow_frame_model(flow_frame_input)

        if self.node_aggr_sections == 3:
            flow_forward_aggregated = aggregate_features(flow_forward, future_nodes,
                                                        len(x), self.node_agg_mode, self.attention_model, edge_attr=edge_attr)
            flow_backward_aggregated = aggregate_features(flow_backward, past_nodes,
                                                        len(x), self.node_agg_mode, self.attention_model, edge_attr=edge_attr)
            flow_frame_aggregated = aggregate_features(torch.vstack((flow_frame, flow_frame)),
                                                    torch.cat((early_frame_nodes, later_frame_nodes)),
                                                    len(x), self.node_agg_mode, self.attention_model, 
                                                    edge_attr=torch.vstack((same_frame_edge_attr, same_frame_edge_attr)))
        elif self.node_aggr_sections == 2:
            flow_temporal_aggregated = aggregate_features(torch.vstack((flow_forward, flow_backward)), 
                                                        torch.cat((future_nodes, past_nodes)),
                                                        len(x), self.node_agg_mode, self.attention_model, 
                                                        edge_attr=edge_attr)
            # print("flow_temporal_aggregated", flow_temporal_aggregated.shape)
            flow_temporal_aggregated /= 2.0
            flow_forward_aggregated = flow_temporal_aggregated
            flow_backward_aggregated = flow_temporal_aggregated
            flow_frame_aggregated = aggregate_features(torch.vstack((flow_frame, flow_frame)),
                                                    torch.cat((early_frame_nodes, later_frame_nodes)),
                                                    len(x), self.node_agg_mode, self.attention_model, 
                                                    edge_attr=torch.vstack((same_frame_edge_attr, same_frame_edge_attr)))
        elif self.node_aggr_sections == 1:
            flow_total_aggregated = aggregate_features(torch.vstack((flow_forward, flow_frame, flow_frame, flow_backward)),
                                                    torch.cat((future_nodes, early_frame_nodes, later_frame_nodes, past_nodes)),
                                                    len(x), self.node_agg_mode, self.attention_model, 
                                                    edge_attr=torch.vstack((edge_attr, same_frame_edge_attr, same_frame_edge_attr, edge_attr)))
            # print("flow_total_aggregated", flow_total_aggregated.shape)
            flow_total_aggregated /= 3.0
            flow_forward_aggregated = flow_total_aggregated
            flow_backward_aggregated = flow_total_aggregated
            flow_frame_aggregated = flow_total_aggregated

        # stack and aggregate everything, then triple duplicate for the total_flow_model

        # Concat both flows for each node
        flow_total = torch.cat((flow_forward_aggregated, flow_frame_aggregated, flow_backward_aggregated), dim=1)
        return self.total_flow_model(flow_total)


class UniformAggNodeModel(nn.Module):
    """ Class used to peform a node update during neural message passing """

    def __init__(self, flow_model, node_mlp, node_agg_mode: str):
        super().__init__()
        assert (flow_model.output_dim == node_mlp.input_dim), f"Flow models have incompatible output/input dims"
        self.flow_model = flow_model
        self.node_mlp = node_mlp
        self.node_agg_mode = node_agg_mode

    def forward(self, x, edge_index, edge_attr, **kwargs):
        past_nodes, future_nodes = edge_index

        """
        x[past_nodes]  # features of nodes emitting messages, past -> future
        edge_attr       # emitted messages
        x[future_nodes] # features of nodes receiving messages, future nodes receiving from earlier ones
        """
        # input features order does not matter as long as it is symmetric between two flow inputs
        #                               nodes receiving, nodes sending, edges
        flow_forward_input = torch.hstack((x[future_nodes], x[past_nodes], edge_attr))
        assert len(flow_forward_input) == len(edge_attr)
        # [n_edges x edge_feature_count]
        flow_backward_input = torch.hstack((x[past_nodes], x[future_nodes], edge_attr))
        assert flow_forward_input.shape == flow_backward_input.shape, f"{flow_forward_input.shape} != {flow_backward_input.shape}"

        # [2*n_edges x edge_feature_count]
        flow_total_input = torch.vstack((flow_forward_input, flow_backward_input))
        flow_processed = self.flow_model(flow_total_input)

        # aggregate features for each node based on features taken over each node
        # the index has to account for both incoming and outgoing edges - so that each edge is considered by both of its nodes
        flow_total = scatter(src=flow_processed, index=torch.cat((future_nodes, past_nodes)),
                             reduce=self.node_agg_mode, dim=0, dim_size=len(x))
        return self.node_mlp(flow_total)


class InitialTimeAwareNodeModel(nn.Module):
    """ Class used to peform an initial update for empty nodes at the beginning of message passing
    The initial features are simply a transformation of edge features and therefore depend largely on the graph structure
    The aggregation logic is the same as for `TimeAwareNodeModel` but without node features.

    The abscense of node features means that transforming features before aggregation is the same as transforming the input edge features before the node update. Therefore, the transform is only applied after aggregation and instead the initial edge model is more powerful in comparison.
    """

    def __init__(self, node_mlp, node_agg_mode: str):
        super().__init__()
        self.node_mlp = node_mlp
        self.node_agg_mode = node_agg_mode

    def forward(self, edge_index, edge_attr, num_nodes: int, **kwargs):
        past_nodes, future_nodes = edge_index
        # print(f"edge_attr, {edge_attr.shape}")

        flow_forward_aggregated = scatter(src=edge_attr, index=future_nodes,
                                          reduce=self.node_agg_mode, dim=0, dim_size=num_nodes)
        flow_backward_aggregated = scatter(src=edge_attr, index=past_nodes,
                                           reduce=self.node_agg_mode, dim=0, dim_size=num_nodes)

        # Concat both flows for each node
        flow_total = torch.cat((flow_forward_aggregated, flow_backward_aggregated), dim=1)
        # print(f"flow_total {flow_total.shape}")
        # print(f"initial flow_total {flow_total.grad_fn}")
        assert len(flow_total) == num_nodes
        return self.node_mlp(flow_total)


class InitialContextualNodeModel(nn.Module):
    """ Class used to peform an initial update for empty nodes at the beginning of message passing
    The initial features are simply a transformation of edge features and therefore depend largely on the graph structure
    The aggregation logic is the same as for `ContextualNodeModel` but without node features.

    The abscense of node features means that transforming features before aggregation is the same as transforming the input edge features before the node update. Therefore, the transform is only applied after aggregation and instead the initial edge model is more powerful in comparison.
    """

    def __init__(self, node_mlp, node_agg_mode: str, attention_model: nn.Module = None):
        super().__init__()
        self.node_mlp = node_mlp
        self.node_agg_mode = node_agg_mode
        self.attention_model = attention_model

    def forward(self, edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int,
                same_frame_edge_index: torch.Tensor, same_frame_edge_attr: torch.Tensor, **kwargs):
        past_nodes, future_nodes = edge_index
        early_frame_nodes, later_frame_nodes = same_frame_edge_index

        flow_forward_aggregated = aggregate_features(edge_attr, future_nodes,
                                                     num_nodes, self.node_agg_mode, self.attention_model)
        flow_backward_aggregated = aggregate_features(edge_attr, past_nodes,
                                                      num_nodes, self.node_agg_mode, self.attention_model)
        flow_frame_aggregated = aggregate_features(torch.vstack((same_frame_edge_attr, same_frame_edge_attr)),
                                                   torch.cat((early_frame_nodes, later_frame_nodes)),
                                                   num_nodes, self.node_agg_mode, self.attention_model)
        # Concat all flows for each node
        flow_total = torch.cat((flow_forward_aggregated, flow_frame_aggregated, flow_backward_aggregated), dim=1)
        assert len(flow_total) == num_nodes
        return self.node_mlp(flow_total)


class InitialUniformAggNodeModel(nn.Module):
    """ Class used to peform an initial update for empty nodes at the beginning of message passing
    The initial features are simply a transformation of edge features and therefore depend largely on the graph structure
    The aggregation logic is the same as for `UniformAggNodeModel` but without node features.

    The abscense of node features means that transforming features before aggregation is the same as transforming the input edge features before the node update. Therefore, the transform is only applied after aggregation and instead the initial edge model is more powerful in comparison.
    """

    def __init__(self, node_mlp, node_agg_mode: str):
        super().__init__()
        self.node_mlp = node_mlp
        self.node_agg_mode = node_agg_mode

    def forward(self, edge_index, edge_attr, num_nodes: int, **kwargs):
        past_nodes, future_nodes = edge_index
        flow_total = scatter(src=torch.vstack((edge_attr, edge_attr)),
                             index=torch.cat((future_nodes, past_nodes)),
                             reduce=self.node_agg_mode, dim=0, dim_size=num_nodes)
        return self.node_mlp(flow_total)


class InitialZeroNodeModel(nn.Module):
    """ Class used to peform an initial update for empty nodes at the beginning of message passing
    The initial features are simply a transformation of edge features and therefore depend largely on the graph structure
    The aggregation logic is the same as for `ContextualNodeModel` but without node features.

    The abscense of node features means that transforming features before aggregation is the same as transforming the input edge features before the node update. Therefore, the transform is only applied after aggregation and instead the initial edge model is more powerful in comparison.
    """

    def __init__(self, node_dim: int):
        super().__init__()
        self.node_dim = node_dim

    def forward(self, edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int,
                same_frame_edge_index: torch.Tensor, same_frame_edge_attr: torch.Tensor, device, **kwargs):
        return torch.zeros((num_nodes, self.node_dim), dtype=edge_attr.dtype, device=device)
    
###########################
## Utils Models ###########
##########################
#utils_models.py
def dims_from_multipliers(output_dim: int, multipliers: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(output_dim * mult) for mult in multipliers)


###########################
## Graph Tracker Offline  #
###########################
#graph_tracker_offline.py

from typing import List, Mapping, Any, Iterable, Tuple
import traceback
import os

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Data
from torchmetrics import Precision, Recall

"""
from models.mlp import MLP
from models.mpn import MessagePassingNetworkRecurrent, MessagePassingNetworkRecurrentNodeEdge, MessagePassingNetworkNonRecurrent
from models.node_models import (InitialTimeAwareNodeModel, TimeAwareNodeModel,
                                UniformAggNodeModel, InitialUniformAggNodeModel,
                                ContextualNodeModel, InitialContextualNodeModel,
                                InitialZeroNodeModel)
from models.edge_models import BasicEdgeModel
"""
#import PolarMOT.models.utils as utils_models

from PolarMOT.training.radam import RAdam
from PolarMOT.training.focal_loss import focal_loss_binary


def _build_params_dict(initial_edge_model_input_dim, edge_dim, fc_dims_initial_edge_model_multipliers, nonlinearity_initial_edge,
                       fc_dims_initial_node_model_multipliers, nonlinearity_initial_node, 
                       directed_flow_agg, fc_dims_directed_flow_attention_model_multipliers,
                       fc_dims_edge_model_multipliers, nonlinearity_edge,
                       fc_dims_directed_flow_model_multipliers, nonlinearity_directed_flow, 
                       fc_dims_total_flow_model_multipliers, nonlinearity_total_flow,
                       fc_dims_edge_classification_model_multipliers, nonlinearity_edge_classification,
                       use_batchnorm: bool,
                       mpn_steps: int, is_recurrent: bool, node_dim_multiplier: int, pos_weight_multiplier: int,
                       use_timeaware: bool, use_same_frame: bool, use_separate_edge_model: bool, use_initial_node_model: bool,
                       edge_mlps_count: int,
                       node_aggr_sections: int,
                       lr, wd, loss_type: str,
                       seed,
                       optimizer_type,
                       scheduler_params: Mapping,
                       trainer_params: Mapping,
                       **kwargs,
                       ):
    # workaround before adding sacred
    params = {
        "seed": seed,

        "initial_edge_model_input_dim": initial_edge_model_input_dim,
        "edge_dim": edge_dim,

        "fc_dims_initial_edge_model_multipliers": fc_dims_initial_edge_model_multipliers,
        "nonlinearity_initial_edge": nonlinearity_initial_edge,

        "fc_dims_initial_node_model_multipliers": fc_dims_initial_node_model_multipliers,
        "nonlinearity_initial_node": nonlinearity_initial_node,
        "directed_flow_agg": directed_flow_agg,
        "fc_dims_directed_flow_attention_model_multipliers": fc_dims_directed_flow_attention_model_multipliers,

        "fc_dims_edge_model_multipliers": fc_dims_edge_model_multipliers,
        "nonlinearity_edge": nonlinearity_edge,

        "fc_dims_directed_flow_model_multipliers": fc_dims_directed_flow_model_multipliers,
        "nonlinearity_directed_flow": nonlinearity_directed_flow,

        "fc_dims_total_flow_model_multipliers": fc_dims_total_flow_model_multipliers,
        "nonlinearity_total_flow": nonlinearity_total_flow,

        "fc_dims_edge_classification_model_multipliers": fc_dims_edge_classification_model_multipliers,
        "nonlinearity_edge_classification": nonlinearity_edge_classification,

        "use_batchnorm": use_batchnorm,

        "mpn_steps": mpn_steps,
        "is_recurrent": is_recurrent,
        "node_dim_multiplier": node_dim_multiplier,
        "pos_weight_multiplier": pos_weight_multiplier,

        "use_timeaware": use_timeaware,
        "use_same_frame": use_same_frame,
        "use_separate_edge_model": use_separate_edge_model,
        "use_initial_node_model": use_initial_node_model,
        "edge_mlps_count": edge_mlps_count,
        "node_aggr_sections": node_aggr_sections,

        "lr": lr,
        "wd": wd,
        "loss_type": loss_type,
        "optimizer_type": optimizer_type,
        "scheduler_params": scheduler_params,

        "trainer_params": trainer_params,
    }
    params.update(kwargs)
    return params


def _build_models(params: Mapping[str, Any]):
    use_batchnorm = params["use_batchnorm"]

    edge_dim = params["edge_dim"]
    node_dim_multiplier = params.get("node_dim_multiplier", 2)
    node_dim = edge_dim * node_dim_multiplier  # Have nodes hold 2x info of edges
    use_timeaware = params.get("use_timeaware", True)
    use_same_frame = params.get("use_same_frame", False)
    # separate backward/forward/sameframe MLPs or inter/intraframe or single MLP for all
    edge_mlps_count = params.get("edge_mlps_count", 3)
    assert edge_mlps_count > 0 and edge_mlps_count <= 3, f"edge_mlps_count must be 1/2/3, not {edge_mlps_count}"
    node_aggr_sections = params.get("node_aggr_sections", 3)
    assert node_aggr_sections > 0 and node_aggr_sections <= 3, f"node_aggr_sections must be 1/2/3, not {node_aggr_sections}"
    # only makes sense when using intraframe
    use_separate_edge_model = use_same_frame and params.get("use_separate_edge_model", False) 
    use_initial_node_model = params.get("use_initial_node_model", True)

    # Edge classification model
    fc_dims_edge_classification_model_multipliers = params["fc_dims_edge_classification_model_multipliers"]
    if fc_dims_edge_classification_model_multipliers is not None:
        fc_dims_edge_classification_model = dims_from_multipliers(
            edge_dim, fc_dims_edge_classification_model_multipliers) + (1,)
    else:
        fc_dims_edge_classification_model = (1,)
    edge_classifier = MLP(edge_dim, fc_dims_edge_classification_model,
                          params["nonlinearity_edge_classification"], last_output_free=True)

    # Initial edge model:
    fc_dims_initial_edge = dims_from_multipliers(
        edge_dim, params["fc_dims_initial_edge_model_multipliers"])
    initial_edge_model = MLP(params["initial_edge_model_input_dim"], fc_dims_initial_edge,
                            params["nonlinearity_initial_edge"], use_batchnorm=use_batchnorm)
    if use_separate_edge_model:
        initial_same_frame_edge_model = MLP(params["initial_edge_model_input_dim"], fc_dims_initial_edge,
                                            params["nonlinearity_initial_edge"], use_batchnorm=use_batchnorm)
    else:
        initial_same_frame_edge_model = None

    # Initial node model
    if use_initial_node_model:
        initial_node_agg_mode = params["directed_flow_agg"]
        if "attention" in initial_node_agg_mode:
            if "classifier" in initial_node_agg_mode:
                initial_node_attention_model = edge_classifier
            else:
                fc_dims_directed_flow_attention_model_multipliers = params["fc_dims_directed_flow_attention_model_multipliers"]
                if fc_dims_directed_flow_attention_model_multipliers is not None:
                    fc_dims_initial_node_attention = dims_from_multipliers(
                        edge_dim, fc_dims_directed_flow_attention_model_multipliers) + (1,)
                else:
                    fc_dims_initial_node_attention = (1,)
                initial_node_attention_model = MLP(edge_dim, fc_dims_initial_node_attention,
                                                params["nonlinearity_initial_node"], last_output_free=True)
        else:
            initial_node_attention_model = None

        fc_dims_initial_node = dims_from_multipliers(
            node_dim, params["fc_dims_initial_node_model_multipliers"])
        if use_timeaware:
            if use_same_frame:
                initial_node_model = InitialContextualNodeModel(MLP(edge_dim * 3, fc_dims_initial_node,
                                                                params["nonlinearity_initial_node"], use_batchnorm=use_batchnorm),
                                                                initial_node_agg_mode, initial_node_attention_model)
            else:
                initial_node_model = InitialTimeAwareNodeModel(MLP(edge_dim * 2, fc_dims_initial_node,  # x2 for [forward|backward] edge features
                                                                   params["nonlinearity_initial_node"], use_batchnorm=use_batchnorm),
                                                               initial_node_agg_mode)
        else:
            assert not use_same_frame
            initial_node_model = InitialUniformAggNodeModel(MLP(edge_dim, fc_dims_initial_node,
                                                                params["nonlinearity_initial_node"], use_batchnorm=use_batchnorm),
                                                            initial_node_agg_mode)
    else:  # initial nodes are zero vectors
        initial_node_model = InitialZeroNodeModel(node_dim)

    # Define models in MPN
    edge_models, node_models = [], []
    steps = params["mpn_steps"]
    assert steps > 1, "Fewer than 2 MPN steps does not make sense as in that case nodes do not get a chance to update"
    is_recurrent = params["is_recurrent"]
    for step in range(steps):
        # Edge model
        edge_model_input = node_dim * 2 + edge_dim  # edge_dim * 5
        fc_dims_edge = dims_from_multipliers(
            edge_dim, params["fc_dims_edge_model_multipliers"])
        edge_models.append(BasicEdgeModel(MLP(edge_model_input, fc_dims_edge,
                                              params["nonlinearity_edge"], use_batchnorm=use_batchnorm)))

        if step == steps - 1: # don't need a node update at the last step
            continue

        # Node model
        flow_model_input = node_dim * 2 + edge_dim  # two nodes and their edge
        fc_dims_directed_flow = dims_from_multipliers(
            node_dim, params["fc_dims_directed_flow_model_multipliers"])
        fc_dims_aggregated_flow = dims_from_multipliers(
            node_dim, params["fc_dims_total_flow_model_multipliers"])
        
        node_agg_mode = params["directed_flow_agg"]
        if "attention" in node_agg_mode:
            if "classifier" in node_agg_mode:
                attention_model = edge_classifier
            else:
                fc_dims_directed_flow_attention_model_multipliers = params["fc_dims_directed_flow_attention_model_multipliers"]
                if fc_dims_directed_flow_attention_model_multipliers is not None:
                    fc_dims_directed_flow_attention = dims_from_multipliers(
                        node_dim, fc_dims_directed_flow_attention_model_multipliers) + (1,)
                else:
                    fc_dims_directed_flow_attention = (1,)
                attention_model = MLP(node_dim, fc_dims_directed_flow_attention,
                                    params["nonlinearity_directed_flow"], last_output_free=True)
        else:
            attention_model = None

        if use_timeaware:
            forward_flow_model = MLP(flow_model_input, fc_dims_directed_flow,
                                     params["nonlinearity_directed_flow"], use_batchnorm=use_batchnorm)
            if edge_mlps_count < 3:
                backward_flow_model = forward_flow_model
            else:
                backward_flow_model = MLP(flow_model_input, fc_dims_directed_flow,
                                        params["nonlinearity_directed_flow"], use_batchnorm=use_batchnorm)
            if use_same_frame:
                if edge_mlps_count == 1:
                    frame_flow_model = forward_flow_model
                else:
                    frame_flow_model = MLP(flow_model_input, fc_dims_directed_flow,
                                        params["nonlinearity_directed_flow"], use_batchnorm=use_batchnorm)
                aggregated_flow_model = MLP(node_dim * 3, fc_dims_aggregated_flow,
                                            params["nonlinearity_total_flow"], use_batchnorm=use_batchnorm)
                node_models.append(ContextualNodeModel(
                    forward_flow_model, frame_flow_model, backward_flow_model, aggregated_flow_model, node_agg_mode, attention_model, node_aggr_sections=node_aggr_sections))

            else:
                aggregated_flow_model = MLP(node_dim * 2, fc_dims_aggregated_flow,
                                            params["nonlinearity_total_flow"], use_batchnorm=use_batchnorm)
                node_models.append(TimeAwareNodeModel(
                    forward_flow_model, backward_flow_model, aggregated_flow_model, node_agg_mode))
        else:
            individual_flow_model = MLP(flow_model_input, fc_dims_directed_flow,
                                        params["nonlinearity_directed_flow"], use_batchnorm=use_batchnorm)
            aggregated_flow_model = MLP(node_dim, fc_dims_aggregated_flow,
                                        params["nonlinearity_total_flow"], use_batchnorm=use_batchnorm)
            node_models.append(UniformAggNodeModel(individual_flow_model,
                               aggregated_flow_model, node_agg_mode))

        if is_recurrent:  # only one model to use at each step
            break

    if is_recurrent:
        assert len(edge_models) == len(node_models) == 1
        if use_separate_edge_model:
            same_frame_edge_model = BasicEdgeModel(MLP(edge_model_input, fc_dims_edge, params["nonlinearity_edge"],
                                                    use_batchnorm=use_batchnorm))
        else:
            same_frame_edge_model = None

        if use_initial_node_model:
            mpn_model = MessagePassingNetworkRecurrent(edge_models[0], node_models[0], steps,
                                                    use_same_frame, same_frame_edge_model)
        else:  # use a node-to-edge MPN
            mpn_model = MessagePassingNetworkRecurrentNodeEdge(edge_models[0], node_models[0], steps,
                                                               use_same_frame, same_frame_edge_model)
    else:
        mpn_model = MessagePassingNetworkNonRecurrent(edge_models, node_models, steps, use_same_frame)

    return initial_edge_model, initial_same_frame_edge_model, initial_node_model, mpn_model, edge_classifier


class GraphTrackerOffline(pl.LightningModule):
    def __init__(self, params: Mapping):
        """ Top level model class holding all components necessary to perform tracking on a graph

        :param initial_edge_model: a torch model processing initial edge attributes
        :param initial_same_frame_edge_model: a torch model processing initial edge attributes for same frame edges
        :param initial_node_model: a torch model processing edge attributes to get initial node features
        :param mpn_model: a message passing model
        :param edge_classifier: a final classification model operating on final edge features
        :param params: params
        """
        super().__init__()
        self.params = params
        (self.initial_edge_model, self.initial_same_frame_edge_model, self.initial_node_model,
         self.mpn_model, self.edge_classifier) = _build_models(params)

        self.loss_type = self.params["loss_type"]
        self.use_same_frame = self.params["use_same_frame"]
        self.pos_weight_multiplier = self.params["pos_weight_multiplier"]

        self.recall_train = Recall(2, threshold=0.5, average='none', multiclass=True)
        self.recall_val = Recall(2, threshold=0.5, average='none', multiclass=True)
        self.precision_train = Precision(2, threshold=0.5, average='none', multiclass=True)
        self.precision_val = Precision(2, threshold=0.5, average='none', multiclass=True)

        self.save_hyperparameters()

    def forward(self, data):
        edge_index, edge_attr, num_nodes = data.edge_index.long(), data.edge_attr, data.num_nodes
        same_frame_edge_index = data.same_frame_edge_index.long() if self.use_same_frame else None
        same_frame_edge_attr = data.same_frame_edge_attr if self.use_same_frame else None

        # Initial Edge embeddings with Null node embeddings
        edge_attr = self.initial_edge_model(edge_attr)
        if self.use_same_frame:
            if self.initial_same_frame_edge_model is not None:
                same_frame_edge_attr = self.initial_same_frame_edge_model(same_frame_edge_attr)
            else:
                same_frame_edge_attr = self.initial_edge_model(same_frame_edge_attr)

        # Initial Node embeddings with Null original embeddings
        x = self.initial_node_model(edge_index, edge_attr, num_nodes,
                                    same_frame_edge_index=same_frame_edge_index, 
                                    same_frame_edge_attr=same_frame_edge_attr, 
                                    device=self.device)
        assert len(x) == num_nodes

        # Message Passing
        x, final_edge_embeddings = self.mpn_model(x, edge_index, edge_attr, num_nodes,
                                                 same_frame_edge_index=same_frame_edge_index,
                                                 same_frame_edge_attr=same_frame_edge_attr)
        return self.edge_classifier(final_edge_embeddings)

    def _compute_bce_loss(self, final_logits, y, pos_weight):
        return F.binary_cross_entropy_with_logits(final_logits.view(-1), y.view(-1),
                                                  pos_weight=pos_weight, reduction="none")

    def _compute_focal_loss(self, final_logits, y, pos_weight, gamma: float):
        return focal_loss_binary(final_logits.view(-1), y.view(-1), pos_weight=pos_weight,
                                 gamma=gamma, reduction="none")

    def compute_loss(self, final_class_logits, y):
        pos_count = y.sum()
        pos_weight = ((len(y) - pos_count) / pos_count) * self.pos_weight_multiplier if pos_count else None

        # TODO: extract focal gamma into a hparam
        if self.loss_type == "bce":
            loss = self._compute_bce_loss(final_class_logits, y, pos_weight)
        elif self.loss_type == "focal":
            loss = self._compute_focal_loss(final_class_logits, y, pos_weight, gamma=2)
        else:
            raise NotImplementedError(f"Unknown {self.loss_type} loss")

        loss = loss.mean()
        return loss, loss

    def training_step(self, batch, batch_idx):
        mode = "train"
        y = batch.y.float()
        final_class_logits = self.forward(batch)
        loss, final_mpn_edge_loss = self.compute_loss(final_class_logits, y)

        self.log(f"{mode}/loss", loss.detach().item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{mode}_loss", loss.detach().item(), on_step=False, on_epoch=True,  prog_bar=True, logger=True)
        self.compute_and_log_metrics(final_class_logits, y, mode=mode)
        return loss

    def validation_step(self, batch, batch_idx):
        mode = "val"
        y = batch.y.float()
        final_class_logits = self.forward(batch)
        loss, final_mpn_edge_loss = self.compute_loss(final_class_logits, y)
        self.log("hp_metric", loss.detach().item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.log_value_pairs((("loss", loss.detach().item()),), prefix=mode)
        self.compute_and_log_metrics(final_class_logits, y, mode=mode)
        return

    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        with torch.no_grad():
            cont_preds = self._cont_preds_from_forward(self.forward(batch))
            labels = batch.y.detach().int() if batch.y is not None else None
            return cont_preds, labels

    def _cont_preds_from_forward(self, final_class_logits):  # returns continuous preds [0, 1]
        with torch.no_grad():
            return nn.Sigmoid()(final_class_logits).detach()  # .round().int()

    def compute_and_log_metrics(self, final_class_logits, y, mode: str):
        preds = self._cont_preds_from_forward(final_class_logits)
        targets = y.detach().int()
        if mode == "train":
            recall_metric = self.recall_train
            precision_metric = self.precision_train
        elif mode == "val":
            recall_metric = self.recall_val
            precision_metric = self.precision_val
        else:
            raise NotImplementedError(f"Unknown mode {mode} for compute_and_log_metrics")
        recall_metric(preds, targets)
        precision_metric(preds, targets)
        self.log_value_pairs((("positive_recall", recall_metric[1]),
                              ("positive_precision", precision_metric[1]),
                              ("negative_recall", recall_metric[0]),
                              ("negative_precision", precision_metric[0]),
                              ), prefix=mode)

    def configure_optimizers(self):
        lr = self.params["lr"]
        wd = self.params["wd"]
        optimizer_type: str = self.params["optimizer_type"].lower()
        if optimizer_type == "radam":
            self.optimizer = RAdam(self.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=wd,
                                             nesterov=True, momentum=0.9)

        scheduler_params: Mapping = self.params["scheduler_params"]
        if "T_0" in scheduler_params:
            self.annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                                            T_0=scheduler_params["T_0"],
                                                                                            T_mult=scheduler_params.get("T_mult", 1),
                                                                                            eta_min=scheduler_params.get("eta_min", 1e-5))
        else:
            self.annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                                  T_max=scheduler_params["T_max"],
                                                                                  eta_min=scheduler_params.get("eta_min", 1e-5))

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.annealing_scheduler,
            }
        }

    def log_value_pairs(self, label_value_pairs: Iterable[Tuple[str, Any]], prefix=""):
        for label, value in label_value_pairs:
            self.log(f"{prefix}/{label}", value, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log(f"{prefix}_{label}", value, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def training_epoch_end(self, outputs):
        for name, params in self.named_parameters():
            try:
                self.logger[0].experiment.add_histogram(name, params, self.current_epoch)
            except:
                os.write(1, f"{name} cannot be added as a histogram\n{traceback.format_exc()}\n\n".encode())
                pass

###################
## Custom #########
###################

def build_param_combos_polarmot(number_edge_features):
    nonlinearity_common = nn.LeakyReLU(inplace=True, negative_slope=0.2)  # nn.Tanh()
    use_batchnorm_list = [False]
    # Initial edge model
    initial_edge_model_input_dim_list = [number_edge_features]#[4]  # fixed
    edge_dim_list = [16]  # [16, 32]
    fc_dims_initial_edge_model_multipliers_list = [(1, 1)]  # (1, 1)
    nonlinearity_initial_edge_list = [nonlinearity_common]

    # Initial node model
    fc_dims_initial_node_model_multipliers_list = [(2, 4, 1)]  # (1, 2, 1), (2,4,1)
    nonlinearity_initial_node_list = [nonlinearity_common]

    # Edge model
    fc_dims_edge_model_multipliers_list = [(4, 1)]  # (6,4,1), (4,1)
    nonlinearity_edge_list = [nonlinearity_common]

    # TimeAware Node model
    fc_dims_directed_flow_model_multipliers_list = [(2, 1)]  # (4,2,1), (2,1); (4,2,2,1) for online
    nonlinearity_directed_flow_list = [nonlinearity_common]
    node_model_agg_list = ["max"]  # ["max", "attention", "attention_classifier", "attention_normalized"]
    # multiplies node_dim, last layer output is always 1 [None, (2,1)]
    fc_dims_node_attention_model_multipliers_list = [None]

    # (4,2,1), (6,4,2,1)  (2,1) online - just an extra MLP
    fc_dims_total_flow_model_multipliers_list = [(4, 2, 1)]
    nonlinearity_total_flow_list = [nonlinearity_common]

    # Edge classification model
    fc_dims_edge_classification_model_multipliers_list = [
        (4, 2, 1,)]  # (2,1) [(0.5, ), None]  # mutliplies edge_dim
    nonlinearity_edge_classification_list = [nonlinearity_common]

    mpn_steps_list = [4]#[args.mpn_steps] -> number of message passing steps default = 4
    is_recurrent_list = [True]
    node_dim_multiplier_list = [2]
    pos_weight_multiplier_list = [0.5]  # 0.5

    use_timeaware_list = [False]
    use_same_frame_list = [False]#[not args.no_sameframe] -> default False
    use_separate_edge_model_list = [False]
    use_initial_node_model_list = [True]
    edge_mlps_count_list = [3]#[args.edge_mlps_count] - > Number of distinct node MLPs default = 3
    node_aggr_sections_list = [3]#[args.node_aggr_sections] -> "Number of distinct sections in Node aggregation default=3

    lr_list = [2e-3]  # 2e-3 for batch=32 on both datasets
    wd_list = [0.005]  # 0.15 for Nu, 0.3 on mini, 0.005 KITTI. 0.02
    loss_type_list = ["focal"]  # ["bce", "focal"]
    optimizer_list = ["radam"]  # ["radam"]  radam seems more unstable

    scheduler_params_list = [
        # {"T_0": 40, "T_mult":1, "eta_min": 1e-4},  # on train val full with 0.15 limits
        {"T_0": 80, "T_mult": 1, "eta_min": 5e-5},  # on full trainval, more hops between local minimals
    ]

    args_mini = False #default=False

    args_continue_training = False #default = False

    online = False #default = False
    
    seeds = [123]
    
    _trainer_params: Dict[str, Union[str, float, int]] = {"max_epochs": 1200, "limit_train_batches": 0.15, "limit_val_batches": 0.15}
    if args_mini:
        _trainer_params["limit_train_batches"] = 1.0

    if args_continue_training:
        if seg_class_id == NuScenesClasses.car:
            if args.mpn_steps == 3:
                _trainer_params = {"max_epochs": 400, "limit_train_batches": 0.15, "limit_val_batches": 0.15,
                                   "pretrained_runs_folder": "gnn_training_ablation_done",
                                   "pretrained_folder": "21-09-16_17:05_aug1_smaller_newaug_offline_0.5xpos_max_sameframe_recurr_edgedim16_steps3_focal_lr0.002_wd0.005_batch64_data21-09-06_car",
                                   "pretrained_ckpt": "val_loss=0.003121-step=51199-epoch=799",
                                   }
    elif online:
        _trainer_params = {"max_epochs": 600, "limit_train_batches": 0.15, "limit_val_batches": 0.15,
                           "pretrained_runs_folder": offline_models[seg_class_id][0],
                           "pretrained_folder": offline_models[seg_class_id][1],
                           "pretrained_ckpt": offline_models[seg_class_id][2],
                           }

    trainer_params_list = [_trainer_params]

    param_combos = list(product(initial_edge_model_input_dim_list,
                            edge_dim_list, fc_dims_initial_edge_model_multipliers_list, nonlinearity_initial_edge_list,
                            fc_dims_initial_node_model_multipliers_list, nonlinearity_initial_node_list,
                            node_model_agg_list, fc_dims_node_attention_model_multipliers_list,
                            fc_dims_edge_model_multipliers_list, nonlinearity_edge_list,
                            fc_dims_directed_flow_model_multipliers_list, nonlinearity_directed_flow_list,
                            fc_dims_total_flow_model_multipliers_list, nonlinearity_total_flow_list,
                            fc_dims_edge_classification_model_multipliers_list, nonlinearity_edge_classification_list,
                            use_batchnorm_list,
                            mpn_steps_list, is_recurrent_list, node_dim_multiplier_list, pos_weight_multiplier_list,
                            use_timeaware_list, use_same_frame_list, use_separate_edge_model_list, use_initial_node_model_list,
                            edge_mlps_count_list,
                            node_aggr_sections_list,
                            lr_list, wd_list,
                            loss_type_list,
                            seeds,
                            optimizer_list,
                            scheduler_params_list,
                            trainer_params_list,
                            ))

    return param_combos
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Dataset
from torch_geometric.nn import global_max_pool, global_mean_pool, summary

import mlsimkit.learn.common.tracking as tracking
import mlsimkit.learn.common.training
from mlsimkit.learn.common.utils import calculate_mean_stddev

from mlsimkit.common.logging import getLogger
from mlsimkit.learn.common.schema.training import LossMetric, PoolingType, GlobalConditionMethod

log = getLogger(__name__)
EPS_TOLERANCE = 1e-8


class StandardScaler:
    def __init__(self, mean=None, std=None, eps=torch.tensor(EPS_TOLERANCE), device=None):
        self.device = device
        if self.device:
            self.mean = mean.to(device)
            self.std = std.to(device)
            self.eps = eps.to(device)
        else:
            self.mean = mean
            self.std = std
            self.eps = eps

    def to(self, device):
        self.device = device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.eps = self.eps.to(device)

    def transform(self, data):
        return (data - self.mean) / (torch.maximum(self.std, self.eps))

    def inverse_transform(self, data):
        return data * torch.maximum(self.std, self.eps) + self.mean


class DataScaler:
    def __init__(
        self,
        data_list,
        normalize_keys=("x", "edge_attr", "y"),
        device=None,
        inplace=False,
        dimensions=None,
        shapes=None,
    ):
        self.normalize_keys = normalize_keys
        self.device = device
        self.inplace = inplace
        self.scaler = self.get_scaler(data_list, dimensions, shapes)

    @property
    def inplace(self):
        return self._inplace

    @inplace.setter
    def inplace(self, value):
        self._inplace = bool(value)

    def to(self, device):
        for key in self.normalize_keys:
            self.scaler[key].to(device)

    def normalize(self, data, key):
        return self.scaler[key].transform(data)

    def unnormalize(self, data, key):
        return self.scaler[key].inverse_transform(data)

    def normalize_all(self, data):
        data_copy = data
        if not self.inplace:
            data_copy = copy.deepcopy(data)
        for key in self.normalize_keys:
            if key in data_copy:
                data_copy[key] = self.normalize(data_copy[key], key)
        return data_copy

    def unnormalize_all(self, data):
        data_copy = data
        if not self.inplace:
            data_copy = copy.deepcopy(data)
        for key in self.normalize_keys:
            if key in data_copy:
                data_copy[key] = self.unnormalize(data_copy[key], key)
        return data_copy

    def get_scaler(self, data_list, dimensions, shapes):
        missing_keys = [k for k in self.normalize_keys if k not in data_list[0]]
        if missing_keys:
            raise RuntimeError(
                f"Cannot normalize data, missing keys in dataset: {missing_keys}, found: {data_list[0]}"
            )

        means, stds = calculate_mean_stddev(
            data_list,
            keys=self.normalize_keys,
            dims=dimensions,
            shapes=shapes,
            device=self.device,
        )
        return {
            k: StandardScaler(means[k], stds[k])  # , device=self.device)
            for k in self.normalize_keys
        }


def calc_loss(pred, inputs, loss_metric=LossMetric.MSE.value):
    actual = inputs.y.reshape(pred.size()[0], -1)
    actual = actual.to(pred.dtype)
    loss_fn = nn.MSELoss(reduction="mean")
    mse_loss = loss_fn(pred, actual)
    if loss_metric == LossMetric.MSE.value:
        return mse_loss
    elif loss_metric == LossMetric.RMSE.value:
        return torch.sqrt(mse_loss)
    else:
        raise RuntimeError(f"Unknown loss metric '{loss_metric}'")


def make_mlp(mlp_input_size, mlp_hidden_size, mlp_output_size, layer_norm=True):
    """Build a multilayer perceptron (MLP)."""
    network = nn.Sequential(
        nn.Linear(mlp_input_size, mlp_hidden_size),
        nn.ReLU(),
        nn.Linear(mlp_hidden_size, mlp_output_size),
    )
    if layer_norm:
        network = nn.Sequential(network, nn.LayerNorm(normalized_shape=mlp_output_size))
    return network


class MeshGraphNet(nn.Module):
    """Construct the encode-process-decode model."""

    def __init__(
        self,
        node_input_size,
        edge_input_size,
        node_encoder_hidden_size,
        edge_encoder_hidden_size,
        node_message_passing_mlp_hidden_size,
        edge_message_passing_mlp_hidden_size,
        node_decoder_hidden_size,
        output_size,
        message_passing_steps,
        dropout_prob,
        graph_level_prediction,
        pooling_type,
        global_condition_size=0,
        global_condition_method=None,
    ):
        super().__init__()
        self._dropout_prob = dropout_prob
        self._graph_level_prediction = graph_level_prediction
        self._pooling_type = pooling_type
        self._global_condition_method = global_condition_method

        self._node_encoder = make_mlp(
            mlp_input_size=node_input_size,
            mlp_hidden_size=node_encoder_hidden_size,
            mlp_output_size=node_message_passing_mlp_hidden_size,
        )
        self._edge_encoder = make_mlp(
            mlp_input_size=edge_input_size,
            mlp_hidden_size=edge_encoder_hidden_size,
            mlp_output_size=edge_message_passing_mlp_hidden_size,
        )

        self._processor_list = nn.ModuleList()
        for _ in range(message_passing_steps):
            self._processor_list.append(
                GraphNetBlock(node_message_passing_mlp_hidden_size, edge_message_passing_mlp_hidden_size)
            )

        if self._graph_level_prediction:
            self._decoder = nn.Linear(
                node_message_passing_mlp_hidden_size + global_condition_size, output_size
            )
        else:
            self._decoder = make_mlp(
                mlp_input_size=node_message_passing_mlp_hidden_size + global_condition_size,
                mlp_hidden_size=node_decoder_hidden_size + global_condition_size,
                mlp_output_size=output_size,
                layer_norm=False,
            )

    def forward(self, data):
        node_features, edge_features, edge_indices = (
            data.x,
            data.edge_attr,
            data.edge_index,
        )

        node_embeddings = self._node_encoder(node_features)
        edge_embeddings = self._edge_encoder(edge_features)

        for processor in self._processor_list:
            node_embeddings, edge_embeddings = processor(node_embeddings, edge_embeddings, edge_indices)

        if hasattr(data, "global_condition") and self._global_condition_method == GlobalConditionMethod.MODEL:
            if data.batch is not None:
                unique_batches, batch_counts = torch.unique(data.batch, return_counts=True)
                # Repeat the global_condition rows according to batch_counts
                global_condition = data.global_condition.repeat_interleave(batch_counts, dim=0)
            else:
                global_condition = data.global_condition.repeat_interleave(node_embeddings.shape[0], dim=0)

            node_embeddings = torch.cat((node_embeddings, global_condition), dim=-1).type(torch.float)

        if self._graph_level_prediction:
            if self._pooling_type == PoolingType.MEAN:
                graph_embeddings = global_mean_pool(node_embeddings, data.batch)
            elif self._pooling_type == PoolingType.MAX:
                graph_embeddings = global_max_pool(node_embeddings, data.batch)
            graph_embeddings = F.dropout(graph_embeddings, p=self._dropout_prob, training=self.training)
            output_embeddings = self._decoder(graph_embeddings)
        else:
            output_embeddings = self._decoder(node_embeddings)

        return output_embeddings

    def summary(self, data):
        return summary(self.cpu(), data.cpu())


class GraphNetBlock(nn.Module):
    """Construct a GraphNet block for message passing."""

    def __init__(self, node_message_passing_mlp_hidden_size, edge_message_passing_mlp_hidden_size):
        super().__init__()
        edge_mlp_input_size = node_message_passing_mlp_hidden_size * 2 + edge_message_passing_mlp_hidden_size
        node_mlp_input_size = node_message_passing_mlp_hidden_size + edge_message_passing_mlp_hidden_size
        self._edge_mlp = make_mlp(
            mlp_input_size=edge_mlp_input_size,
            mlp_hidden_size=edge_message_passing_mlp_hidden_size,
            mlp_output_size=edge_message_passing_mlp_hidden_size,
        )
        self._node_mlp = make_mlp(
            mlp_input_size=node_mlp_input_size,
            mlp_hidden_size=node_message_passing_mlp_hidden_size,
            mlp_output_size=node_message_passing_mlp_hidden_size,
        )

    def _update_edge_features(self, node_features, edge_features, edge_indices):
        sender_ids, receiver_ids = edge_indices
        sender_features = node_features[sender_ids]
        receiver_features = node_features[receiver_ids]
        feature_list = [sender_features, receiver_features, edge_features]
        features = torch.cat(feature_list, dim=-1)
        return self._edge_mlp(features)

    def _aggregate_edge_features(self, data, index, dim_size):
        index = index.unsqueeze(-1).expand(data.shape)
        base = torch.zeros((dim_size, data.shape[1]), dtype=data.dtype, device=data.device)
        return base.scatter_add_(0, index, data)

    def _update_node_features(self, node_features, edge_features, edge_indices):
        _, receiver_ids = edge_indices
        num_nodes = node_features.shape[0]
        aggregated_edge_features = self._aggregate_edge_features(edge_features, receiver_ids, num_nodes)
        feature_list = [node_features, aggregated_edge_features]
        features = torch.cat(feature_list, dim=-1)
        return self._node_mlp(features)

    def forward(self, node_features, edge_features, edge_indices):
        updated_edge_features = self._update_edge_features(node_features, edge_features, edge_indices)
        updated_node_features = self._update_node_features(node_features, updated_edge_features, edge_indices)

        output_edge_features = edge_features + updated_edge_features
        output_node_features = node_features + updated_node_features

        return output_node_features, output_edge_features


class MeshGraphNetDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, idx):
        dataset = self.dataset[idx]
        dataset.x[:, 0:3] = dataset.x[:, 0:3] + 0.01 * np.random.normal(0, 0.1, size=(1, 3))
        return dataset

    def __len__(self):
        return len(self.dataset)


def load_model(model_path):
    model_dict = torch.load(model_path)
    model_config = model_dict["config"]
    if hasattr(model_config, "dropout_prob"):
        dropout_prob = model_config.dropout_prob
    else:
        dropout_prob = None
    if hasattr(model_config, "pooling_type"):
        pooling_type = model_config.pooling_type
    else:
        pooling_type = None
    model = MeshGraphNet(
        node_input_size=model_dict["node_input_size"],
        edge_input_size=model_dict["edge_input_size"],
        node_encoder_hidden_size=model_config.node_encoder_hidden_size,
        edge_encoder_hidden_size=model_config.edge_encoder_hidden_size,
        node_message_passing_mlp_hidden_size=model_config.node_message_passing_mlp_hidden_size,
        edge_message_passing_mlp_hidden_size=model_config.edge_message_passing_mlp_hidden_size,
        node_decoder_hidden_size=model_config.node_decoder_hidden_size,
        output_size=model_dict["num_classes"],
        message_passing_steps=model_config.message_passing_steps,
        dropout_prob=dropout_prob,
        graph_level_prediction=model_dict["graph_level_prediction"],
        pooling_type=pooling_type,
        global_condition_size=model_dict["global_condition_size"],
        global_condition_method=model_dict["global_condition_method"],
    )
    model.load_state_dict(model_dict["model_state_dict"])
    return model, model_dict


class ModelIO:
    def __init__(
        self,
        config,
        data_scaler,
        graph_shape,
        accelerator,
        graph_level_prediction=True,
        metadata=None,
        global_condition_size=0,
    ):
        self.config = config
        self.data_scaler = data_scaler
        self.node_input_size = graph_shape[0]
        self.edge_input_size = graph_shape[1]
        self.num_classes = graph_shape[2]
        self.accelerator = accelerator
        self.graph_level_prediction = graph_level_prediction
        self.global_condition_size = global_condition_size
        if hasattr(config, "dropout_prob"):
            self.dropout_prob = config.dropout_prob
        else:
            self.dropout_prob = None
        if hasattr(config, "pooling_type"):
            self.pooling_type = config.pooling_type
        else:
            self.pooling_type = None
        if hasattr(config, "global_condition_method"):
            self.global_condition_method = config.global_condition_method
        else:
            self.global_condition_method = None
        self.metadata = metadata or {}

    def new(self):
        return MeshGraphNet(
            node_input_size=self.node_input_size,
            edge_input_size=self.edge_input_size,
            node_encoder_hidden_size=self.config.node_encoder_hidden_size,
            edge_encoder_hidden_size=self.config.edge_encoder_hidden_size,
            node_message_passing_mlp_hidden_size=self.config.node_message_passing_mlp_hidden_size,
            edge_message_passing_mlp_hidden_size=self.config.edge_message_passing_mlp_hidden_size,
            node_decoder_hidden_size=self.config.node_decoder_hidden_size,
            output_size=self.num_classes,
            message_passing_steps=self.config.message_passing_steps,
            dropout_prob=self.dropout_prob,
            graph_level_prediction=self.graph_level_prediction,
            pooling_type=self.pooling_type,
            global_condition_size=self.global_condition_size,
            global_condition_method=self.global_condition_method,
        )

    def load(self, config):
        # pass through to built-checkpoint loader
        return mlsimkit.learn.common.training.load_checkpoint_model(self, config)

    def save(self, model, model_path, train_loss, validation_loss, optimizer, epoch):
        self.accelerator.save(
            {
                "model_state_dict": self.accelerator.unwrap_model(model).state_dict(),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "optimizer_state_dict": optimizer.state_dict(),
                "config": self.config,
                "data_scaler": self.data_scaler,
                "epoch": epoch,
                "node_input_size": self.node_input_size,
                "edge_input_size": self.edge_input_size,
                "num_classes": self.num_classes,
                "graph_level_prediction": self.graph_level_prediction,
                "metadata": self.metadata,
                "global_condition_size": self.global_condition_size,
                "global_condition_method": self.global_condition_method,
            },
            model_path,
        )
        log.info(f"Model saved to '{model_path}'")
        tracking.log_artifact(model_path)

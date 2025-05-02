# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import mlsimkit.learn.common.tracking as tracking
import numpy as np
import torch
from mlsimkit.common.logging import getLogger
from mlsimkit.learn.common.training import load_checkpoint_model
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from .schema.autoencoder import ConvAutoencoderSettings

log = getLogger(__name__)


class ConvAutoencoder(nn.Module):
    def __init__(self, settings: ConvAutoencoderSettings):
        super(ConvAutoencoder, self).__init__()
        self.input_channels = settings.input_channels
        self.start_out_channel = settings.start_out_channel
        self.div_rate = settings.div_rate
        self.add_sigmoid = settings.add_sigmoid
        self.dropout_prob = settings.dropout_prob
        self.image_size = settings.image_size

        if self.image_size.width % self.image_size.height != 0:
            raise ValueError(
                "image_size.width = {}  must be a multiple of image_size.height = {}".format(
                    self.image_size.width, self.image_size.height
                )
            )
        if self.start_out_channel % (self.div_rate * 2) != 0:
            raise ValueError("start_out_channel must be a multiple of div_rate * 2")
        if self.image_size.width % (self.div_rate * 4) != 0:
            raise ValueError("image_size.width must be a multiple of div_rate * 4")
        if self.image_size.height % (self.div_rate * 4) != 0:
            raise ValueError("image_size.height must be a multiple of div_rate * 4")

        self.unflatten_dim = [
            int(self.image_size.height / int(self.start_out_channel / (self.div_rate * 2))),
            int(self.image_size.width / int(self.start_out_channel / (self.div_rate * 2))),
        ]

        self.encoder_output_size = int(
            self.unflatten_dim[0] * self.unflatten_dim[1] * int(self.start_out_channel / (self.div_rate * 4))
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=self.start_out_channel,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(int(self.start_out_channel / 1)),
            nn.MaxPool2d(2),
            nn.Dropout(self.dropout_prob),
            nn.Conv2d(
                in_channels=self.start_out_channel,
                out_channels=int(self.start_out_channel / self.div_rate),
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(int(self.start_out_channel / self.div_rate)),
            nn.MaxPool2d(2),
            nn.Dropout(self.dropout_prob),
            nn.Conv2d(
                in_channels=int(self.start_out_channel / self.div_rate),
                out_channels=int(self.start_out_channel / (self.div_rate * 2)),
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(int(self.start_out_channel / (self.div_rate * 2))),
            nn.MaxPool2d(2),
            nn.Conv2d(
                in_channels=int(self.start_out_channel / (self.div_rate * 2)),
                out_channels=int(self.start_out_channel / (self.div_rate * 3)),
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(int(self.start_out_channel / (self.div_rate * 3))),
            nn.MaxPool2d(2),
            nn.Conv2d(
                in_channels=int(self.start_out_channel / (self.div_rate * 3)),
                out_channels=int(self.start_out_channel / (self.div_rate * 4)),
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(int(self.start_out_channel / (self.div_rate * 4))),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        decoder_layers = [
            nn.Unflatten(
                dim=1,
                unflattened_size=(
                    int(self.start_out_channel / (self.div_rate * 4)),
                    self.unflatten_dim[0],
                    self.unflatten_dim[1],
                ),
            ),
            nn.Conv2d(
                int(self.start_out_channel / (self.div_rate * 4)),
                int(self.start_out_channel / (self.div_rate * 3)),
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(int(self.start_out_channel / (self.div_rate * 3))),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(
                int(self.start_out_channel / (self.div_rate * 3)),
                int(self.start_out_channel / (self.div_rate * 2)),
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(int(self.start_out_channel / (self.div_rate * 2))),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(
                int(self.start_out_channel / (self.div_rate * 2)),
                int(self.start_out_channel / self.div_rate),
                3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(int(self.start_out_channel / self.div_rate)),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Dropout(self.dropout_prob),
            nn.Conv2d(
                int(self.start_out_channel / self.div_rate),
                int(self.start_out_channel / self.div_rate),
                3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(int(self.start_out_channel / self.div_rate)),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Dropout(self.dropout_prob),
            nn.Conv2d(
                int(self.start_out_channel / self.div_rate),
                int(self.start_out_channel / 1),
                3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(int(self.start_out_channel / 1)),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(int(self.start_out_channel / 1), self.input_channels, 3, padding=1),
        ]

        if self.add_sigmoid:
            decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)
        self.embedding_size = int(self.start_out_channel / (self.div_rate * 4)) * 16
        log.debug(f"ConvAutoencoder embedding size: {self.embedding_size}")

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def summary(self, data):
        return summary(self, data.shape)


class DataScaler:
    def __init__(
        self,
    ):
        pass

    def to(self, device):
        pass

    def normalize_all(self, data):
        return data / 255

    def unnormalize_all(self, data):
        if isinstance(data, np.ndarray):
            return np.clip(data * 255, 0, 255).astype(int)
        elif isinstance(data, torch.Tensor):
            return torch.clip(data * 255, 0, 255).to(torch.int)
        else:
            raise Exception(f"Data type not supported by scaler: {type(data)}")


def get_loss(recon_x, x):
    loss = F.mse_loss(x, recon_x, reduction="none")
    loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
    return loss


class ModelIO:
    def __init__(self, config, data_scaler, accelerator):
        self.config = config
        self.data_scaler = data_scaler
        self.accelerator = accelerator

    def new(self):
        return ConvAutoencoder(self.config.autoencoder_settings)

    def load(self, config, device):
        # pass through to built-checpoint loader
        return load_checkpoint_model(self, config, device)

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
            },
            model_path,
        )
        log.info(f"Model saved to '{model_path}'")
        tracking.log_artifact(model_path)

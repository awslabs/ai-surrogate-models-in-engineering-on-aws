# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Callable, Optional
from pathlib import Path

import numpy as np
import torch
import torch_geometric
import torch.utils.data

from mlsimkit.learn.manifest.manifest import read_manifest_file, resolve_file_path


class SlicesDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path: Path, transform: Optional[Callable] = None):
        """
        Args:
            manifest_path (Path): Path to the JSON lines file (manifest).
            data_key (str, optional): Key in the manifest dictionary to access the data URI.
            transform (Callable, optional): Optional transform to be applied on a sample.

        NOTE: assumes that all runs have the same number of frames, uses the first row as the
              number of frames
        """
        # TODO: validate frame count in manifest
        slices_frame = read_manifest_file(manifest_path)
        self.rows = slices_frame
        # when slices data is not present, default to empty/zero frames for predict-only
        self.slices = slices_frame.get("slices_data_uri", [])
        self.nframes = int(slices_frame.get("slices_data_frame_count", [0])[0])
        self.transform = transform

    def has_slices(self):
        return len(self.slices) > 0

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get the slices data for the ith row
        """
        if not self.has_slices():
            raise RuntimeError("Dataset is missing slices")

        if torch.is_tensor(idx):
            idx = idx.tolist()
            if max(idx) >= len(self.slices):
                raise IndexError()
        elif idx >= len(self.slices):
            # default sampler is [0, len()] so we need to stop TODO: why doesnt navier hit this?
            raise IndexError()

        if isinstance(idx, (slice, list)):
            data = np.array([np.load(resolve_file_path(self.slices[i])) for i in idx]).astype(np.float32)
        else:
            data = np.load(resolve_file_path(self.slices[idx])).astype(np.float32)

        data = self.transform(data) if self.transform else torch.Tensor(data.copy())  # TODO: why copy?
        return data


def add_noise_and_flip(data, flip_probability=0.5, noise_factor=0.01):
    if torch.rand(1) < flip_probability:
        data = np.fliplr(data)
    if torch.rand(1) < flip_probability:
        data = np.flipud(data)

    if noise_factor > 0.2 or noise_factor < 0:
        raise ValueError(
            f"noise_factor set at {noise_factor} which is outside the recommended range of 0 to 0.2."
        )

    data = torch.Tensor(data.copy())
    min_value = torch.min(data)
    max_value = torch.max(data)
    data_n = data + (noise_factor * (max_value - min_value)) * torch.randn(*data.shape)
    data_n = torch.clip(data_n, min_value, max_value)
    return data_n


class GraphDataset(torch_geometric.data.Dataset):
    def __init__(self, manifest_path, key, device="cuda"):
        super(GraphDataset, self).__init__(root=None, transform=None, pre_transform=None)
        self.device = device
        manifest = read_manifest_file(manifest_path)
        self.items = manifest[key]

    def len(self):
        return len(self.items)

    def name(self, idx):
        return resolve_file_path(self.items[idx])

    def get(self, idx):
        filepath = resolve_file_path(self.items[idx])
        data = torch.load(filepath)
        if self.device:
            data.x = data.x.to(self.device)
            if "y" in data:
                data.y = data.y.to(self.device)
        if "y" in data:
            data.y = data.y.unsqueeze(
                0
            ).detach()  # TODO: why detach is needed to avoid "deecopy" errors with non-leaf nodes
        return data

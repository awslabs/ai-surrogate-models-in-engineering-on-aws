# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_geometric.data

import pandas as pd

from mlsimkit.learn.manifest.manifest import read_manifest_file, resolve_file_path


class SurfaceDataset(torch_geometric.data.Dataset):
    def __init__(self, manifest, device="cuda"):
        super(SurfaceDataset, self).__init__(root=None, transform=None, pre_transform=None)
        self.device = device
        if isinstance(manifest, pd.DataFrame):
            self.manifest = manifest
        else:  # assume manifest is a filepath, will fail otherwise
            self.manifest = read_manifest_file(manifest)

    def len(self):
        return len(self.manifest)

    def run_id(self, idx):
        return self.manifest["id"][idx]

    def surface_variables(self):
        return self.get(0).y_variables

    def ptfile(self, idx):
        return resolve_file_path(self.manifest["preprocessed_files"][idx])

    def has_data_files(self):
        return "data_files" in self.manifest

    def has_geometry_files(self):
        return "geometry_files" in self.manifest

    def has_predictions(self):
        return "predicted_file" in self.manifest

    def data_files(self, idx, missing_ok=True):
        if "data_files" in self.manifest:
            return list(map(resolve_file_path, self.manifest["data_files"][idx]))
        elif missing_ok:
            return None
        else:
            raise RuntimeError("Data files not found in manifest")

    def geometry_files(self, idx, missing_ok=True):
        if "geometry_files" in self.manifest:
            return list(map(resolve_file_path, self.manifest["geometry_files"][idx]))
        elif missing_ok:
            return None
        else:
            raise RuntimeError("Geometry files not found in manifest")

    def predicted_file(self, idx, missing_ok=True):
        if "predicted_file" in self.manifest:
            return self.manifest["predicted_file"][idx]
        elif missing_ok:
            return None
        else:
            raise RuntimeError("Predicted files not found in manifest")

    def set_predicted_file(self, idx, filepath):
        if "predicted_file" not in self.manifest.columns:
            self.manifest["predicted_file"] = [None] * len(self.manifest)
        self.manifest.loc[idx, "predicted_file"] = filepath

    def get(self, idx):
        return torch.load(self.ptfile(idx)).to(self.device)

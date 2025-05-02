# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_geometric.data

import pandas as pd

from mlsimkit.learn.manifest.manifest import read_manifest_file, resolve_file_path
from .schema.training import GlobalConditionMethod


class KPIDataset(torch_geometric.data.Dataset):
    def __init__(self, manifest, output_kpi_indices=None, global_condition_method=None, device="cuda"):
        super(KPIDataset, self).__init__(root=None, transform=None, pre_transform=None)
        self.device = device
        self.global_condition_method = global_condition_method
        if isinstance(manifest, pd.DataFrame):
            self.manifest = manifest
        else:  # assume manifest is a filepath, will fail otherwise
            self.manifest = read_manifest_file(manifest)
        self.kpi_indices = None

        if output_kpi_indices:
            kpi_indices = str(output_kpi_indices).replace(" ", "")
            self.kpi_indices = [int(s.strip()) for s in kpi_indices.split(",")]
        elif "kpi" in self.manifest.columns:
            # grab first row, assumes all rows in manifest are same length i.e, a valid manifest
            self.kpi_indices = list(range(len(self.manifest.iloc[0]["kpi"])))

    def len(self):
        return len(self.manifest)

    def ptfile(self, idx):
        return resolve_file_path(self.manifest["preprocessed_files"][idx])

    def get(self, idx):
        tmp_data = torch.load(self.ptfile(idx)).to(self.device)
        if "y" in tmp_data:
            tmp_data.y = tmp_data.y[:, self.kpi_indices]
        if (
            "global_condition" in tmp_data
            and self.global_condition_method == GlobalConditionMethod.NODE_FEATURES
        ):
            global_condition = tmp_data.global_condition[0].unsqueeze(0).repeat(tmp_data.x.shape[0], 1)
            tmp_data.x = torch.cat((tmp_data.x[:, :-3], global_condition, tmp_data.x[:, -3:]), dim=-1).type(
                torch.float
            )

        return tmp_data

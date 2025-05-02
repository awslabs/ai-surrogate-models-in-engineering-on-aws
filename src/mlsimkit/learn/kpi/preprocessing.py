# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
from multiprocessing import cpu_count
from pathlib import Path
from urllib.parse import urljoin

import torch
import pandas as pd

from .schema.preprocessing import PreprocessingSettings

import mlsimkit.learn.common.tracking as tracking

from mlsimkit.common.logging import getLogger
from mlsimkit.learn.manifest.manifest import (
    get_base_path,
    get_path_list,
    get_array_list,
    read_manifest_file,
    make_working_manifest,
    write_manifest_file,
)

import mlsimkit.learn.common.mesh as mesh

log = getLogger(__name__)


def get_paths(config, manifest, key):
    base_relative_path = get_base_path(config.manifest_base_relative_path, config.manifest_path)
    return get_path_list(manifest, key, base_relative_path)


def add_preprocessed_files(manifest, files):
    if not files:
        return manifest

    max_id = 0 if manifest.empty else int(manifest["id"].max())
    files = [urljoin("file://", Path(f).resolve().as_posix()) for f in files]
    new_rows = pd.DataFrame({"preprocessed_files": files, "id": range(max_id + 1, max_id + len(files) + 1)})
    return pd.concat([manifest, new_rows], ignore_index=True)


def process_mesh_files(config, manifest, use_labels=True, use_global_conditions=True):
    output_dir = Path(config.output_dir)
    downsample_perc = config.downsample_remaining_perc
    num_processes = config.num_processes or (cpu_count() - 1)

    log.info(f"Preprocessing mesh files (num_processes={num_processes})")

    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_paths = get_paths(config, manifest, "geometry_files")

    if not mesh_paths:
        return manifest

    labels = get_array_list(manifest, "kpi") if use_labels else None
    global_conditions = get_array_list(manifest, "simulation_condition") if use_global_conditions else None

    # keep a map of the original files in case of conversions and downsampling
    manifest = manifest.rename(columns={"geometry_files": "original_geometry_files"})
    manifest["geometry_files"] = ""  # prepare to set while iterating, makes the column a dtype
    manifest["id"] = int  # set column type to integer

    # now process the data lazily by using a generator, actual processing happens during enumerate after
    processed_data = mesh.construct_processed_data(
        mesh_paths, labels, global_conditions, output_dir, downsample_perc, num_processes
    )

    # we process each run during iteration, this allows us to release resources when we are
    # done processing each run. This is important for multi-processing so torch can release
    # file descriptors used for inter-process comms.
    for i, datum in enumerate(processed_data):
        filepath = output_dir / f"preprocessed_run_{i:0>5}.pt"
        torch.save(datum, filepath)
        log.debug("Run %i preprocessed, written to '%s'", i, filepath)
        manifest.loc[i, ("id", "preprocessed_files")] = (
            i,
            urljoin("file://", filepath.resolve().as_posix()),
        )
        manifest.at[i, "geometry_files"] = datum.mesh_path

    log.info(f"Saved output files in {output_dir}")

    return manifest


def run_preprocess(config: PreprocessingSettings, project_root: Path):
    log.info(f"Preprocessing configuration: {config}")
    os.makedirs(config.output_dir, exist_ok=True)

    t_start = time.time()

    # Use a working manifest for writing, avoid changing user manifests
    working_manifest_path = make_working_manifest(config.manifest_path, project_root)
    manifest = read_manifest_file(working_manifest_path)
    manifest = process_mesh_files(
        config,
        manifest,
        use_labels=("kpi" in manifest.columns),
        use_global_conditions=("simulation_condition" in manifest.columns),
    )
    write_manifest_file(manifest, working_manifest_path)

    t_end = time.time()
    log.info(f"Total preprocessing time: {(t_end - t_start):.3f} seconds")
    tracking.log_metric("total_processing_time", t_end - t_start)
    return working_manifest_path

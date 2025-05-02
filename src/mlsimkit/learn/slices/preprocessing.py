# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from urllib.parse import urljoin

import cv2
import numpy as np

import mlsimkit.learn.common.tracking as tracking

from mlsimkit.common.logging import getLogger
from mlsimkit.learn.manifest.manifest import (
    read_manifest_file,
    resolve_file_path,
    write_manifest_file,
    make_working_manifest,
)

from .schema.preprocessing import ConvertImageSettings

log = getLogger(__name__)


def load_image(filename, grayscale=False, resolution=(128, 128)):
    if not Path(filename).exists():
        raise ValueError(f"Image file '{filename}' does not exist")

    img = cv2.imread(filename, 1)
    img_res = cv2.resize(img, resolution, interpolation=cv2.INTER_AREA)
    if grayscale:
        img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        img_res = np.expand_dims(img_res, axis=0)
    else:
        img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        img_res = np.transpose(img_res, (2, 0, 1))
    return img_res


def load_image_files(files, grayscale=False, resolution=(128, 128)):
    log.debug("Loading image files n=%s", len(files))
    data = []
    for file in files:
        img = load_image(file, grayscale, resolution)
        log.debug("Loaded image shape=%s: %s", img.shape, file)
        data.append(img)
    data = np.concatenate(data, axis=0)
    log.debug("Image data shape: %s", data.shape)
    return data


def run_preprocessing(
    manifest_path: Path, project_root: Path, base_relative_path: Path, settings: ConvertImageSettings
):
    log.info("Preprocessing manifest '%s'", manifest_path)

    # Use a working manifest for writing, avoid changing user manifests
    working_manifest_path = make_working_manifest(manifest_path, project_root)
    manifest = read_manifest_file(working_manifest_path)

    local_slice_dir = project_root / "slices"
    local_slice_dir.mkdir(parents=True, exist_ok=True)

    for i, row in manifest.iterrows():
        # ith row is used as the id, assuming preprocessing is using a shared manifest. To parallelize, we
        # can keep the monolithic input manifest but give each preprocess command start-stop indices
        image_files = [
            resolve_file_path(f, default_dir=base_relative_path) for f in row.get("slices_uri", [])
        ]
        if image_files:
            image_data = load_image_files(image_files, **settings.dict())
            image_data_file = local_slice_dir / f"slice-group-{i}.npy"
            np.save(image_data_file, image_data)
            log.info("Image data written to '%s'", image_data_file)

            # update manifest with new data, use absolute paths
            manifest.loc[i, ("id", "slices_data_uri", "slices_data_frame_count")] = (
                i,
                urljoin("file://", image_data_file.resolve().as_posix()),
                len(image_files),
            )

    write_manifest_file(manifest, working_manifest_path)
    tracking.log_artifact(working_manifest_path, "manifest")
    return working_manifest_path

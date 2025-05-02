# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import importlib
import json
import logging
import shutil
import re
from pathlib import Path
from typing import Optional, Dict
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd

from .schema import RelativePathBase, SplitSettings

log = logging.getLogger(__name__)


def get_base_path(base_enum: RelativePathBase, manifest_path):
    if isinstance(base_enum, str):
        base_enum = RelativePathBase[base_enum]
    if base_enum == RelativePathBase.PackageRoot:
        return importlib.resources.files("mlsimkit")
    elif base_enum == RelativePathBase.ManifestRoot:
        return Path(manifest_path).parent
    return None


def resolve_file_path(file_path, default_dir=None, missing_ok=True):
    """Resolve a file path to an absolute path or URI."""
    parsed_url = urlparse(file_path)
    if parsed_url.scheme:  # Check if the path is a URI
        return parsed_url.path
    elif Path(file_path).is_absolute():  # Check if the path is an absolute path
        return parsed_url.path
    elif default_dir:
        # Assume relative path and resolve it against the default directory
        return str(default_dir / file_path)

    return file_path if missing_ok else FileNotFoundError(file_path)


def generate_manifest_entries(run_folders, data_files, file_lists, skip_on_error):
    """Generate manifest entries from a list of simulation "run" folders.

    Args:
        run_folders (list): A list of run folder paths.
        data_files (list): A list of DataFile objects representing data files.
        file_lists (list): A list of FileList objects representing file lists.
        skip_on_error (bool): If True, skip over a run folder on error and continue.

    Yields:
        dict: A manifest entry dictionary for each run folder.
    """
    for run_folder in run_folders:
        try:
            yield create_manifest_entry(run_folder, data_files, file_lists)
        except Exception as e:
            if skip_on_error:
                log.error(f"Skipping '{run_folder}' on error: {str(e)}")
            else:
                raise e


def create_manifest_entry(run_folder, data_files, file_lists):
    """Create one manifest entry for a simulation "run" folder.

    Args:
        run_folder (str): The path to the run folder.
        data_files (list): A list of DataFile objects representing data files.
        file_lists (list): A list of FileList objects representing file lists.

    Returns:
        dict: A manifest entry dictionary for the run folder.

    Raises:
        RuntimeError: If there are issues with the data files or file lists.
    """
    manifest_entry = {}
    run_folder_path = Path(run_folder)

    # Find files matching the input glob patterns
    for p in file_lists:
        if p.file_glob:
            files = [urljoin("file://", str(file.resolve())) for file in run_folder_path.glob(p.file_glob)]
            manifest_entry[f"{p.name}"] = sorted(files)
            if len(files) == 0:
                raise RuntimeError(f"Run '{run_folder_path}' missing file(s) '{p}'")

    #
    # Find data files matching their glob pattern. Only one data file is supported per key (-d argument).
    #
    found_data_files = []
    for param in data_files:
        if param.file_glob is not None:
            files = list(Path(run_folder_path).glob(param.file_glob))
        elif param.file_regex is not None:
            regex = re.compile(param.file_regex)
            files = [f for f in run_folder_path.iterdir() if regex.match(f.name)]
        if len(files) == 0:
            raise RuntimeError(f"Run '{run_folder_path}' missing a data file '{param}'")
        if len(files) > 1:
            raise RuntimeError(f"Run '{run_folder_path}' has more than one data file '{param}': {files}")
        found_data_files.append((param, files[0]))  # remember this actual file

    #
    # Read each data file and extract the values for each parameter set
    #
    for param, file_path in found_data_files:
        try:
            with file_path.open("r") as csvfile:
                rows = list(csv.reader(csvfile, delimiter=param.delimiter))
                if len(rows) != 2:
                    raise RuntimeError(
                        f"Data file '{file_path}' has {len(rows)} rows, expects two (header, values)"
                    )
                keys = [k.strip().lower() for k in rows[0]]
                values = {keys[i]: float(value) for i, value in enumerate(rows[1])}
                extracted_values = [values[key.lower()] for key in param.columns]
                manifest_entry[param.name] = extracted_values
        except Exception as e:
            raise RuntimeError(f"Failed to parse '{file_path}' for param '{param.name}', error: {str(e)}")

    if not manifest_entry:
        raise RuntimeError(f"Failed to create entry for '{file_path}', error: no data found")

    return manifest_entry


def write_manifest_files(manifest_entries, manifest_file, run_folders):
    """
    Write the manifest and description files using the provided manifest entries iterator.
    """
    count = 0
    for entry in manifest_entries:
        manifest_file.write(json.dumps(entry) + "\n")
        count += 1
    log.info("Manifest written to %s (%s rows)", manifest_file.name, count)


def read_manifest_file(manifest_filepath):
    """
    Read a manifest file into a Dataframe where each line is a record
    """
    return pd.read_json(manifest_filepath, lines=True, orient="records")


def make_manifest(geometry_files):
    return pd.DataFrame({"geometry_files": geometry_files})


def write_manifest_file(records: pd.DataFrame, filename: str) -> None:
    """Write manifest records as a JSON lines file.

    Args:
        records (pd.DataFrame): A pandas DataFrame containing the records to be written.
        filename (str): The path to the output JSON lines file.
    """
    # Note: pandas escapes '/' so manually write to json for now rather
    # than using built-in `records.to_json(filename, orient='records', lines=True)`
    with open(filename, "w", encoding="utf-8") as f:
        for record in records.to_dict("records"):
            json.dump(record, f)
            f.write("\n")
    log.info(f"Manifest '{filename}' written ({len(records)} records)")


def make_working_manifest(manifest_path, output_dir):
    working_manifest_path = Path(output_dir) / f"{Path(manifest_path).stem}-copy.manifest"
    if manifest_path == working_manifest_path:
        raise RuntimeError(
            f"Cannot copy to working manifest, input manifest is the same file: {manifest_path}."
            f"Move your manifest outside the output directory or rename."
        )
    working_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(manifest_path, working_manifest_path)
    return working_manifest_path


def get_path_list(manifest, key, default_dir=None):
    return list(list((resolve_file_path(f, default_dir) for f in x)) for x in manifest[key].values)


def get_array_list(manifest, key):
    return np.array([np.array(x) for x in manifest[key].values])


def split_manifest(
    manifest_file: str, settings: SplitSettings, output_dir: str = None
) -> Dict[str, Optional[str]]:
    """Split a JSON lines manifest file into train, validation, and test sets.

    Args:
        manifest_file (str): The path to the input JSON lines manifest file.
        settings (SplitSettings): The settings for splitting the data.
        output_dir (str, optional): The path to the output directory for the split files. Defaults to the same directory as the input manifest file.

    Returns:
        Dict[str, Optional[str]]: A dictionary containing the file paths for the train, validation, and test sets. If a set is empty, its value in the dictionary will be `None`.

    The function handles the following cases:
    - If `settings.test_size` is 1.0 (100%), the entire dataset is assigned to the test set, and the train and validation sets are set to `None`.
    - If `settings.test_size` is 0.0, the dataset is split into train and validation sets according to `settings.train_size`, and the test set is set to `None`.
    - If `settings.test_size` is between 0.0 and 1.0 (exclusive), the dataset is split into train, validation, and test sets according to the provided percentages.

    The function writes the split manifest files to the specified `output_dir` or the same directory as the input manifest file if `output_dir` is not provided.
    The filenames of the split manifest files are derived from the input manifest file's name with "-train", "-valid", and "-test" suffixes.
    If a split set is empty (`None`), the corresponding value in the returned dictionary will be `None`.
    """

    # lazy import due to time to first import sklearn (avoid file-level so --help is quick)
    from sklearn.model_selection import train_test_split

    # split randomly
    records = read_manifest_file(manifest_file)

    if settings.test_size == 1.0:
        # Handle the case where test_size is 1.0 and train_size and valid_size are 0.0
        data_train = None
        data_valid = None
        data_test = records
    elif settings.test_size <= 0.0:
        data_train, data_valid = train_test_split(
            records, train_size=settings.train_size, random_state=settings.random_seed
        )
        data_test = None
    else:
        data_train, data_remaining = train_test_split(
            records, train_size=settings.train_size, random_state=settings.random_seed
        )
        data_valid, data_test = train_test_split(
            data_remaining,
            test_size=settings.test_size / (1 - settings.train_size),
            random_state=settings.random_seed,
        )

    # write to same folder as manifest if not overridden
    manifest_file = Path(manifest_file)
    output_dir = Path(output_dir or manifest_file.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / "train.manifest" if data_train is not None else None
    valid_file = output_dir / "validate.manifest" if data_valid is not None else None
    test_file = output_dir / "test.manifest" if data_test is not None else None

    if data_train is not None:
        write_manifest_file(data_train, train_file)
    if data_valid is not None:
        write_manifest_file(data_valid, valid_file)
    if data_test is not None:
        write_manifest_file(data_test, test_file)

    return {
        "train": str(train_file.resolve()) if train_file else None,
        "validation": str(valid_file.resolve()) if valid_file else None,
        "test": str(test_file.resolve()) if test_file else None,
    }

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
from pathlib import Path

from mlsimkit.learn.manifest.schema import SplitSettings, DataFile, FileList
from mlsimkit.learn.manifest.manifest import (
    read_manifest_file,
    resolve_file_path,
    split_manifest,
    generate_manifest_entries,
    create_manifest_entry,
)


# Test cases for resolve_file_path
@pytest.mark.parametrize(
    "file_path, manifest_dir, expected_path",
    [
        # Test case for a relative path
        ("data/sample.txt", Path("/path/to/manifest"), "/path/to/manifest/data/sample.txt"),
        # Test case for an absolute path
        ("/abs/path/sample.txt", Path("/path/to/manifest"), "/abs/path/sample.txt"),
        # Test case for a URI path
        ("s3://my-bucket/path/to/data.txt", Path("/path/to/manifest"), "/path/to/data.txt"),
        # Test case for a file URI path
        ("file:///abs/path/sample.txt", Path("/path/to/manifest"), "/abs/path/sample.txt"),
    ],
)
def test_resolve_file_path(file_path, manifest_dir, expected_path):
    resolved_path = resolve_file_path(file_path, manifest_dir)
    assert resolved_path == expected_path


# Fixture to create a sample manifest file
@pytest.fixture
def sample_manifest_file(tmp_path):
    manifest_data = [
        {"geometry_files": ["file:///path/to/file1.stl"], "kpi": [0.1, 0.2, 0.3]},
        {"geometry_files": ["file:///path/to/file2.stl"], "kpi": [0.4, 0.5, 0.6]},
        {"geometry_files": ["file:///path/to/file3.stl"], "kpi": [0.7, 0.8, 0.9]},
        {"geometry_files": ["file:///path/to/file4.stl"], "kpi": [1.0, 1.1, 1.2]},
        {"geometry_files": ["file:///path/to/file5.stl"], "kpi": [1.3, 1.4, 1.5]},
        {"geometry_files": ["file:///path/to/file6.stl"], "kpi": [1.6, 1.7, 1.8]},
        {"geometry_files": ["file:///path/to/file7.stl"], "kpi": [1.9, 2.0, 2.1]},
        {"geometry_files": ["file:///path/to/file8.stl"], "kpi": [2.2, 2.3, 2.4]},
        {"geometry_files": ["file:///path/to/file9.stl"], "kpi": [2.5, 2.6, 2.7]},
        {"geometry_files": ["file:///path/to/file10.stl"], "kpi": [2.8, 2.9, 3.0]},
    ]
    manifest_file = tmp_path / "sample_manifest.manifest"
    pd.DataFrame(manifest_data).to_json(manifest_file, orient="records", lines=True)
    return str(manifest_file)


@pytest.fixture
def run_folder(tmp_path):
    run_folder_path = tmp_path / "run_folder"
    run_folder_path.mkdir()
    (run_folder_path / "param1_1.csv").write_text("value1,value2\n1.0,2.0")
    (run_folder_path / "param2_123.dat").write_text("value3\n3.0")
    (run_folder_path / "param2_blah.dat").write_text("value3\n4.0")
    (run_folder_path / "file1.txt").touch()
    (run_folder_path / "file2.txt").touch()
    return str(run_folder_path)


def test_split_manifest_invalid_settings():
    with pytest.raises(ValueError):
        settings = SplitSettings(train_size=0.4, valid_size=0.4, test_size=0.3)
        split_manifest("dummy_file.manifest", settings)


def test_split_manifest_output_dir(sample_manifest_file, tmp_path):
    output_dir = tmp_path / "output"
    settings = SplitSettings(train_size=0.6, valid_size=0.2, test_size=0.2)
    split_manifest(sample_manifest_file, settings, str(output_dir))

    assert (output_dir / "train.manifest").exists()
    assert (output_dir / "validate.manifest").exists()
    assert (output_dir / "test.manifest").exists()

    assert 6 == len(read_manifest_file(output_dir / "train.manifest"))
    assert 2 == len(read_manifest_file(output_dir / "validate.manifest"))
    assert 2 == len(read_manifest_file(output_dir / "test.manifest"))


def test_split_manifest_non_equal(sample_manifest_file, tmp_path):
    output_dir = tmp_path / "output"
    settings = SplitSettings(train_size=0.5, valid_size=0.3, test_size=0.2)
    split_manifest(sample_manifest_file, settings, str(output_dir))

    assert 5 == len(read_manifest_file(output_dir / "train.manifest"))
    assert 3 == len(read_manifest_file(output_dir / "validate.manifest"))
    assert 2 == len(read_manifest_file(output_dir / "test.manifest"))


def test_generate_manifest_entry_csv_invalid(tmp_path):
    run_folder = str(tmp_path / "run_1")
    data_file_path = tmp_path / "run_1" / "params.csv"
    data_file_path.parent.mkdir(parents=True)
    with data_file_path.open("w") as f:
        f.write("diameter,height\nHello,2.0")

    data_files = [DataFile(name="params", file_glob="params.csv", columns=["diameter", "height"])]
    file_lists = []

    with pytest.raises(RuntimeError):
        list(generate_manifest_entries([run_folder], data_files, file_lists, skip_on_error=False))

    entries = list(generate_manifest_entries([run_folder], data_files, file_lists, skip_on_error=True))
    assert len(entries) == 0


def test_generate_manifest_entry_valid(tmp_path):
    run_folder = str(tmp_path / "run_1")
    data_file_path = tmp_path / "run_1" / "params.csv"
    data_file_path.parent.mkdir(parents=True)
    with data_file_path.open("w") as f:
        f.write("diameter,height\n1.0,2.0")

    data_files = [DataFile(name="params", file_glob="params.csv", columns=["diameter", "height"])]
    file_lists = []

    entries = list(generate_manifest_entries([run_folder], data_files, file_lists, skip_on_error=False))
    assert len(entries) == 1
    assert entries[0] == {"params": [1.0, 2.0]}


def test_create_manifest_entry_glob(run_folder):
    data_files = [
        DataFile(name="param1", file_glob="param1_*.csv", columns=["value1", "value2"]),
    ]
    file_lists = [FileList(name="files1", file_glob="*.txt")]

    manifest_entry = create_manifest_entry(run_folder, data_files, file_lists)

    expected_entry = {
        "param1": [1.0, 2.0],
        "files1": [
            f"file://{run_folder}/file1.txt",
            f"file://{run_folder}/file2.txt",
        ],
    }
    assert manifest_entry == expected_entry


def test_create_manifest_entry_regex(run_folder):
    data_files = [
        DataFile(name="param2", file_regex=r"param2_\d+\.dat", columns=["value3"]),
    ]
    file_lists = [FileList(name="files1", file_glob="*.txt")]

    manifest_entry = create_manifest_entry(run_folder, data_files, file_lists)

    expected_entry = {
        "param2": [3.0],
        "files1": [
            f"file://{run_folder}/file1.txt",
            f"file://{run_folder}/file2.txt",
        ],
    }
    assert manifest_entry == expected_entry

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import click
import mlsimkit
from mlsimkit.common.logging import LogConfig, configure_logging

from .manifest import (
    generate_manifest_entries,
    split_manifest,
    write_manifest_files,
)
from .schema import DataFile, FileList, SplitSettings


@mlsimkit.cli.group(name="manifest")
@mlsimkit.cli.options(LogConfig, dest="log_config", prefix="log", help_group="Logging")
def manifest_cli(ctx: click.Context, log_config: LogConfig):
    """
    Tools to create and manipulate manifests for datasets
    """
    configure_logging(log_config)


# fmt: off
@manifest_cli.command()
@click.argument(
    "run_folders", nargs=-1, type=click.Path(exists=True)
)
@mlsimkit.cli.options_from_schema_shorthand("-d", "--datafile", model=DataFile, multiple=True,
    help='Extract parameter sets from CSV files. Expects one file per run folder with two rows (header, values).'
         'Format is "name=<str>,file_glob=<glob_pattern>,columns=<name> <name>" or'
         'Format is "name=<str>,file_regex=<regex_pattern>,columns=<name> <name>"'
)
@mlsimkit.cli.options_from_schema_shorthand("-f", "--filelist", model=FileList, multiple=True,
    help='Add lists of files to the manifest. Format is "name=<str>,file_glob=<glob_pattern>"'
)
@click.option( "--manifest-file", "-m", type=click.File("w"), default="manifest.jsonl",
    help="Output filename for the manifest",
)
@click.option("--skip-on-error/--no-skip-on-error", default=False,
              help="Skip runs that have errors parsing. If not set, the program will exit on error.")
# fmt: on
def create(ctx, run_folders, datafile, filelist, manifest_file, skip_on_error):
    """
    Create a manifest file from a dataset.

    A manifest file is a JSON lines file that contains metadata and file references for each run in a dataset.
    This command generates a manifest file from a list of run folders, extracting parameter values from data files
    and file paths from glob patterns.

    Examples:

    \b
    # Create a manifest from all run folders matching a pattern, including STL and slice image files
    mlsimkit-manifest create /path/to/dataset/run_* -f "name=geometry_files,file_glob=*.stl" -f "name=slices_uri,file_glob=slices/velocityxavg/view3_consty*.png" -m manifest.jsonl

    \b
    # Create a manifest from specific run folders matching a shell pattern, including STL and extracting parameters from a CSV data file
    mlsimkit-manifest create /path/to/dataset/run_[1-5] -d "name=params,file_glob=params.csv,columns=diameter height" -f "name=geometry_files,file_glob=*.stl" -m manifest.jsonl

    \b
    # Additionally the data files can be matched using a glob pattern or a regular expression pattern
    \b
    # Using the file_glob pattern matching will match any set of characters designated by `*` 
    mlsimkit-manifest create /path/to/dataset/run_[1-5] -d "name=params,file_glob=params-*.csv,columns=diameter height" -f "name=geometry_files,file_glob=*.stl" -m manifest.jsonl

    \b
    # Or using the file_regex pattern matching a more specific condition can be defined such as matching a set of digits
    mlsimkit-manifest create /path/to/dataset/run_[1-5] -d "name=params,file_regex=params-\\d+\\.csv,columns=diameter height" -f "name=geometry_files,file_glob=*.stl" -m manifest.jsonl
     
    \b
    # Pass a list of run folders through stdin 
    ls -d /path/to/dataset/run_folders* | xargs mlsimkit-manifest create -f "name=geometry_files,file_glob=*.stl" -m manifest.jsonl

    """
    try:
        manifest_entries = generate_manifest_entries(run_folders, datafile, filelist, skip_on_error)
        write_manifest_files(manifest_entries, manifest_file, run_folders)
    except Exception as e:
        raise click.UsageError(str(e))


@manifest_cli.command()
@click.option( "--manifest-file", "-m", required=False, type=click.Path(exists=True, path_type=Path), help="Manifest file to split")
@click.option( "--output-dir", default=None, type=click.Path(exists=False), help="Output directory for the split manifest file. If not specified, uses the input manifest file directory.")
@mlsimkit.cli.options(SplitSettings, dest='settings')
def split(ctx, manifest_file, output_dir, settings):
    """
    Split a manifest file into train, valid & test sets
    """

    # manifest is required, so check here to allow for config loading, and because
    #  click.option required=True are not supported with config loading (TODO)
    if not manifest_file:
        raise click.UsageError("Missing option --manifest-file")
    if not Path(manifest_file).exists():
        raise click.UsageError(f"--manifest-file'{manifest_file}' not found")

    split_manifest(manifest_file, settings, output_dir)

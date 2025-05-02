# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time

from functools import partial
from itertools import repeat, count
from multiprocessing import Pool, cpu_count
from pathlib import Path
from urllib.parse import urljoin

import torch
import numpy as np

from .schema.preprocessing import PreprocessingSettings, InterpolationMethod

from mlsimkit.common.logging import getLogger
from mlsimkit.learn.manifest.manifest import (
    get_base_path,
    resolve_file_path,
    read_manifest_file,
    make_working_manifest,
    write_manifest_file,
)

from mlsimkit.learn.common.mesh import load_mesh_files_pyvista, as_torch_data, downsample, convert

log = getLogger(__name__)


def calculate_node_weights(mesh):
    # Compute cell sizes
    cell_areas = mesh.compute_cell_sizes()['Area']

    # Initialize an array to store node weights
    node_weights = np.zeros(mesh.n_points)

    # Extract faces connectivity array
    faces = mesh.faces

    # Iterate over each face to distribute the cell area to its nodes
    index = 0
    for cell_id in range(mesh.n_cells):
        num_points = faces[index]  # Number of points in the face
        point_ids = faces[index + 1:index + 1 + num_points]  # The point IDs making up the face
        cell_area = cell_areas[cell_id]

        # Distribute the cell area to the points of the cell
        for point_id in point_ids:
            node_weights[point_id] += cell_area / num_points

        # Move to the next face in the faces array
        index += num_points + 1

    return node_weights


def process_mesh(args, config, surface_variables, output_dir, mesh_loader):
    """
    Process a single mesh file or a set of mesh files.

    This function loads the mesh data using the provided mesh_loader function, computes normals,
    and converts the mesh data into a torch_geometric.data.Data object. If surface variables are
    provided, it also extracts the corresponding ground truth data from the mesh and attaches it
    to the Data object.

    Args:
        args (tuple): A tuple containing the following elements:
            path_num (int): The index of the current file being processed.
            path_list (list or tuple): A list or tuple containing the mesh file paths.
            total_files (int): The total number of files to be processed.
            downsample_perc (float): The percentage of points to keep during downsampling.
        config (PreprocessingSettings): The configuration settings for preprocessing.
        surface_variables (list): A list of surface variables to extract from the mesh data.
        output_dir (str or Path): The directory where intermediate files will be saved.
        mesh_loader (callable): A function that loads the mesh data based on the provided file paths.

    Returns:
        torch_geometric.data.Data: A Data object containing the processed mesh data and ground truth data (if provided).
    """
    path_num, path_list, total_files, downsample_perc = args
    log.info(f"Preprocessing file {path_num} out of {total_files}")

    mesh = mesh_loader(path_list, config=config, output_dir=output_dir, downsample_perc=downsample_perc)
    mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)

    # FIXME: monkey patch to make this look like trimesh as a stop-gap for reusing .as_torch_data()
    assert not hasattr(mesh, "vertex_normals"), "vertex_normals already present in mesh, cannot preprocess"
    assert not hasattr(mesh, "vertices"), "vertices already present in mesh, cannot preprocess"
    assert not hasattr(mesh, "vertex_defects"), "vertex_defects already present in mesh, cannot preprocess"
    mesh.vertex_normals = mesh.point_normals
    mesh.vertices = mesh.points
    mesh.vertex_defects = None

    mesh.node_weights = calculate_node_weights(mesh)

    base_graph = as_torch_data(mesh, config.save_cell_data, config.normalize_node_positions)

    # Get output data if ground truth exist
    if surface_variables:
        mesh = mesh.cell_data_to_point_data()
        y_list = []

        # variable names are the same shape as y variables, but replace
        # the SurfaceVariable type with dict to avoid mlsimkit-dependency in the
        # .pt files
        name_list = []
        for i, var in enumerate(surface_variables):
            if var.dimensions:
                if mesh.get_array(var.name).ndim == 1:
                    raise TypeError(
                        f'Variable "{var.name}" has only one component, please remove the "dimensions" field for this variable or set it to [].'
                    )
                variable = torch.tensor(mesh.get_array(var.name)[:, var.dimensions])
                for d in var.dimensions:
                    name_list.append({"name": var.name, "dimension": d})
            else:
                variable = torch.tensor(mesh.get_array(var.name))
                if variable.ndim > 1:
                    surface_variables[i].dimensions = list(range(variable.shape[1]))
                name_list.append({"name": var.name})
            y_list.append(variable.reshape(variable.shape[0], -1))

        y = torch.cat(y_list, dim=1)
        base_graph.y = y
        base_graph.y_variables = name_list

    base_graph.mesh_path = path_list
    base_graph.downsample_perc = downsample_perc
    return base_graph


def construct_processed_data(
    mesh_loader,
    mesh_paths,
    downsample_perc,
    num_processes,
    config=None,
    surface_variables=None,
    output_dir=".",
):
    """
    Construct a generator that yields processed mesh data.

    This function is responsible for preprocessing mesh data in parallel using multiprocessing if
    num_processes > 1, or in a single process if num_processes <= 1. It yields torch_geometric.data.Data
    objects containing the processed mesh data and ground truth data (if provided).

    Args:
        mesh_loader (callable): A function that loads the mesh data based on the provided file paths.
        mesh_paths (list or tuple): A list or tuple containing the mesh file paths.
        downsample_perc (float): The percentage of points to keep during downsampling.
        num_processes (int): The number of processes to use for parallel processing.
        config (PreprocessingSettings, optional): The configuration settings for preprocessing.
        surface_variables (list, optional): A list of surface variables to extract from the mesh data.
        output_dir (str or Path, optional): The directory where intermediate files will be saved.

    Yields:
        torch_geometric.data.Data: A Data object containing the processed mesh data and ground truth data (if provided).
    """
    log.info(f"Selected surface variables: {surface_variables}")
    process_mesh_partial = partial(
        process_mesh,
        config=config,
        surface_variables=surface_variables,
        output_dir=output_dir,
        mesh_loader=mesh_loader,
    )
    items = zip(count(1), mesh_paths, repeat(len(mesh_paths)), repeat(downsample_perc))
    if num_processes <= 1:
        # allow user to work-around multi-processing known bug (due to too many open FDs)
        log.info("Using single processor for preprocessing data")
        for item in items:
            yield process_mesh_partial(item)
    else:
        with Pool(processes=num_processes) as pool:
            for datum in pool.imap(process_mesh_partial, items):
                yield datum


def get_paths(config, row, key):
    base_path = get_base_path(config.manifest_base_relative_path, config.manifest_path)
    return [resolve_file_path(f, default_dir=base_path, missing_ok=False) for f in row[key]]


def mesh_loader_map(mesh_file_pairs, config, output_dir, downsample_perc):
    """
    Load mesh data by mapping data files to geometry files. Geometry files formats
    are converted if supported and downsampled if requested.
    """
    geometry_files = mesh_file_pairs[0]
    data_files = mesh_file_pairs[1]
    for i, filepath in enumerate(geometry_files):
        geometry_files[i] = str(
            downsample(convert(filepath, output_dir), output_dir, downsample_perc).resolve()
        )
    return map_data_to_stl(geometry_files, data_files, config)


def mesh_loader_geometry(mesh_files, config, output_dir, downsample_perc):
    """
    Load mesh data geometry files. Geometry files formats are converted if supported and downsampled if requested.
    """
    for i, filepath in enumerate(mesh_files):
        mesh_files[i] = str(downsample(convert(filepath, output_dir), output_dir, downsample_perc).resolve())
    return load_mesh_files_pyvista(*mesh_files)


def mesh_loader_data_only(mesh_files, config, output_dir, downsample_perc):
    """
    Load mesh data from data files (e.g., VTU, VTK) without any preprocessing.
    """
    return load_mesh_files_pyvista(*mesh_files)


def map_data_to_stl(geometry_path, data_path, config):
    """
    Map data files (e.g., VTU, VTK) to geometry files (STL) using interpolation.

    This function loads the geometry files (STL) and data files (VTU, VTK), converts the data files to point data,
    and interpolates the data onto the geometry files using a specified radius or a calculated radius based on cell sizes.
    The mapped data can optionally be saved to disk.

    Args:
        geometry_path (list): A list of geometry file paths (STL).
        data_path (list): A list of data file paths (VTU, VTK).
        config (PreprocessingSettings): The configuration settings for preprocessing.

    Returns:
        pyvista.UnstructuredGrid: The mapped mesh data.
    """
    geometry = load_mesh_files_pyvista(geometry_path)
    data = load_mesh_files_pyvista(data_path)
    data = data.cell_data_to_point_data()
    
    if config.mapping_interpolation_method == InterpolationMethod.POINTS:
        mapped_data = geometry.interpolate(data, n_points=config.mapping_interpolation_n_points)
    else: 
        if config.mapping_interpolation_radius:
            radius = config.mapping_interpolation_radius
        else:
            data_tmp = data.extract_all_edges().compute_cell_sizes(length=True, area=False, volume=False)
            radius = max(data_tmp["Length"])

        mapped_data = geometry.interpolate(data, radius=radius)

    if config.save_mapped_files:
        # TODO: use run names or ids in mapped data file names
        path_split = data_path[0].split("/")[-1].split(".")
        mapped_file_dir = os.path.join(config.output_dir, "mapped_data_files")
        os.makedirs(mapped_file_dir, exist_ok=True)
        mapped_data.save(os.path.join(mapped_file_dir, path_split[-2] + "_mapped.vtp"))
    return mapped_data


def read_manifest(manifest_path, config, base_relative_path: Path = None):
    """
    Read the manifest file and prepare the mesh file paths for preprocessing.

    This function reads the manifest file and checks if both 'data_files' and 'geometry_files' are provided.
    If 'map_data_to_stl' is enabled in the configuration, it pairs the data files and geometry files for each sample
    into a new 'mesh_paths' field. Otherwise, it selects either the 'geometry_files'
    or 'data_files' for preprocessing.

    Args:
        manifest_path (str or Path): The path to the manifest file.
        config (PreprocessingSettings): The configuration settings for preprocessing.
        base_relative_path (Path, optional): The base relative path for resolving file paths.

    Returns:
        tuple: A tuple containing:
            manifest (pandas.DataFrame): The manifest DataFrame with the appropriate file paths.
            mesh_key (str): The key in the manifest DataFrame for the selected mesh file paths.
    """
    manifest = read_manifest_file(manifest_path)

    if (
        not config.map_data_to_stl
        and "data_files" in manifest.columns
        and "geometry_files" in manifest.columns
    ):
        raise TypeError(
            "Both 'data_files' and 'geometry_files' are provided in the manifest. Please choose one of the following options: "
            'either keep only the geometry files or data files for preprocessing, or enable mapping data files to STLs by setting "map_data_to_stl" argument to True.'
        )

    if config.map_data_to_stl and (
        "geometry_files" not in manifest.columns or "data_files" not in manifest.columns
    ):
        raise TypeError(
            "Cannot apply 'map_data_to_stl=True' argument, requires both 'data_files' and 'geometry_files' in the manifest."
        )

    def map_mesh_paths(row):
        # Pair data files and geometry files per sample if all the file paths exist in both of the sub-lists
        return (get_paths(config, row, "geometry_files"), get_paths(config, row, "data_files"))

    if config.map_data_to_stl:
        manifest["mesh_paths"] = manifest.apply(map_mesh_paths, axis=1)
        mesh_key = "mesh_paths"
        log.info("Mapped 'data_files' to 'geometry_files' for preprocessing")
    elif "geometry_files" in manifest.columns:
        mesh_key = "geometry_files"
        manifest[mesh_key] = manifest.apply(lambda row: get_paths(config, row, mesh_key), axis=1)
        log.info("Using 'geometry_files' for preprocessing")
    elif "data_files" in manifest.columns:
        # select data files (e.g., vtu, vtk) for preprocessing
        mesh_key = "data_files"
        manifest[mesh_key] = manifest.apply(lambda row: get_paths(config, row, mesh_key), axis=1)
        log.info("Using 'data_files' for preprocessing")

    return manifest, mesh_key


def process_mesh_files(config, manifest, mesh_key, surface_variables):
    """
    Preprocess the mesh files and save the processed data.

    This function preprocesses the mesh files listed in the manifest DataFrame using the specified mesh key.
    It constructs a generator that yields processed mesh data, which is then saved to disk as PyTorch files.
    The manifest DataFrame is updated with the paths to the saved files.

    Args:
        config (PreprocessingSettings): The configuration settings for preprocessing.
        manifest (pandas.DataFrame): The manifest DataFrame.
        mesh_key (str): The key in the manifest DataFrame for the mesh file paths.
        surface_variables (list): A list of surface variables to extract from the mesh data.

    Returns:
        pandas.DataFrame: The updated manifest DataFrame with the paths to the saved processed data files.
    """
    output_dir = Path(config.output_dir)
    downsample_perc = config.downsample_remaining_perc
    num_processes = config.num_processes or (cpu_count() - 1)

    log.info(f"Preprocessing mesh files (num_processes={num_processes})")

    output_dir.mkdir(parents=True, exist_ok=True)

    # We load the mesh data differently depending on source data, kinda a stop-gap until we refactor mesh.py module better.
    mesh_loaders = {
        "mesh_paths": mesh_loader_map,
        "geometry_files": mesh_loader_geometry,
        "data_files": mesh_loader_data_only,
    }

    # now process the data lazily by using a generator, actual processing happens during enumerate after
    processed_data = construct_processed_data(
        mesh_loaders[mesh_key],
        manifest[mesh_key],
        downsample_perc,
        num_processes,
        config,
        surface_variables,
        output_dir,
    )

    # we process each run during iteration, this allows us to release resources when we are
    # done processing each run. This is important for multi-processing so torch can release
    # file descriptors used for inter-process comms.
    for i, datum in enumerate(processed_data):
        filepath = output_dir / f"preprocessed_run_{i:0>5}.pt"
        torch.save(datum, filepath)
        log.info("Run %i preprocessed, written to '%s'", i, filepath)
        manifest.loc[i, ("id", "preprocessed_files")] = (
            i,
            urljoin("file://", filepath.resolve().as_posix()),
        )

    log.info(f"Saved output files in {output_dir}")

    return manifest


def run_preprocess(config: PreprocessingSettings, project_root: Path = None, surface_variables = None):
    log.info(f"Preprocessing configuration: {config}")
    os.makedirs(config.output_dir, exist_ok=True)

    project_root = project_root or config.output_dir

    t_start = time.time()

    # Use a working manifest for writing, avoid changing user manifests
    working_manifest_path = make_working_manifest(config.manifest_path, project_root)
    manifest, mesh_key = read_manifest(working_manifest_path, config)
    manifest = process_mesh_files(config, manifest, mesh_key, surface_variables)
    write_manifest_file(manifest, working_manifest_path)

    t_end = time.time()
    log.info(f"Total preprocessing time: {(t_end - t_start):.3f} seconds")

    return working_manifest_path

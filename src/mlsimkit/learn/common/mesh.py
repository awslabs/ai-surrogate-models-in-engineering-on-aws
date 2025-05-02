# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from multiprocessing import Pool
from itertools import repeat, count
from typing import List, Optional, Generator, Tuple

import trimesh
import numpy as np
import pyvista as pv
import vtk

import torch
import torch_geometric.data

from mlsimkit.common.logging import getLogger


log = getLogger(__name__)


def downsample_mesh_file(mesh_filepath: str, output_path: str, downsample_perc: float) -> Path:
    """
    Downsample a mesh file to the specified percentage and save the downsampled mesh to the output path.

    Args:
        mesh_filepath (str): Path to the input mesh file.
        output_path (str): Path to save the downsampled mesh file.
        downsample_perc (float): Percentage of vertices to keep after downsampling (between 0 and 100).

    Returns:
        Path: Path to the downsampled mesh file.

    Example:
        >>> downsample_mesh_file('mesh.stl', 'downsampled_mesh.stl', 50.0)
        Path('downsampled_mesh.stl')
    """
    mesh = pv.read(mesh_filepath)
    decimated = mesh.decimate(1 - (downsample_perc / 100))
    decimated.save(output_path)
    return Path(output_path)


def downsample(original_mesh_filepath: str, output_dir: str, downsample_perc: Optional[float]) -> Path:
    """
    Wrapper to downsample a mesh file to the specified percentage and save the downsampled mesh to the output directory.
    If downsample_perc is None, the original mesh file is returned.

    Args:
        original_mesh_filepath (str): Path to the input mesh file.
        output_dir (str): Directory to save the downsampled mesh file.
        downsample_perc (float, optional): Percentage of vertices to keep after downsampling (between 0 and 100).

    Raises:
        RuntimeError: If downsample_perc is not between 0 and 100.

    Returns:
        Path: Path to the downsampled mesh file, or the original mesh file if downsample_perc is None.

    Example:
        >>> downsample('mesh.stl', 'output', 50.0)
        Path('output/50perc_ds/mesh_50perc_ds.stl')
        >>> downsample('mesh.stl', 'output', None)
        Path('mesh.stl')
    """
    if not downsample_perc:
        return Path(original_mesh_filepath)

    if downsample_perc <= 0.0 or downsample_perc > 100.0:
        raise RuntimeError(
            f"Expected downsample between 0.0 (exclusive) and 100.0 (inclusive), got {downsample_perc}"
        )

    log.debug("Downsampling mesh to '%s' percent: '%s'", downsample_perc, original_mesh_filepath)
    mesh_output_dir = Path(output_dir) / f"{downsample_perc}perc_ds/"
    mesh_output_dir.mkdir(parents=True, exist_ok=True)
    output_mesh_filepath = (
        mesh_output_dir / f"{Path(original_mesh_filepath).stem}_{downsample_perc}perc_ds.stl"
    )
    return downsample_mesh_file(original_mesh_filepath, str(output_mesh_filepath), downsample_perc)


def convert_vtp_to_stl(vtp_file_path: Path, stl_file_path: Path) -> Path:
    """
    Convert a VTP file to an STL file.

    Args:
        vtp_file_path (Path): Path to the input VTP file.
        stl_file_path (Path): Path to save the output STL file.

    Returns:
        Path: Path to the output STL file.

    Example:
        >>> convert_vtp_to_stl(Path('mesh.vtp'), Path('mesh.stl'))
        Path('mesh.stl')
    """
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_file_path.as_posix())
    reader.Update()
    stlWriter = vtk.vtkSTLWriter()
    stlWriter.SetFileName(stl_file_path.as_posix())
    stlWriter.SetInputConnection(reader.GetOutputPort())
    stlWriter.Write()
    return stl_file_path


def convert(filepath: str, output_dir: str) -> Path:
    """
    Convert a mesh file to an STL file if the file extension is not supported.

    Args:
        filepath (str): Path to the input mesh file.
        output_dir (str): Directory to save the converted STL file.

    Raises:
        RuntimeError: If the file extension is not supported and there is no converter available.

    Returns:
        Path: Path to the converted STL file, or the original file if it is already an STL.

    Example:
        >>> convert('mesh.stl', 'output')
        Path('mesh.stl')
        >>> convert('mesh.vtp', 'output')
        Path('output/stl/mesh.stl')
    """
    supported_types = [".stl"]
    converters = {".vtp": convert_vtp_to_stl}

    filepath = Path(filepath)
    if filepath.suffix in supported_types:
        return filepath

    converter = converters.get(filepath.suffix, None)
    if converter:
        log.debug("Converting mesh '%s'", filepath)
        mesh_output_dir = Path(output_dir) / "stl"
        mesh_output_dir.mkdir(parents=True, exist_ok=True)
        new_mesh_filepath = mesh_output_dir / Path(filepath.name).with_suffix(".stl")
        return converter(filepath, new_mesh_filepath)
    else:
        raise RuntimeError(f"Unsupported geometry suffix in file: '{filepath}'")


def get_edges(mesh, cells: torch.Tensor) -> torch.Tensor:
    """
    Get two-way mesh edges via mesh cell data.

    Args:
        cells (torch.Tensor): Tensor of mesh cell data (faces).

    Returns:
        torch.Tensor: Tensor of two-way mesh edges.

    Example:
        >>> cells = torch.tensor([[0, 1, 2], [2, 3, 4]])
        >>> get_edges(cells)
        tensor([[0, 1],
                [1, 2],
                [2, 0],
                [2, 3],
                [3, 4],
                [4, 2]])
    """
    if isinstance(mesh, pv.PolyData):
        all_edges = mesh.extract_all_edges(use_all_points=True)
        edges = torch.tensor(
            all_edges.point_data_to_cell_data()
            .cast_to_unstructured_grid()
            .cell_connectivity[:]
            .reshape((-1, 2))
        )
    else:
        edges = torch.cat(
            [cells[:, 0:2], cells[:, 1:3], torch.stack([cells[:, 2], cells[:, 0]], dim=1)],
            dim=0,
        )

    tails, _ = torch.min(edges, dim=1)
    heads, _ = torch.max(edges, dim=1)
    ordered_edges = torch.stack([tails, heads], dim=1)
    unique_edges = torch.unique(ordered_edges, dim=0)
    tails, heads = unique_edges[:, 0], unique_edges[:, 1]
    two_way_edges = torch.cat(
        (
            torch.cat([tails, heads], dim=0).unsqueeze(0),
            torch.cat([heads, tails], dim=0).unsqueeze(0),
        ),
        dim=0,
    )
    return two_way_edges


def load_mesh(path: str) -> trimesh.Trimesh:
    """
    Load a mesh from file. Always flattens if the file is a scene.

    Args:
        path (str): Path to the input mesh file.

    Returns:
        trimesh.Trimesh: Loaded mesh.

    Example:
        >>> mesh = load_mesh('mesh.stl')
    """
    mesh = trimesh.load_mesh(path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def load_mesh_files(*paths: str):
    mesh = trimesh.util.concatenate([load_mesh(p) for p in paths])
    mesh.merge_vertices()
    return mesh


def load_mesh_files_pyvista(*paths: str):
    mesh = pv.PolyData()
    for p in paths:
        mesh_object = pv.read(p)
        if isinstance(mesh_object, pv.UnstructuredGrid):
            mesh_object = mesh_object.extract_surface()
        mesh = mesh.merge(mesh_object, merge_points=True)
    return mesh


def normalize_points(original_node_positions):
    """
    Normalize a numpy array of 3D points so that they fit within the cube [-1, 1] in all dimensions,
    while maintaining the shape and aspect ratio of the original distribution.

    Args:
        original_node_positions (np.ndarray): A 2D numpy array where each row represents a 3D point (x, y, z).

    Returns:
        np.ndarray: A 2D numpy array of normalized 3D points.

    Example:
        >>> original_node_positions = np.array([
        ...     [2, 5, -3],
        ...     [4, -2, 1],
        ...     [5, 3, -1],
        ...     [-1, -4, 2]
        ... ])
        >>> normalize_points(original_node_positions)
        array([[ 0.          1.         -0.55555556],
               [ 0.44444444 -0.55555556  0.33333333],
               [ 0.66666667  0.55555556 -0.11111111],
               [-0.66666667 -1.          0.55555556]])
    """

    # Find the min and max of each dimension
    min_vals = np.min(original_node_positions, axis=0)
    max_vals = np.max(original_node_positions, axis=0)

    # Calculate the range and determine the max range
    ranges = max_vals - min_vals
    max_range = np.max(ranges)

    # Calculate the scale factor to fit data in [-1, 1]
    scale_factor = 2 / max_range

    # Center and scale the data
    middle_point = (min_vals + max_vals) / 2
    normalized_node_positions = (original_node_positions - middle_point) * scale_factor

    return normalized_node_positions, middle_point, scale_factor


def as_torch_data(
    mesh,
    save_cell_data: bool = False,
    normalize_node_positions: bool = False,
    label: Optional[int] = None,
    global_condition: Optional[int] = None,
) -> torch_geometric.data.Data:
    """
    Convert mesh to PyTorch Geometric Data format.

    Args:
        path_list (List[str]): List of paths to mesh files.
        label (int, optional): Label for the mesh data. Defaults to None.

    Returns:
        torch_geometric.data.Data: PyTorch Geometric Data object containing the mesh data.

    Example:
        >>> mesh_paths = ['mesh1.stl', 'mesh2.stl']
        >>> data = as_torch_data(mesh_paths, label=1)
        >>> print(data)
        Data(x=[2048, 7], edge_attr=[4096, 4], edge_index=[2, 4096], y=[1])
    """

    # Get node features
    if normalize_node_positions:
        normalized_node_positions, middle_point, scale_factor = normalize_points(mesh.vertices)
        mesh.vertices = normalized_node_positions
    node_normals = torch.tensor(mesh.vertex_normals)
    node_positions = torch.tensor(mesh.vertices)
    if mesh.vertex_defects is not None:
        node_defects = torch.tensor(mesh.vertex_defects.reshape(-1, 1))
        node_features = torch.cat((node_normals, node_defects, node_positions), dim=-1).type(torch.float)
    else:
        node_features = torch.cat((node_normals, node_positions), dim=-1).type(torch.float)
    if hasattr(mesh, "node_weights"):
        node_weights = torch.tensor(mesh.node_weights, dtype=torch.float)

    # Get edge indices
    cells = torch.tensor(mesh.faces)
    edge_index = get_edges(mesh, cells)

    # Get edge features
    node_i = node_positions[edge_index[0]]
    node_j = node_positions[edge_index[1]]
    edge_ij = node_i - node_j
    edge_ij_length = torch.linalg.norm(edge_ij, ord=2, dim=1, keepdim=True)
    edge_features = torch.cat((edge_ij, edge_ij_length), dim=-1).type(torch.float)

    data = torch_geometric.data.Data(
        x=node_features,
        edge_attr=edge_features,
        edge_index=edge_index,
        cells=cells if save_cell_data else None,
        middle_point=torch.tensor(middle_point, dtype=torch.float) if normalize_node_positions else None,
        scale_factor=torch.tensor(scale_factor, dtype=torch.float) if normalize_node_positions else None,
        node_weights=node_weights if hasattr(mesh, "node_weights") else None,
    )

    if label is not None:
        data.y = torch.tensor(np.ones((1, 1)) * label)
    if global_condition is not None:
        data.global_condition = torch.tensor(np.ones((1, 1)) * global_condition)

    return data


def process_mesh(
    args: Tuple[int, List[str], Optional[int], Optional[int], str, int, Optional[float]],
) -> torch_geometric.data.Data:
    """
    Pre-process a list of mesh files by converting and downsampling them, and convert them to PyTorch Geometric Data format.

    Args:
        args (Tuple[int, List[str], Optional[int], str, int, Optional[float]]): A tuple containing:
            - path_num (int): Index of the current file being processed.
            - path_list (List[str]): List of paths to mesh files.
            - label (Optional[int]): Label for the mesh data.
            - output_dir (str): Directory to save the converted and downsampled mesh files.
            - total_files (int): Total number of files to be processed.
            - downsample_perc (Optional[float]): Percentage of vertices to keep after downsampling (between 0 and 100).

    Returns:
        torch_geometric.data.Data: PyTorch Geometric Data object containing the mesh data.

    Example:
        >>> mesh_paths = ['mesh1.stl', 'mesh2.stl']
        >>> args = (1, mesh_paths, 1, 'output', 2, 50.0)
        >>> data = process_mesh(args)
        >>> print(data)
        Data(x=[1024, 7], edge_attr=[2048, 4], edge_index=[2, 2048], y=[1], mesh_path=['output/50perc_ds/mesh1_50perc_ds.stl', 'output/50perc_ds/mesh2_50perc_ds.stl'], downsample_perc=50.0)
    """
    path_num, path_list, label, global_condition, output_dir, total_files, downsample_perc = args
    log.info(f"Pre-processing file {path_num} out of {total_files} files")

    # convert and downsample mesh format if necessary
    for i, filepath in enumerate(path_list):
        path_list[i] = str(downsample(convert(filepath, output_dir), output_dir, downsample_perc).resolve())

    # then persist to file as torch data format for training
    mesh = load_mesh_files(*path_list)
    result = as_torch_data(mesh, label=label, global_condition=global_condition)
    result.mesh_path = [Path(p).resolve().as_posix() for p in path_list]
    result.downsample_perc = downsample_perc
    return result


def construct_processed_data(
    mesh_paths: List[str],
    labels: Optional[List[int]] = None,
    global_conditions: Optional[List[int]] = None,
    output_dir: str = ".",
    downsample_perc: Optional[float] = None,
    num_processes: int = 1,
) -> Generator[torch_geometric.data.Data, None, None]:
    """
    Construct a generator that yields processed mesh data in PyTorch Geometric Data format.

    Args:
        mesh_paths (List[str]): List of paths to mesh files.
        labels (List[int], optional): List of labels for the mesh data. If not provided, labels will be set to None.
        global_conditions (List[int], optional): List of global conditions for the mesh data. If not provided, it will be set to None.
        output_dir (str, optional): Directory to save the converted and downsampled mesh files. Defaults to the current directory.
        downsample_perc (float, optional): Percentage of vertices to keep after downsampling (between 0 and 100). If not provided, no downsampling will be performed.
        num_processes (int, optional): Number of processes to use for parallel processing. Defaults to 1 (single process).

    Yields:
        torch_geometric.data.Data: PyTorch Geometric Data object containing the mesh data.

    Example:
        >>> mesh_paths = ['mesh1.stl', 'mesh2.stl', 'mesh3.stl']
        >>> labels = [0, 1, 0]
        >>> data_generator = construct_processed_data(mesh_paths, labels, output_dir='output', downsample_perc=50.0, num_processes=4)
        >>> for data in data_generator:
        ...     print(data)
        Data(x=[1024, 7], edge_attr=[2048, 4], edge_index=[2, 2048], y=[0], mesh_path=['output/50perc_ds/mesh1_50perc_ds.stl'], downsample_perc=50.0)
        Data(x=[2048, 7], edge_attr=[4096, 4], edge_index=[2, 4096], y=[1], mesh_path=['output/50perc_ds/mesh2_50perc_ds.stl'], downsample_perc=50.0)
        Data(x=[1536, 7], edge_attr=[3072, 4], edge_index=[2, 3072], y=[0], mesh_path=['output/50perc_ds/mesh3_50perc_ds.stl'], downsample_perc=50.0)
    """
    labels = labels if labels is not None else repeat(None)
    global_conditions = global_conditions if global_conditions is not None else repeat(None)
    items = zip(
        count(1),
        mesh_paths,
        labels,
        global_conditions,
        repeat(output_dir),
        repeat(len(mesh_paths)),
        repeat(downsample_perc),
    )
    if num_processes <= 1:
        log.info("Using single processor for preprocessing data")
        # allow user to work-around multi-processing known bug (due to too many open FDs)
        for item in items:
            yield process_mesh(item)
    else:
        with Pool(processes=num_processes) as pool:
            for datum in pool.imap(process_mesh, items):
                yield datum

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
from functools import partial

import math
import click
import pyvista as pv

from mlsimkit.common.logging import getLogger
from .schema.view import ViewType

log = getLogger(__name__)


DefaultViews = [ViewType.GroundTruth, ViewType.Predicted, ViewType.Error]


class Viewer:
    def __init__(
        self, dataset, interactive=True, views=DefaultViews, fallback_to_geometry_files=True, window_size=None
    ):
        self.idx = 0
        self.dataset = dataset
        self.interactive = interactive
        self.views = views
        self.views_info = [{}] * len(self.views)
        self.fallback_to_geometry_files = fallback_to_geometry_files

        self.variable_idx = 0
        self.variables = self.dataset.surface_variables()
        self.variable = self.variables[self.variable_idx]
        # filename-friendly variable names (remove the square brackets)
        self.variable_filenames = [
            f"{v['name']}_dim{v['dimension']}" if "dimension" in v else v["name"] for v in self.variables
        ]

        shape = (1, len(self.views))
        window_size = window_size or [1500 * len(self.views), 2000]

        if self.interactive:
            self.plotter = pv.Plotter(
                off_screen=not self.interactive,
                shape=shape,
                window_size=window_size,
                title="MLSimKit Surface Prediction Viewer",
            )
            self.plotter.link_views()  # link all the views for interactive
            self.plotter.add_key_event("Right", lambda: self.next_datum())
            self.plotter.add_key_event("space", lambda: self.next_datum())
            self.plotter.add_key_event("Left", lambda: self.prev_datum())
            self.plotter.add_key_event("Tab", lambda: self.next_variable())
            self.plotter.add_key_event("g", lambda: self.plotter.show_grid())
            self.plotter.add_key_event("i", lambda: self.toggle_info_text())
            self.plotter.add_key_event("h", lambda: self.toggle_help_text())

        else:
            self.plotter = pv.Plotter(off_screen=True, shape=shape)

        self.show_info_text = True
        self.info_text = ""

        self.show_help_text = True
        self.help_text = """

[h]            - toggle this help
[i]            - show info 
[right][space] - next mesh
[left]         - previous mesh
[tab]          - next surface variable
[w]/[s]        - render wireframe/surface 
                        (cursor over view)
"""

    def next_datum(self, obj=None, event=None):
        self.idx = (self.idx + 1) % len(self.dataset)
        self.update_plotter()

    def prev_datum(self, obj=None, event=None):
        self.idx = (self.idx - 1) % len(self.dataset)
        self.update_plotter()

    def next_variable(self, obj=None, event=None):
        self.variable_idx = (self.variable_idx + 1) % len(self.variables)
        self.variable = self.variables[self.variable_idx]
        self.update_plotter()

    def set_variable(self, variable):
        for i, v in enumerate(self.variables):
            if v == variable:
                self.variable = v
                self.variable_idx = i
                return
        raise RuntimeError(f"Variable '{variable}' not in dataset")

    def update_plotter(self):
        self.plotter.clear()
        self._scalar_bar_ranges = {}
        self._mappers = [None] * len(self.views)

        scalar = self.variable.get("name")
        component = self.variable.get("dimension", 0)

        # fmt: off
        # predicted arrays aren't shaped like ground truth
        pscalar = ( self.variable["name"]
            if "dimension" not in self.variable else f"{self.variable['name']}[{self.variable['dimension']}]"
        )
        pscalar_error = f"{pscalar}_error"

        update_view = {
            # note: we share scalar labels across views to ensure the scalar bar range are the same
            ViewType.GroundTruth: lambda i: self.add_groundtruth_mesh(
                i, self.idx, scalar, component, scalar
            ),
            ViewType.Predicted: lambda i: self.add_mesh(
                i, self.dataset.predicted_file(self.idx, missing_ok=False), pscalar, 0, pscalar
            ),
            ViewType.Error: lambda i: self.add_mesh(
                i, self.dataset.predicted_file(self.idx, missing_ok=False), pscalar_error, 0, pscalar_error
            ),
        }
        # fmt: on

        # Add meshes to all views before adding scalar bars to avoid subplots updating over several frames
        # (which breaks screenshots because they get inconsistent scalar bar ranges)
        for i, view in enumerate(self.views):
            self.plotter.subplot(0, i)
            self.plotter.add_title(view.value)
            update_view[view](i)

        for i, view in enumerate(self.views):
            scalar_name = pscalar if view != ViewType.Error else pscalar_error
            self.add_scalar_bar(i, scalar_name, self._mappers[i])

        if self.interactive:
            # set the active subplot so the help text is in the first view for 2 or less splits,
            # otherwise in the 'middle' panel (depend whether odd or even number of splits)
            help_text_subplot = 0 if len(self.views) <= 2 else math.floor((len(self.views) - 1) / 2)
            self.plotter.subplot(0, help_text_subplot)
            if self.show_help_text:
                self.plotter.add_text(
                    text="\n\n" + self.help_text,
                    font_size=self.font_size(8),
                    font="courier",
                    viewport=True,
                    position=self.text_position((0.25, 0.90)),
                )

            self.plotter.subplot(0, 0)

            # update the text even if not showing so users can look at the text programmatically
            self.update_info_text(self.idx, pscalar)

            if self.show_info_text:
                position = ((0.05, 0.90)) if help_text_subplot != 0 else ((0.05, 0.80))
                self.plotter.add_text(
                    text="\n\n" + self.info_text,
                    font_size=self.font_size(8),
                    font="courier",
                    viewport=True,
                    position=self.text_position(position),
                )

    def add_groundtruth_mesh(self, view_idx, datum_idx, scalar, component, scalar_name):
        if self.dataset.has_data_files():
            (
                self.add_mesh(
                    view_idx,
                    self.dataset.data_files(datum_idx, missing_ok=False)[0],
                    scalar,
                    component,
                    scalar_name,
                ),
            )
        elif self.fallback_to_geometry_files:
            # geometry files do NOT have data, so don't try use scalars
            self.add_mesh(view_idx, self.dataset.geometry_files(datum_idx, missing_ok=False)[0], scalar=None)
        else:
            run_id = self.dataset.run_id(datum_idx)
            raise RuntimeError(f"Cannot load ground truth mesh, missing from manifest for run={run_id}")

    def add_mesh(self, view_idx, vtp, scalar=None, component=None, scalar_name=None):
        self.plotter.subplot(0, view_idx)
        self.plotter.add_mesh(pv.read(vtp), scalars=scalar, component=component, show_scalar_bar=False)
        self.views_info[view_idx] = shorten_path(Path(vtp))

        # Manually track scalr bar ranges because otherwise screenshot() doesn't work due pyvista/vtk
        # updating a share scalar bar across multiple frames. :(
        self._mappers[view_idx] = self.plotter.mapper
        new_range = self.plotter.mapper.scalar_range
        srange = self._scalar_bar_ranges.get(scalar, None)
        if srange is not None:
            new_range = (min(new_range[0], srange[0]), max(new_range[1], srange[1]))
        self._scalar_bar_ranges[scalar] = new_range

        return vtp

    def add_scalar_bar(self, view_idx, scalar_name, mapper):
        self.plotter.subplot(0, view_idx)

        # We want (1) a scalar bar displayed in each subplot and (2) the same bar range for the same scalar across subplots.
        # - we solve (1) by appending ' ' charater if the bar by that name already exists.
        # - we solve (2) by using the same mappers.
        fmt_scalar_name = f"{scalar_name}\n"  # '\n increases spacing between the title and color bar'
        if fmt_scalar_name in self.plotter.scalar_bars:
            fmt_scalar_name += " "

        # Updates the mesh color mapping tooo
        mapper.scalar_range = self._scalar_bar_ranges[scalar_name]

        add_bar = partial(
            self.plotter.add_scalar_bar,
            interactive=False,
            title_font_size=self.font_size(20),
            label_font_size=self.font_size(14),
            fmt="%10.4f",
            width=0.6,
            position_x=0.2,
            mapper=mapper,
        )

        bar = add_bar(fmt_scalar_name, mapper=mapper)
        bar.GetTitleTextProperty().SetLineSpacing(1.5)
        bar.SetTextPad(5)

    def update_info_text(self, idx, variable):
        views_text = "\n".join(
            [f"{self.views[i].value}: {self.views_info[i]}" for i in range(len(self.views))]
        )
        self.info_text = f"""

Run id: {self.dataset.run_id(idx)}
Manifest size: {len(self.dataset)} runs
Variable: {variable}
{views_text}
"""
        return self.info_text

    def validate_dataset(self):
        # fmt: off
        if ViewType.GroundTruth in self.views:
            if not self.fallback_to_geometry_files and not self.dataset.has_data_files():
                raise click.UsageError(
                    f"Cannot view ground truth, 'data_files' not in manifest."
                    f"Change the manifest or remove '{ViewType.GroundTruth.value}' from --views: {[v.value for v in self.views]}")
            elif (self.fallback_to_geometry_files and not self.dataset.has_data_files() and not self.dataset.has_geometry_files()):
                raise click.UsageError(
                    f"Cannot view ground truth, 'geometry_files' not in manifest."
                    f"Change the manifest or remove '{ViewType.GroundTruth.value}' from --views: {[v.value for v in self.views]}")

        if (ViewType.Predicted in self.views or ViewType.Error in self.views) and not self.dataset.has_predictions():
            raise click.UsageError(
                f"Cannot view predictions, 'predicted_file' not in manifest."
                f"Change the manifest or remove '{ViewType.Predicted.value}' and '{ViewType.Error.value}' "
                f"from --views: {[v.value for v in self.views]}")
        # fmt: on

    def start(self):
        self.validate_dataset()
        if self.interactive:
            self.update_plotter()
            self.plotter.show()

    def toggle_info_text(self):
        self.show_info_text = not self.show_info_text
        self.update_plotter()

    def toggle_help_text(self):
        self.show_help_text = not self.show_help_text
        self.update_plotter()

    def take_screenshot(self, idx, variable, screenshot_dir, width=None, height=None, filename_prefix=None):
        self.idx = idx
        self.set_variable(variable)

        # ensure unique filenames by using the run ID
        max_zeroes = 4
        run_id = self.dataset.run_id(self.idx)
        run_name = f"run{str(run_id).zfill(max_zeroes)}"
        # append filenames from the manifest so the user can more easily map back to data/geometry files
        if self.dataset.has_data_files():
            run_name += "_" + Path(self.dataset.data_files(self.idx, missing_ok=False)[0]).stem
        elif self.dataset.has_geometry_files():
            run_name += "_" + Path(self.dataset.geometry_files(self.idx, missing_ok=False)[0]).stem

        self.update_plotter()

        window_size = (width, height) if width and height else self.plotter.window_size
        filename_prefix = filename_prefix or ""
        screenshot_filepath = (
            screenshot_dir / f"{filename_prefix}{run_name}_{self.variable_filenames[self.variable_idx]}.png"
        )

        self.plotter.screenshot(screenshot_filepath, return_img=False, window_size=window_size)

        return screenshot_filepath.resolve()

    def font_size(self, base_font_size):
        base_resolution = 1920 * 1080
        current_resolution = self.plotter.window_size[0] * self.plotter.window_size[1]
        scaling_factor = (current_resolution / base_resolution) ** 0.5
        return int(base_font_size * scaling_factor)

    def text_position(self, base_position):
        window_size = self.plotter.window_size

        # Assuming a base resolution of 1920x1080
        base_resolution_x, base_resolution_y = 1920, 1080
        window_width, window_height = window_size

        # Calculate the aspect ratio of the base resolution and the window
        base_aspect_ratio = base_resolution_x / base_resolution_y
        window_aspect_ratio = window_width / window_height

        # Calculate the scaling factors for x and y positions
        x_scaling_factor = window_aspect_ratio / base_aspect_ratio
        y_scaling_factor = 1 / x_scaling_factor

        # Calculate the text position based on the scaling factors
        x_position = base_position[0] * x_scaling_factor
        y_position = base_position[1] * y_scaling_factor

        # Clamp the positions between 0 and 1
        x_position = max(0, min(x_position, 1))
        y_position = max(0, min(y_position, 1))

        return (x_position, y_position)


def take_screenshots(viewer, screenshot_settings):
    screenshot_dir = screenshot_settings.outdir
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    screenshot_settings.variable = (
        screenshot_settings.variable or viewer.dataset.surface_variables()[0]["name"]
    )

    variable = None
    var_name = screenshot_settings.variable
    var_dim = screenshot_settings.dimension
    for i, v in enumerate(viewer.variables):
        variable = v if v["name"] == var_name and v.get("dimension", var_dim) == var_dim else None
        if variable:
            break

    if variable is None:
        if var_dim is not None:
            raise click.UsageError(
                f"Variable '{var_name}[{var_dim}]' not in dataset. Available surface variables: {viewer.dataset.surface_variables()}"
            )
        else:
            raise click.UsageError(
                f"Variable '{var_name}' not in dataset. Available surface variables: {viewer.dataset.surface_variables()}"
            )

    width, height = screenshot_settings.image_size
    prefix = screenshot_settings.prefix

    for idx in range(len(viewer.dataset)):
        filename = viewer.take_screenshot(idx, variable, screenshot_dir, width, height, prefix)
        log.info(f"Screenshot written: {filename}")


def shorten_path(path: Path, max_length: int = 50, separator: str = "...") -> str:
    """
    Shortens the string representation of a Path by replacing the middle parts with an ellipsis,
    while ensuring that the parent directory is always kept.

    Args:
        path (Path): The path to be shortened.
        max_length (int, optional): The maximum length of the shortened path string. Default is 50.
        separator (str, optional): The string separator to use for the ellipsis. Default is "...".

    Returns:
        str: The shortened path string.
    """
    path_str = str(path)
    if len(path_str) <= max_length:
        return path_str

    parts = path.parts
    separator_length = len(separator)

    # Always keep the parent directory
    parent_dir = parts[-2]

    # Calculate the maximum length for start parts
    max_start_length = max_length - len(parent_dir) - separator_length - len(path.name)

    start_parts = parts[:-2]
    end_parts = [parent_dir, path.name]

    # Truncate start parts if needed
    truncated_start = str(Path(*start_parts[:max_start_length]))

    # Combine truncated start parts with parent directory and file name
    shortened_path = f"{truncated_start}{separator}{Path(*end_parts)}"

    # If the shortened path is still too long, further truncate the start parts
    while len(shortened_path) > max_length:
        max_start_length -= 1
        truncated_start = str(Path(*start_parts[:max_start_length]))
        shortened_path = f"{truncated_start}{separator}{Path(*end_parts)}"

    return shortened_path

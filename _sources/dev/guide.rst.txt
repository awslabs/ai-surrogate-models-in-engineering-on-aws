==============================
Code Structure and Concepts
==============================

This guide provides an overview of the MLSimKit codebase and helps you navigate through the different components and modules of the toolkit.

Code Structure
=================

The MLSimKit project is structured as follows:

.. code-block:: text

    src
    └── mlsimkit
       ├── common
       ├── conf
       ├── datasets
       ├── image
       └── learn

The main components of the project are:

- ``mlsimkit``: The core package containing the main codebase.
- ``common``: Common utilities and helper modules used across the toolkit.
- ``conf``: Configuration files and examples.
- ``datasets``: Sample datasets for testing and development.
- ``image``: Modules related to image processing and visualization.
- ``learn``: The main machine learning components, including models, data processing, and training pipelines.

``mlsimkit.common``
---------------------

The ``common`` module contains shared utilities and helper functions used across the toolkit:

- ``cli.py``: Framework for creating command-line tools with automated options and YAML config. For detailed information, see the :ref:`CLI Framework API documentation <api-cli>`.
- ``config.py``: Utilities primarily used by the CLI framework.
- ``logging.py``: Common logging setup including multi-gpu adapters.
- ``schema``: Pydantic schema definitions common to the CLI and logging utilities.

``mlsimkit.learn``
---------------------

The ``learn`` module is the core component of the MLSimKit package, offering functionality for various machine learning tasks related to physics-based simulations. It provides common tools for preprocessing data, training models, performing inference, and visualizing results. The model networks and use cases are located in this module. For detailed information, see the :doc:`learn` page.



``mlsimkit.datasets``
---------------------

The ``datasets`` directory includes very small sample datasets for testing and development purposes:

- ``ahmed-sample``: A sample dataset for Ahmed Body simulations.
- ``drivaer-sample``: A sample dataset for automotive aerodynamics simulations.

.. note::
    These sample datasets are provided for demonstration purposes only. For production use cases, you will need to use your own datasets or obtain publicly available datasets.


Schemas for CLI and Configuration
==================================

MLSimKit extensively uses the `Pydantic library <https://docs.pydantic.dev/latest/>`_ for encoding configuration and CLI inputs. Pydantic classes are called "models" and MLSimKit organizes these in ``schema`` subfolders within each module or submodule. These "schemas" are the interface between commands, configuration files, and code. 

The ``schema`` subfolders serve as a centralized location for defining the structured data models used by the corresponding module or submodule. These models are then used for various purposes, such as:

1. **Auto-Generated CLI Options**: The Pydantic models are used to define the configuration options for the CLI. By leveraging the ``mlsimkit.common.cli`` framework, these models can be seamlessly integrated with the CLI, allowing users to specify configurations through command-line options or configuration files.

2. **Validated Configuration Files**: The Pydantic models can be used to define the expected input parameters and return values for functions within the module. This improves code safety, readability and maintainability.

3. **Shared Across Code**: Pydantic models provide built-in validation and serialization capabilities, ensuring that the data used throughout the codebase adheres to the defined schemas. This helps catch errors early and promotes consistent data handling.


The ``schema`` subfolders typically contain one or more Python files, each defining a set of related Pydantic models. For example, in the ``mlsimkit.learn.kpi`` module, you might find the following structure:

.. code-block:: text

    src
    └── mlsimkit
       ├── ...
       └── learn
            ├── kpi
                ├── cli.py
                ├── ...
                ├── schema
                │   ├── inference.py
                │   ├── preprocessing.py
                │   └── training.py
                └── ...

In this example, the ``schema`` subfolder within the ``kpi`` module contains three files: ``preprocessing.py``, ``inference.py``, and ``training.py``. Each of these files defines the Pydantic models specific to the corresponding functionality (preprocessing, inference, and training, respectively).

For instance, ``kpi/schema/training.py`` defines the KPI-specific training options, while inheriting common training options from ``BaseTrainSettings``:

.. code-block:: python

    class TrainingSettings(BaseTrainSettings):
        train_manifest_path: Optional[str] = Field(None, description="Path to the train manifest")
        validation_manifest_path: Optional[str] = Field(None, description="Path to the validation manifest")

        output_kpi_indices: Optional[str] = Field(
            default=None,
            description="index(es) of desired KPIs to predict, separated by ',' (e.g. 0,2,3) (using all if None)",
        )
        message_passing_steps: int = Field(default=5, ge=0, description="number of message passing steps for MGN")
        hidden_size: int = Field(
            default=8, ge=1, description="size of the hidden layer in the multilayer perceptron (MLP) used in MGN"
        )
        dropout_prob: float = Field(default=0, ge=0, lt=1, description="probability of an element to be zeroed")
        pooling_type: PoolingType = Field(
            default=PoolingType.MEAN,
            description="Pooling type used in the MGN's model architecture",
        )
        loss_metric: LossMetric = Field(default=LossMetric.RMSE, description="loss metric")
        save_predictions_vs_actuals: bool = Field(
            default=True,
            description="save the plots showing the predictions vs the actuals for train and validation datasets",
        )

The CLI is auto-generated combining these schemas. For example, the snippet below is from ``mlsimkit-learn kpi train --help``:

.. code-block:: bash

   [MLSimKit] Learning Tools
    Package Version: 0.2.3.dev8+g0c39dac.d20240821
    Usage: mlsimkit-learn kpi train [OPTIONS]

    Options:
      --training-output-dir TEXT      path of the folder where training outputs
                                      and model checkpoints are saved
      --epochs INTEGER                Number of epochs. Default is low for
                                      quickstart experience. Higher number of
                                      epochs are required for accurate training.
                                      See user guide.  [default: 5]
      --batch-size INTEGER            Batch size determines how to group training
                                      data. Note the batch size is per process.
                                      For multi-GPU, this means you need enough
                                      training and validation data for all
                                      processes.  [default: 4]
      --seed INTEGER                  Random seed  [default: 0]
      --shuffle-data-each-epoch / --no-shuffle-data-each-epoch
                                      shuffle data every epoch  [default: shuffle-
                                      data-each-epoch]
      ...

The configuration can also be set from a YAML file. For example ``tutorials/kpi/ahmed/training.yaml``:

.. code-block:: yaml

    # KPI training configuration for Ahmed dataset
    output-dir: outputs/training    # all artifacts output to CWD/output e,g models, images, metrics

    log:
      prefix-dir: logs              # all logs go here
      config-file: logging.yaml     # tutorial-specific config

    kpi:
      manifest_uri: training.manifest

      train:
        output_kpi_indices: "0"
        epochs: 100
        opt:
          learning_rate: 0.003

      predict:
        # Manifest includes labels, we want to evaluate performance
        compare-groundtruth: true


This modular approach promotes well-defined interfaces shared across commands and code. See the :ref:`Creating Custom CLI commands guide <quickstart-cli-framework>` for a step-by-step walkthrough.


.. _dev-guide-manifests:

Manifest Files for Interfaces with Data Sets and Results
==========================================================

In MLSimKit, manifests play a crucial role in interfacing with data sets and organizing the various files and metadata associated with each simulation run and corresponding training results. A manifest is a JSON lines file that contains metadata and file references for each run in a dataset.

Purpose of Manifests
--------------------

Manifests serve several purposes in the MLSimKit workflow:

1. **Data Organization**: Manifests provide a structured way to organize and reference the files associated with each simulation run, such as geometry files (e.g., STL, VTK), data files containing simulation results, and other related files.

2. **Metadata Storage**: In addition to file references, manifests can store metadata and parameter values related to each simulation run. This metadata can include key performance indicators (KPIs), simulation settings, or any other relevant information.

3. **Data Preparation**: Manifests are used during the data preprocessing stage to keep track of the transformations and operations performed on the data, such as downsampling, mapping data to geometry files, or splitting the data into train, validation, and test sets.

4. **Interfacing with ML Components**: The machine learning components of MLSimKit, such as training and inference, rely on manifests to access the relevant data files and metadata for each simulation run.

By leveraging manifests, MLSimKit provides a flexible and extensible way to handle diverse data sets and simulation scenarios, while maintaining a consistent interface for the machine learning components.

User-generated Manifests vs. Internal Manifests
-----------------------------------------------

There are two types of manifests in the MLSimKit workflow:

1. **User-generated Manifests**: These manifests are created by users to describe their dataset and serve as the initial input to the MLSimKit pipeline. Users can generate these manifests using the ``mlsimkit-manifest`` command, specifying the directories containing simulation runs and the desired file patterns or data files to include.

2. **Internal Manifests**: As the data goes through various preprocessing steps, such as downsampling, mapping data to geometry files, or splitting into train/validation/test sets, MLSimKit generates internal manifests that represent the transformed state of the data. These internal manifests are used by the machine learning components (e.g., training, inference) and are typically stored in the output directory specified by the user.

The internal manifests contain additional information beyond what is present in the user-generated manifests, such as references to the preprocessed data files, split data sets, and any other metadata generated during the preprocessing steps.

Manifest Structure
------------------

A manifest is a JSON lines file, where each line represents a single simulation run. Each line is a JSON object that can contain the following keys:

- ``geometry_files``: A list of file paths or URIs referencing the geometry files (e.g., STL, VTK) associated with the simulation run.
- ``data_files``: A list of file paths or URIs referencing the data files (e.g., CSV, VTK) containing simulation results or other data associated with the run.
- Additional keys specific to the dataset or use case, such as ``kpi`` for KPI prediction, ``slices_uri`` for slice prediction or ``surface_variables`` for surface variable prediction.

During the preprocessing and machine learning stages, MLSimKit may add or modify keys in the internal manifests to include references to preprocessed data files, encoded representations, or any other information required for training and inference. The internal manifests are written and read by subsequent commands. 

The following is an example of a user manifest referencing the sample dataset (see ``tutorials/kpi/sample/training.manifest``):

.. code-block:: text

    {"geometry_files": ["datasets/drivaer-sample/downsampled_stls/run1-frontwheel_0.05perc_ds.stl", "datasets/drivaer-sample/downsampled_stls/run1-rearwheel_0.05perc_ds.stl", "datasets/drivaer-sample/downsampled_stls/run1_0.01perc_ds.stl"], "kpi": [0.3115]}
    {"geometry_files": ["datasets/drivaer-sample/downsampled_stls/run2-frontwheel_0.05perc_ds.stl", "datasets/drivaer-sample/downsampled_stls/run2-rearwheel_0.05perc_ds.stl", "datasets/drivaer-sample/downsampled_stls/run2_0.01perc_ds.stl"], "kpi": [0.31623]}
    {"geometry_files": ["datasets/drivaer-sample/downsampled_stls/run3-frontwheel_0.05perc_ds.stl", "datasets/drivaer-sample/downsampled_stls/run3-rearwheel_0.05perc_ds.stl", "datasets/drivaer-sample/downsampled_stls/run3_0.01perc_ds.stl"], "kpi": [0.31682]}
    {"geometry_files": ["datasets/drivaer-sample/downsampled_stls/run4-frontwheel_0.05perc_ds.stl", "datasets/drivaer-sample/downsampled_stls/run4-rearwheel_0.05perc_ds.stl", "datasets/drivaer-sample/downsampled_stls/run4_0.01perc_ds.stl"], "kpi": [0.26672]}
    {"geometry_files": ["datasets/drivaer-sample/downsampled_stls/run5-frontwheel_0.05perc_ds.stl", "datasets/drivaer-sample/downsampled_stls/run5-rearwheel_0.05perc_ds.stl", "datasets/drivaer-sample/downsampled_stls/run5_0.01perc_ds.stl"], "kpi": [0.27158]}
    {"geometry_files": ["datasets/drivaer-sample/downsampled_stls/run6-frontwheel_0.05perc_ds.stl", "datasets/drivaer-sample/downsampled_stls/run6-rearwheel_0.05perc_ds.stl", "datasets/drivaer-sample/downsampled_stls/run6_0.01perc_ds.stl"], "kpi": [0.27429]}
    {"geometry_files": ["datasets/drivaer-sample/downsampled_stls/run7-frontwheel_0.05perc_ds.stl", "datasets/drivaer-sample/downsampled_stls/run7-rearwheel_0.05perc_ds.stl", "datasets/drivaer-sample/downsampled_stls/run7_0.01perc_ds.stl"], "kpi": [0.27036]}


Manifests with Relative and Absolute Paths
------------------------------------------

In manifests, the file paths or URIs referencing geometry files, data files, or other resources can be specified as either relative or absolute paths. MLSimKit provides flexibility in handling these paths through the ``RelativePathBase`` enum defined in the ``mlsimkit.learn.manifest.schema`` module.

The ``RelativePathBase`` enum has the following options:

- ``CWD``: Relative paths are resolved against the current working directory.
- ``PackageRoot``: Relative paths are resolved against the root directory of the MLSimKit package installation.
- ``ManifestRoot``: Relative paths are resolved against the directory containing the manifest file itself.

When creating or processing manifests, users can specify the base directory for resolving relative paths by setting the ``manifest_base_relative_path`` option in the preprocessing settings. This option accepts values from the ``RelativePathBase`` enum.

For example, if a manifest contains a relative path like ``"geometry_files": ["data/run1.stl"]``, MLSimKit will resolve this path differently based on the ``manifest_base_relative_path`` setting:

- If ``manifest_base_relative_path`` is set to ``CWD``, the path will be resolved against the current working directory.
- If ``manifest_base_relative_path`` is set to ``PackageRoot``, the path will be resolved against the root directory of the MLSimKit package installation.
- If ``manifest_base_relative_path`` is set to ``ManifestRoot``, the path will be resolved against the directory containing the manifest file itself.

This flexibility allows users to organize their data sets in a way that suits their project structure and distribute manifests and data files together without relying on absolute paths.

Additionally, users can choose to use absolute paths in their manifests, and MLSimKit will respect those paths without any further resolution.

By providing this level of control over path resolution, MLSimKit aims to accommodate various data organization strategies and facilitate the integration of diverse data sets into the machine learning workflow.

.. note::

    Currently, MLSimKit supports networked file storage via the ``file://`` URLs in manifests. The intent is to support additional endpoints like S3 and HTTP in the future, enabling seamless integration with cloud storage and remote data sources.


.. _dev-guide-datasets:

Datasets in Code
================

In MLSimKit, the internal representation of manifests is stored as `Pandas DataFrames <https://pandas.pydata.org/docs/user_guide/dsintro.html>`_. A DataFrame is a 2D table where each row contains the metadata and file references for each simulation run, as well as any additional data generated during pipeline steps. 

The ``data.py`` file in each use case submodule (e.g., ``kpi``, ``surface``, ``slices``) acts as the interface between the training code and these internal Pandas DataFrames. It defines custom dataset interfaces that encapsulate the logic for loading and preprocessing the data from the manifests, ensuring that the data is properly formatted and accessible for the machine learning components.

The dataset interfaces in ``data.py`` are typically implemented as subclasses of PyTorch's ``torch.utils.data.Dataset`` or ``torch_geometric.data.Dataset`` classes, depending on the data type and requirements. These classes provide methods for accessing and manipulating the data stored in the Pandas DataFrames, as well as any necessary preprocessing steps.

For example, in the ``surface`` submodule, the ``SurfaceDataset`` class inherits from ``torch_geometric.data.Dataset`` and serves as the interface for handling surface variable prediction data:

.. code-block:: python

    class SurfaceDataset(torch_geometric.data.Dataset):
        def __init__(self, manifest, device="cuda"):
            super(SurfaceDataset, self).__init__(root=None, transform=None, pre_transform=None)
            self.device = device
            if isinstance(manifest, pd.DataFrame):
                self.manifest = manifest
            else:  # assume manifest is a filepath, will fail otherwise
                self.manifest = read_manifest_file(manifest)

    ...

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

    ...

In the __init__ method, the SurfaceDataset class accepts either a Pandas DataFrame or a file path to the manifest. If a file path is provided, it reads the manifest file into a Pandas DataFrame using the read_manifest_file function from the manifest module.

The SurfaceDataset class then provides various methods for accessing and manipulating the data stored in the manifest DataFrame, such as ``ptfile``, ``has_data_files``, ``has_geometry_files``, etc. These methods allow the training code to retrieve the relevant data files, geometry files, and other metadata associated with each simulation run.

By encapsulating the data loading and preprocessing logic within these dataset interfaces, developers can easily adapt or create new dataset interfaces to handle different types of data or introduce new data preprocessing techniques without modifying the core machine learning components.


Project Context and Command Chaining
====================================

.. note::
 
    Project Context is an important concept for pipelining tasks operating on the same datasets and training results. It automates many inputs-outputs across commands and functions.


MLSimKit leverages the concept of a "Project Context" to facilitate command chaining and persistent state management across different subcommands. The ``ProjectContext`` is a data class that stores relevant settings and outputs generated during the execution of one subcommand, making them available for subsequent subcommands within the same project.

The ``ProjectContext`` is defined in the ``mlsimkit.learn.common.schema.project`` module and is typically implemented as a subclass of the ``BaseProjectContext`` class provided by MLSimKit. Each use case submodule (e.g., ``kpi``, ``surface``, ``slices``) defines its own ``ProjectContext`` class tailored to its specific requirements.

Here's an example implementation of the ``ProjectContext`` class from the ``kpi`` submodule:

.. code-block:: python

    class ProjectContext(BaseProjectContext):
        """
        Persist outputs for chaining commands.
        """
        # original input manifest
        manifest_path: Optional[str] = None

        # working manifests
        train_manifest_path: Optional[str] = None
        validation_manifest_path: Optional[str] = None
        test_manifest_path: Optional[str] = None

        model_path: Optional[str] = None
        run_id: Optional[str] = None
        output_kpi_indices: Optional[str] = None

In this example, the ProjectContext class defines attributes for storing the input manifest path, the paths to the train, validation, and test manifests (generated during preprocessing), the path to the trained model, the run ID (for experiment tracking), and the selected KPI indices.

The ProjectContext instance is initialized and loaded within the subcommand functions defined in the cli.py file of each use case submodule. For example, in the kpi submodule:

.. code-block:: python

    @kpi.command()
    @mlsimkit.cli.options(PreprocessingSettings, dest="settings")
    @mlsimkit.cli.options(SplitSettings, dest="split_settings", help_group="Split Manifest")
    @click.option("--split-manifest/--no-split-manifest", is_flag=True, default=True)
    def preprocess(ctx: click.Context, settings: PreprocessingSettings, split_manifest: bool, split_settings: SplitSettings):
        project = ProjectContext.load(ctx)
        # ... (preprocess data and update the ProjectContext)
        project.save(ctx)

In this example, the preprocess subcommand loads the ProjectContext instance using ``ProjectContext.load(ctx)``, performs the necessary preprocessing steps, and updates the ProjectContext with the generated manifest paths. Finally, it persists the updated ProjectContext using ``project.save(ctx)``. Subsequent subcommands, such as train or predict, can then access the persisted values from the ProjectContext instance and use them as inputs or for other purposes.

The ProjectContext class supports multi-GPU commands via the Accelerate library from Hugging Face to ensure that the context is properly loaded and persisted across multiple processes during distributed training scenarios.

.. _dev-guide-dataset-interaction:

Programmatic Dataset Interaction
================================

While MLSimKit provides a comprehensive command-line interface (CLI) for various tasks, it also allows programmatic interaction with the datasets for advanced use cases or custom applications. This section demonstrates how to load and iterate through a dataset programmatically, using the ``SurfaceDataset`` class from the ``surface`` submodule as an example.

Prerequisites
-------------

Before we dive into the example, it's essential to understand the following concepts, which have been covered in previous sections:

- :ref:`Manifests <dev-guide-manifests>`: This section explains how manifests are used to organize and manage data files and metadata for each simulation run.
- :ref:`Datasets <dev-guide-datasets>`: This section introduces the dataset interfaces defined in the ``data.py`` file of each use case submodule, which act as bridges between the raw data and the machine learning components.

With these concepts in mind, let's explore how to programmatically interact with a dataset using the ``SurfaceDataset`` class.

Example: Visualization Application
----------------------------------

Suppose we want to create a visualization application that renders the predicted surface variables for each simulation run in the dataset. We can achieve this by leveraging the ``SurfaceDataset`` class and the ``Viewer`` class from ``learn/surface/visualize.py``.

Here's a simplified example:

.. code-block:: python

    from mlsimkit.learn.surface import data, visualize

    # Load the dataset from a manifest file
    manifest_path = "path/to/manifest.jsonl"
    dataset = data.SurfaceDataset(manifest_path)

    # Create a viewer instance
    viewer = visualize.Viewer(dataset, interactive=False)

    # Iterate over the dataset
    for idx in range(len(dataset)):
        # Get the run ID for the current index
        run_id = dataset.run_id(idx)

        # Check if predictions are available
        if dataset.has_predictions():
            predicted_file = dataset.predicted_file(idx)
            print(f"Rendering prediction for run {run_id}: {predicted_file}")
            # Add code to render the prediction using the viewer

        # Optionally, you can access other data components
        if dataset.has_data_files():
            data_files = dataset.data_files(idx)
            print(f"Data files for run {run_id}: {data_files}")

        if dataset.has_geometry_files():
            geometry_files = dataset.geometry_files(idx)
            print(f"Geometry files for run {run_id}: {geometry_files}")

        # Additional visualization or processing logic...

In this example, we first import the necessary components from the ``surface`` submodule. Then, we create an instance of the ``SurfaceDataset`` by providing the path to the manifest file.

Next, we create an instance of the ``Viewer`` class from ``visualize.py``, passing the ``SurfaceDataset`` instance and setting ``interactive=False`` for a non-interactive visualization.

We iterate over the dataset using a ``for`` loop and retrieve the run ID for the current index using ``dataset.run_id(idx)``. Within the loop, we check if predictions are available using ``dataset.has_predictions()`` and access the predicted file path using ``dataset.predicted_file(idx)``. You can then add code to render the prediction using the ``Viewer`` instance.

Additionally, the example demonstrates how to access other data components, such as data files (``dataset.has_data_files()`` and ``dataset.data_files(idx)``), and geometry files (``dataset.has_geometry_files()`` and ``dataset.geometry_files(idx)``).

This example showcases how you can programmatically interact with the dataset, access different components (e.g., predictions, data files, geometry files), and incorporate custom logic or visualization techniques based on your specific requirements.

By leveraging the dataset interfaces and the documented methods provided by the ``SurfaceDataset`` class, you can create custom applications or scripts that go beyond the built-in CLI functionality of MLSimKit.

Custom Model Saving and Loading 
====================================

.. note::

    Skip this section if you do NOT need to customize model code.

In MLSimKit, the ``ModelIO`` interface manages saving and loading of trained machine learning models, used for checkpointing and persisting the best model. This interface is implemented in the ``networks`` submodule and is designed to provide a consistent and reusable approach for persisting and retrieving model states across different use cases. This allows for the common training code across different use cases.

The ``ModelIO`` interface is typically defined within the network architecture modules, such as ``networks/mgn.py`` for the MeshGraphNet architecture. It encapsulates the logic for creating, saving, and loading models, ensuring a standardized approach across different use cases.

Here's an example of the ``ModelIO`` implementation for the MeshGraphNet architecture:

.. code-block:: python

    class ModelIO:
        def __init__(self, ...):
           # ...

        def new(self):
            return MeshGraphNet(...)

        def load(self, config):
            # Load model checkpoint and return the model, optimizer, and other relevant states

        def save(self, model, model_path, train_loss, validation_loss, optimizer, epoch):
            # Save the model checkpoint, including the model state, optimizer state, and other relevant information

The ``ModelIO`` interface must provide the following methods:

- ``new``: Creates a new instance of the model based on the provided configurations and graph shapes.
- ``load``: Loads a saved model checkpoint, returning the model, optimizer, and other relevant states.
- ``save``: Saves the model checkpoint, including the model state, optimizer state, and other relevant information.

The ``ModelIO`` interface is used within the use case submodules, such as ``kpi`` and ``surface``, to facilitate model creation, saving, and loading during the training and inference stages.

For example, in the ``training.py`` file of the ``kpi`` submodule, the ``run_train`` function creates an instance of ``ModelIO`` and utilizes its ``new`` and ``save`` methods:

.. code-block:: python

    def run_train(config, accelerator):
        # ...
        model_loader = mgn.ModelIO(
            config,
            data_scaler,
            graph_shape=(node_input_size, edge_input_size, num_classes),
            accelerator=accelerator,
        )

        model = model_loader.new()
        # ...
        model_loader.save(model, model_path, train_loss, validation_loss, optimizer, epoch)

Similarly, in the ``inference.py`` file of the ``surface`` submodule, the ``run_predict`` function uses the ``load`` method of ``ModelIO`` to load a trained model for inference:

.. code-block:: python

    def run_predict(config):
        # ...
        model, model_dict = load_model(config.model_path)
        # ...

The common training code leverages the ``ModelIO`` interface. This interface abstracts away the low-level details of handling model states and checkpoints, allowing the developer to focus on their use case implementation. 


Next Steps
----------

Explore how the core ML code is organized in :doc:`learn`.

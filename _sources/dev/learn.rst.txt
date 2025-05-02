====================================
Learning Module (``mlsimkit.learn``)
====================================

Before reading this, first understand :doc:`guide`. You are encouraged to complete the tutorials and read the user guides to understand the use case applications before modifying the code.

The ``learn`` module is the core component of the MLSimKit package, offering functionality for various machine learning tasks related to physics-based simulations. It follows a modular design, with a centralized ``common`` submodule containing shared utilities and helper functions used across the toolkit, and dedicated submodules for different use cases.

The ``common`` submodule contains shared components like the command-line interface (CLI) utilities, configuration management, logging utilities, mesh data processing, and a shared training loop. The ``training.py`` file within this submodule provides a generic training function that can be utilized by different use cases.

The ``networks`` submodule contains the implementations of various neural network architectures used in the toolkit, such as convolutional autoencoders and MeshGraphNets (a graph neural network architecture). These network architectures are designed to handle different types of data and serve different use cases.

Use Cases
------------------

There are dedicated submodules for each specific use case, such as Key Performance Indicator (KPI) prediction, surface variable prediction, and slice prediction (``kpi``, ``surface``, and ``slices``, respectively). These submodules encapsulate the functionality related to their respective use cases, including data loading and preprocessing, training, inference, and visualization (where applicable).

The ``learn`` module follows a consistent structure *by convention* across the different use case submodules (e.g., ``kpi``, ``surface``, ``slices``). Each submodule typically contains the following components:

- ``preprocessing.py``: This file contains utilities for preprocessing the data specific to the use case, such as processing mesh files, converting data formats, and generating manifests.
- ``data.py``: This file defines custom dataset interfaces for loading and preprocessing the data from manifests, tailored to the specific data types and requirements of the use case.
- ``schema/``: This directory contains Pydantic schema definitions for various settings and configurations related to the use case, such as preprocessing settings, training settings, and inference settings.
- ``training.py``: This file implements the training functionality for the use case, including the ``run_train`` function that leverages the shared training loop from the ``common`` submodule.
- ``inference.py``: This file provides utilities for performing inference and generating predictions using the trained models specific to the use case.
- ``cli.py``: This file serves as the command-line interface (CLI) entry point for the use case, allowing users to interact with the functionality through subcommands (e.g., ``mlsimkit-learn kpi preprocess``, ``mlsimkit-learn surface train``).

Within each of these components, the code is structured to encapsulate the logic and utilities specific to the use case, while leveraging shared utilities and abstractions from the ``common`` submodule for common tasks like configuration management, logging, and training.

Training Flow
------------------

The ``training.py`` file within each use case submodule follows a similar flow for training the respective machine learning models. Here's a general overview of the training process:

1. **Configuration Parsing**: The ``run_train`` function, typically the entry point for training, parses the configuration settings from the command-line arguments or configuration files. These settings may include hyperparameters, data paths, and other training-related options.

2. **Data Loading**: The function loads the training and validation data from the respective manifests using the dataset interfaces defined in ``data.py``. This step typically involves creating instances of the custom dataset classes and passing the appropriate manifest paths.

3. **Model and Optimizer Initialization**: Based on the configuration settings and the characteristics of the input data (e.g., node and edge input sizes for geometric data), the function initializes the appropriate model architecture. It also creates an instance of the ``ModelIO`` class from the ``networks`` submodule, which encapsulates the logic for creating, saving, and loading models. Additionally, an optimizer is initialized for the training process.

4. **Training Loop**: The core training process is typically delegated to the shared ``train`` function from the ``common.training`` module. This function handles the iterative training loop, computing the loss, backpropagation, and model updates. It also manages checkpointing, validation, and early stopping based on the provided configurations.

5. **Model Saving**: After the training process is complete, the ``run_train`` function saves the trained model using the ``ModelIO`` instance. This step typically involves saving the model state, optimizer state, and other relevant metadata to a file or directory specified in the configuration.

6. **Optional Steps**: Depending on the use case and configuration settings, additional steps may be performed after training, such as:

   - Generating predictions on the training and validation datasets, and saving the results for comparison or visualization purposes.
   - Logging training metrics and artifacts using MLflow or other experiment tracking tools.
   - Updating the internal manifests with the paths to the trained model or other generated artifacts.

While the specific implementation details may vary across different use cases, the general flow of the ``training.py`` file follows this structure, leveraging the shared utilities from the ``common`` submodule and the custom components defined within the use case submodule.


Programmatically Training and Predicting
-----------------------------------------

The ``learn`` module can be imported and used in your Python scripts or notebooks. For example, to perform KPI prediction, you can import the necessary components from the ``kpi`` submodule:

.. code-block:: python

    from mlsimkit.learn.kpi import preprocessing, training, inference

    # Preprocess data
    settings = PreprocessingSettings(...)
    working_manifest = preprocessing.run_preprocess(settings, project_root)

    # Train the model
    train_settings = TrainingSettings(...)
    accelerator = Accelerator(...)
    training.run_train(train_settings, accelerator)

    # Perform inference
    inference_settings = InferenceSettings(...)
    inference.run_predict(inference_settings, compare_groundtruth=True)

Similarly, for other tasks like surface variable prediction or slice prediction, you can import the relevant components from the corresponding submodules.

For more detailed usage examples and configuration options, refer to the user guides and tutorials provided in the MLSimKit documentation.


``mlsimkit.learn.common``
------------------------------------

The ``common`` module contains shared utilities and helper functions used across the learning module:

- ``cli.py``: CLI command entry for ``mlsimkit-learn``. Submodules add sub-commands e.g, ``mlsimkit-learn kpi ...``.
- ``config.py``: Configuration management and parsing utilities.
- ``logging.py``: Logging utilities and configuration.
- ``mesh.py``: Utilities for working with mesh data, including loading, converting, downsampling, and preprocessing mesh files.
- ``schema``: Pydantic schema definitions for various components, including optimizers, training settings, and project configuration.
- ``tracking.py``: Utilities for tracking and logging machine learning experiments using MLflow.
- ``training.py``: Utilities for training machine learning models, including a generic training loop that can be used across different use cases.
- ``utils.py``: General utility functions for tasks like calculating mean and standard deviation, obtaining optimizers and learning rate schedulers, and saving loss plots and prediction results.

``mlsimkit.learn.kpi``
------------------------------------

The ``kpi`` module contains components for Key Performance Indicator (KPI) prediction tasks:

- ``cli.py``: CLI command entry KPI prediction, including options for preprocessing, training, and inference.
- ``data.py``: Data loading and preprocessing utilities for KPI prediction, including the ``KPIDataset`` class for loading and handling KPI data.
- ``inference.py``: Inference functionality for KPI prediction, including utilities for getting predictions and saving prediction results.
- ``preprocessing.py``: Preprocessing utilities for KPI prediction, including functions for processing mesh files and adding preprocessed data to the manifest.
- ``schema``: Pydantic schema definitions for KPI prediction tasks, including preprocessing, inference, and training settings.
- ``training.py``: Training functionality for KPI prediction models, including the ``run_train`` function for training KPI models using the shared training loop.

``mlsimkit.learn.slices``
------------------------------------

The ``slices`` module contains components for slice prediction tasks:

- ``cli.py``: CLI command entry for slice prediction, including options for preprocessing, training, and inference.
- ``data.py``: Data loading and preprocessing utilities for slice prediction, including the ``SlicesDataset`` and ``GraphDataset`` classes for loading and handling slice data.
- ``inference.py``: Inference functionality for slice prediction, including utilities for running inference on autoencoders and the final prediction model.
- ``preprocessing.py``: Preprocessing utilities for slice prediction, including functions for loading and converting slice image data.
- ``schema``: Pydantic schema definitions for slice prediction tasks, including preprocessing, inference, and training settings.
- ``training.py``: Training functionality for slice prediction models, including the ``run_train_ae`` and ``run_train_mgn`` functions for training autoencoders and the final prediction model, respectively.

``mlsimkit.learn.surface``
------------------------------------

The ``surface`` module contains components for surface variable prediction tasks:

- ``cli.py``: CLI command entry for surface variable prediction, including options for preprocessing, training, inference, and visualization.
- ``data.py``: Data loading and preprocessing utilities for surface variable prediction, including the ``SurfaceDataset`` class for loading and handling surface data.
- ``inference.py``: Inference functionality for surface variable prediction, including utilities for running inference and converting predictions to VTK/PyVista formats.
- ``preprocessing.py``: Preprocessing utilities for surface variable prediction, including functions for processing mesh files, mapping data to STL files, and handling surface variables.
- ``schema``: Pydantic schema definitions for surface variable prediction tasks, including preprocessing, inference, training, and visualization settings.
- ``training.py``: Training functionality for surface variable prediction models, including the ``run_train`` function for training surface prediction models using the shared training loop.
- ``visualize.py``: Visualization utilities for surface variable prediction results, including the ``Viewer`` class for rendering and visualizing predictions.

``mlsimkit.learn.manifest``
------------------------------------

The ``manifest`` module provides utilities for working with data manifests:

- ``manifest.py``: This file contains the core functionality for creating, splitting, and processing manifests. It includes functions for:

  - Generating manifest entries from simulation "run" folders, extracting parameter values from data files and file paths from glob patterns.
  - Reading and writing manifest files in JSON lines format.
  - Creating and copying working manifests to avoid modifying the original user manifests.
  - Resolving file paths within manifests, handling relative and absolute paths.
  - Splitting manifests into train, validation, and test sets based on specified percentages and random seeds.

- ``cli.py``: This file provides the command-line interface (CLI) for working with manifests:

  - The ``create`` command generates a manifest file from a dataset, extracting parameter values from data files and file paths from glob patterns.
  - The ``split`` command splits an existing manifest file into train, validation, and test sets based on the provided split settings.

The ``manifest`` module plays a crucial role in managing and preprocessing data for the various machine learning tasks supported by MLSimKit. It ensures that the necessary data files and metadata are organized and accessible through the manifest files, which are then used by other components of the toolkit for tasks such as training and inference.


``mlsimkit.learn.networks``
------------------------------------

The ``networks`` module contains implementations of various neural network architectures used in the toolkit:

- ``autoencoder.py``: Implementation of convolutional autoencoders, including the ``ConvAutoencoder`` class and related utilities for training and inference.
- ``mgn.py``: Implementation of MeshGraphNets, a graph neural network architecture, including the ``MeshGraphNet`` class and related utilities for training and inference.
- ``schema``: Pydantic schema definitions for network architectures, including settings for convolutional autoencoders.


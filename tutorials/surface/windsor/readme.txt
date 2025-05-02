MLSimKit Tutorial - Surface Variable Prediction for WindsorML Dataset

**Setup**
1. Install MLSimkit
2. Ensure the dataset is available on a filesystem (mounted or downloaded). To help download, use ./download-dataset <target_dir>.

**Train and test the model**
3. Create a manifest with ground truth: ./run-create-manifest-training <dataset_dir>
4. Run the training and testing pipeline: ./run-training-pipeline

**Predict surface variable values on new geometries**
5. Create a manifest with geometry files: ./run-create-manifest-prediction <dataset_dir>
6. Run the prediction pipeline: ./run-prediction

Customize the training and prediction datasets by changing variables in ./run-create-manifest-* files.

Note you only need to run preprocessing once per dataset.

To remove outputs and logs, run ./run-clean

You may run individual commands manually. For example,

    cd <this directory>
    mlsimkit-learn --config prediction.yaml surface preprocess predict

See the User Guide for more details.

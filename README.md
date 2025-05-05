# ML for Simulation Toolkit 

ML for Simulation Toolkit (MLSimKit) provides engineers and designers a starting point for near real-time predictions of physics-based simulations using ML models. It enables engineers to quickly iterate on their designs and see immediate results rather than having to wait hours for a full physics-based simulation to run. 

The toolkit is a collection of commands and SDKs that insert into the traditional iterative design-simulate workflow (see diagram below), where it can take days to run a single cycle. Instead, the “train-design-predict” workflow becomes an additional choice to train models on simulated ground truth and then use the newly trained models for predictions on new designs in minutes.

## Getting started

### Documentation

[https://awslabs.github.io/ai-surrogate-models-in-engineering-on-aws/](https://awslabs.github.io/ai-surrogate-models-in-engineering-on-aws/)

### Environment setup
ML for Simulation Toolkit requires Python  >= 3.9, < 3.13. It is tested on Cuda >= 12.1. CPU-only usage is supported. 

Get a copy of the MLSimKit source into your environment.

It is optional but recommended to set up a virtual environment.
```shell
pip install virtualenv
cd mlsimkit
virtualenv .venv
source .venv/bin/activate
```

Install ``mlsimkit`` and dependencies via pip:
```shell
pip3 install .
```

You now have `mlsimkit-learn` installed:
```
% mlsimkit-learn --version
ML for Simulation Toolkit, version 0.1.dev47+g26aa53e.d20240313
```

### Build the documentation

Generate the HTML documentation locally for quickstart, tutorials, and user guides:

```shell
make docs
```

In a browser, open `docs/_build/html/index.html`. Proceed to the quickstart to learn more.

### Sample Tutorial

To train a sample model end-to-end, navigate to either `tutorials/kpi/sample` or `tutorials/slices/sample` and follow their `readme.txt`:
```
tutorials
├── kpi
│   ├── sample
│   └── windsor
└── slices
    ├── sample
    └── windsor
```

(get-started-installation)=
# Installation

It's recommended to install HintEval in a [virtual environment](https://docs.python.org/3/library/venv.html) using [Python 3.11.9](https://www.python.org/downloads/release/python-3119/). If you're not familiar with Python virtual environments, check out this [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). Alternatively, you can create a new environment using [Conda](https://anaconda.org/anaconda/conda).

### Set up the virtual environment

First, create and activate a virtual environment with Python 3.11.9:

```bash
conda create -n hinteval_env python=3.11.9 --no-default-packages
conda activate hinteval_env
```

### Install PyTorch 2.4.0

You'll need PyTorch 2.4.0 for HintEval. Refer to the [PyTorch installation page](https://pytorch.org/get-started/previous-versions/) for platform-specific installation commands. If you have access to GPUs, it's recommended to install the CUDA version of PyTorch, as many of the evaluation metrics are optimized for GPU use.

### Install HintEval

Once PyTorch 2.4.0 is installed, you can install HintEval via pip:

```bash
pip install hinteval
```

For the latest features, you can install the most recent version from the main branch:

```bash
pip install git+https://github.com/DataScienceUIBK/HintEval
```

Now that HintEval is installed, you can create a [synthetic dataset](get-started-testset-generation) using your own questions and answers. If you already have a dataset, learn how to [evaluate it](get-started-evaluation) with HintEval.
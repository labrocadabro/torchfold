# TorchFold: Dynamic Batching Library for PyTorch

## Project Overview

TorchFold is a library for dynamic batching in PyTorch, designed to optimize computations by batching operations dynamically. Inspired by TensorFlow Fold, it provides a simple interface to enhance performance in neural network computations, particularly for tasks with variable-sized inputs such as tree-structured data or recursive computations.

### Main Purpose and Problems Solved
TorchFold addresses the challenge of efficiently processing variable-sized or recursive data structures in neural networks. Traditional batching methods often struggle with such data, leading to inefficient computation. TorchFold solves this by dynamically batching operations, allowing for optimized execution on GPUs and significantly improving performance in scenarios like natural language processing tasks (e.g., tree-structured models for sentence parsing).

### Key Features and Benefits
- **Dynamic Batching**: Automatically batches operations for variable-sized inputs, reducing computational overhead.
- **Simple Interface**: Easily integrate with existing PyTorch models by replacing direct calls to neural network modules with `fold.add()` calls.
- **Performance Optimization**: Leverages dynamic computation graphs to execute batched operations efficiently, especially beneficial for recursive or tree-based computations.
- **Debugging Support**: Includes an `Unfold` class for immediate computation during debugging, bypassing dynamic batching when needed.

TorchFold is ideal for researchers and developers working on complex neural network architectures that require efficient handling of non-uniform data structures.

## Getting Started, Installation, and Setup

### Quick Start Guide

TorchFold provides dynamic batching for PyTorch with a simple interface. Here's how to quickly get started:

1. Install TorchFold using pip:
   ```bash
   pip install torchfold
   ```
2. Use the `torchfold.Fold()` class to dynamically batch computations. Replace direct calls to `nn` modules with `f.add()` calls.
3. Example usage:
   ```python
   import torchfold
   f = torchfold.Fold()
   
   def dfs(node):
       if is_leaf(node):
           return f.add('leaf', node)
       else:
           prev = f.add('init')
           for child in children(node):
               prev = f.add('child', prev, child)
           return prev

   class Model(nn.Module):
       def __init__(self, ...):
           ...

       def leaf(self, leaf):
           ...

       def child(self, prev, child):
           ...

   res = dfs(my_tree)
   model = Model(...)
   f.apply(model, [[res]])
   ```

For more detailed examples and advanced usage, refer to the [Usage Examples](#usage-examples) section.

### Installation and Setup

#### Prerequisites
- Python 3.x
- PyTorch (must be installed separately as it is not listed as a dependency in the setup file)

#### Installation Steps
1. Install TorchFold via pip (recommended):
   ```bash
   pip install torchfold
   ```
   Alternatively, you can install from source by cloning the repository and running:
   ```bash
   python setup.py install
   ```
2. Ensure PyTorch is installed in your environment. If not, install it following the instructions on the [PyTorch official website](https://pytorch.org/get-started/locally/).

#### Platform-Specific Instructions
- TorchFold is platform-independent as long as Python and PyTorch are supported on your system. Ensure your PyTorch installation matches your operating system and hardware (CPU/GPU).

#### Development Setup
If you plan to contribute or modify the source code:
1. Clone the repository:
   ```bash
   git clone https://github.com/nearai/torchfold.git
   cd torchfold
   ```
2. Install in development mode:
   ```bash
   python setup.py develop
   ```

#### Production Build
TorchFold is a library and does not require a separate production build process. Once installed via pip or from source, it is ready to be used in production environments as part of your PyTorch projects.

## Features / Capabilities

This section highlights the primary features and capabilities of TorchFold, a library designed for dynamic batching in PyTorch.

### Core Features

- **Dynamic Batching**: TorchFold enables dynamic batching of computations in PyTorch, optimizing performance by grouping operations efficiently. This is analogous to TensorFlow Fold but implemented with a simpler interface for PyTorch users.
- **Simplified Interface**: Replace direct calls to neural network modules with `f.add('function_name', arguments)` to construct an optimized computation graph. Execute the graph with `f.apply(model, nodes)` to dynamically batch and process data.
- **Support for Recursive Computations**: Ideal for processing hierarchical or recursive data structures such as trees, TorchFold handles operations like depth-first search (DFS) over tree structures with ease.
- **Flexible Node Operations**: Supports splitting results into multiple values using `node.split(num)` and disabling batching for specific nodes with `node.nobatch()`.
- **CUDA Support**: Offers optional CUDA acceleration for GPU-based computations, configurable with `fold.cuda()`.
- **Debugging Mode with Unfold**: Provides an `Unfold` class for immediate computation during debugging, bypassing the batching mechanism to simplify error tracking.

### Example Use Case

TorchFold is particularly useful for natural language processing tasks involving tree structures. An included example demonstrates its application in a Simplified Parsing Neural Network (SPINN) model for the Stanford Natural Language Inference (SNLI) dataset:

- **Tree LSTM Implementation**: The example showcases how to use TorchFold to encode tree structures with a Tree LSTM, dynamically batching operations for efficiency.
- **Comparison with Regular Encoding**: The code provides both a folded (batched) and a regular (non-batched) implementation, highlighting performance improvements with TorchFold.

```python
f = torchfold.Fold()

def dfs(node):
    if is_leaf(node):
        return f.add('leaf', node)
    else:
        prev = f.add('init')
        for child in children(node):
            prev = f.add('child', prev, child)
        return prev

class Model(nn.Module):
    def __init__(self, ...):
        ...

    def leaf(self, leaf):
        ...

    def child(self, prev, child):
        ...

res = dfs(my_tree)
model = Model(...)
f.apply(model, [[res]])
```

This example illustrates how TorchFold simplifies the handling of complex recursive computations by managing batching internally, allowing developers to focus on model logic rather than optimization details.

## Usage Examples

This section provides examples of how to use the project for training a SPINN model on the SNLI dataset. The repository includes a sample implementation that showcases the processing of tree-structured data using TorchFold.

#### Running the SPINN Example

The `spinn-example.py` script demonstrates how to implement and train a SPINN model for natural language inference tasks using the SNLI dataset. Follow these steps to run the example:

1. **Prepare Your Environment**: Ensure you have the necessary dependencies installed, including PyTorch, TorchText, and TorchFold. Refer to the Installation section for setup instructions.

2. **Navigate to the Examples Directory**: Change to the directory containing the example script:
   ```bash
   cd examples/snli
   ```

3. **Run the Example Script**: Execute the script to train the model. You can specify whether to use TorchFold for dynamic batching and whether to enable CUDA if available:
   ```bash
   python spinn-example.py --fold --batch_size 128
   ```
   - `--fold`: Enables the use of TorchFold for dynamic batching of tree structures.
   - `--batch_size`: Sets the batch size for training (default is 128).
   - `--no-cuda`: Disables CUDA even if it's available (remove this flag to use GPU if supported).

4. **Monitor Training Progress**: The script will output the average time per iteration every 10 iterations, helping you monitor the training performance.

#### Understanding the Example

The example script implements a SPINN model using a TreeLSTM for encoding tree-structured data. It supports two modes of operation:
- **Regular Encoding**: Processes trees without TorchFold, handling each tree individually.
- **Fold Encoding**: Uses TorchFold to dynamically batch operations on trees, potentially improving performance.

The script downloads and processes the SNLI dataset using TorchText, builds a vocabulary, and trains the model over 10 epochs using the Adam optimizer and CrossEntropyLoss.

#### Customizing the Example

To adapt this example for your own dataset or model architecture:
- Modify the `Tree` class to parse your data into tree structures.
- Adjust the `SPINN` class to change the model architecture, such as the number of units or layers.
- Update hyperparameters like `batch_size` or learning rate in the `main()` function.

## Project Structure

This section outlines the layout of the project, detailing the key directories and files present in the repository.

### Directory and File Overview

- **examples/**: Contains example scripts demonstrating the usage of the project. Currently includes:
  - **snli/spinn-example.py**: An example script related to the SNLI dataset, likely showcasing a specific implementation or model.

- **torchfold/**: The core module of the project, containing the main source code.
  - **__init__.py**: Initialization file for the `torchfold` package.
  - **torchfold.py**: Primary source file for the `torchfold` module, likely containing the main functionality or implementation.
  - **torchfold_test.py**: Test file for the `torchfold` module, used for validating the functionality of the codebase.

- **.gitignore**: Specifies intentionally untracked files to ignore in version control.
- **LICENSE**: Contains the licensing information for the project.
- **README.md**: The main documentation file for the project, providing an overview and instructions.
- **README_Prometheus.md**: An additional README file, possibly specific to Prometheus or a related context.
- **logo.jpg**: An image file, likely used for branding or documentation purposes.
- **setup.py**: A setup script for installing the project as a Python package.

## Technologies Used

This project leverages the following major technologies, frameworks, libraries, SDKs, and tools:

- **PyTorch**: A deep learning framework used as the core technology for implementing dynamic batching and neural network computations in TorchFold.
- **Python**: The primary programming language used for development.
- **setuptools**: Utilized for packaging and distributing the TorchFold library, as seen in the setup.py configuration.

## Additional Notes

### Performance Considerations

TorchFold is designed to optimize computations through dynamic batching, which can significantly improve performance for operations on variable-sized inputs like trees or graphs. However, the effectiveness of batching depends on the structure of your data and the operations being performed. For highly irregular structures, the overhead of dynamic batching might outweigh the benefits. It is recommended to benchmark your specific use case with and without TorchFold to determine the performance impact.

### Debugging with Unfold

For debugging purposes, TorchFold provides an `Unfold` class that serves as a drop-in replacement for `Fold`. Unlike `Fold`, which delays computation to batch operations, `Unfold` executes computations immediately. This can be useful for tracing through the computation step-by-step to identify issues in your model or data processing pipeline. To use it, simply replace `Fold` with `Unfold` in your code and pass your neural network module to its constructor.

### Limitations

- **Argument Types**: All arguments passed to `Fold.add()` must be of type `Tensor`, `Variable`, `int`, or a `Fold.Node`. Mixing incompatible types or using unsupported types will result in errors.
- **No-Batch Constraints**: When using the `.nobatch()` method on a node, ensure that it is not combined with other nodes in a way that violates batching constraints, as this will raise an error during computation.
- **CUDA Support**: While TorchFold supports CUDA for accelerating computations on GPUs, you must explicitly enable it by calling `.cuda()` on your `Fold` instance. Ensure your model and data are also CUDA-compatible to avoid runtime errors.

### Community and Support

TorchFold is an open-source project maintained by Illia Polosukhin and NEAR Inc. For additional resources, refer to the blog post on dynamic batching with PyTorch at [near.ai/articles](http://near.ai/articles/2017-09-06-PyTorch-Dynamic-Batching/). If you encounter issues or have questions, consider contributing to the project or opening an issue on the [GitHub repository](https://github.com/nearai/torchfold).

### Citation

If you use TorchFold in your research or publications, please cite it as follows:

```
@misc{illia_polosukhin_2018_1299387,
  author       = {Illia Polosukhin and Maksym Zavershynskyi},
  title        = {nearai/torchfold: v0.1.0},
  month        = jun,
  year         = 2018,
  doi          = {10.5281/zenodo.1299387},
  url          = {https://doi.org/10.5281/zenodo.1299387}
}
```

## Contributing

Thank you for your interest in contributing to TorchFold! We welcome contributions from the community to help improve and expand this project. Below are some guidelines to ensure a smooth contribution process.

### How to Contribute

1. **Fork the Repository**: Start by forking the repository to your own GitHub account.
2. **Clone the Repository**: Clone the forked repository to your local machine to start working on your changes.
3. **Create a Branch**: Create a new branch with a descriptive name related to the feature or bug fix you are working on.
4. **Make Your Changes**: Implement your changes, ensuring that your code adheres to the guidelines outlined below.
5. **Test Your Changes**: Make sure to test your changes thoroughly to avoid introducing bugs.
6. **Commit Your Changes**: Write clear and concise commit messages that describe the purpose of your changes.
7. **Push Your Changes**: Push your branch to your forked repository on GitHub.
8. **Submit a Pull Request**: Create a pull request from your branch to the main repository. Provide a detailed description of your changes and the motivation behind them.

### Contribution Guidelines and Requirements

- **Code Style**: Please follow the PEP 8 style guide for Python code to maintain consistency across the codebase. Use tools like `flake8` or `pylint` to check your code style before submitting.
- **Testing**: Ensure that your contributions include appropriate tests to validate the functionality. We aim to maintain a high level of code coverage, so please add or update tests as necessary.
- **Documentation**: If your contribution introduces new features or changes existing functionality, update the relevant documentation to reflect these changes.
- **Issue Tracking**: If your contribution addresses a specific issue, reference the issue number in your pull request description.
- **License**: By contributing to this project, you agree that your contributions will be licensed under the same license as the repository.

We review all contributions and may provide feedback or request changes before merging. Thank you for helping to make TorchFold better!

## License

This project is licensed under the Apache License Version 2.0. For full details, please see the [LICENSE](./LICENSE) file in the repository.
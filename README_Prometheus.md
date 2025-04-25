# TorchFold: Dynamic Batching for PyTorch Deep Learning Models

## Project Overview

This project, **TorchFold**, is a PyTorch library designed to enable dynamic batching for deep learning models. Its main purpose is to optimize computation by grouping operations into batches dynamically during the forward pass, which can significantly improve performance when dealing with variable-sized inputs or complex computational graphs.

### Key Purpose and Problems Solved
TorchFold addresses the challenge of efficiently processing data with varying sizes or structures in neural networks. Traditional batching methods often require padding or other workarounds that can waste computational resources. TorchFold solves this by allowing operations to be batched dynamically, ensuring optimal use of resources and faster training and inference times.

### Key Features and Benefits
- **Dynamic Batching**: Automatically groups operations into batches at runtime, adapting to the structure of the input data.
- **Efficiency**: Reduces computational overhead by minimizing unnecessary padding or redundant calculations.
- **Flexibility**: Works seamlessly with PyTorch, supporting a wide range of neural network architectures.
- **Debugging Support**: Includes an `Unfold` class for debugging, allowing immediate computation for easier tracing and validation.
- **GPU Support**: Offers CUDA compatibility for accelerated computation on GPU hardware.

TorchFold is particularly beneficial for researchers and developers working on models with non-uniform input sizes, such as natural language processing tasks or graph-based neural networks, providing a robust tool to enhance performance and simplify implementation.

## Getting Started, Installation, and Setup

### Quick Start Guide

TorchFold is a library for dynamic batching in PyTorch, providing a simple interface to optimize computations. Here's how to quickly get started:

1. **Install TorchFold**: Use pip to install the package (detailed instructions in the Installation section below).
   ```bash
   pip install torchfold
   ```
2. **Basic Usage**: Replace direct calls to `nn` module functions with `f.add()` calls to enable dynamic batching.
3. **Example Code**: Below is a simple example to demonstrate usage with a tree structure.
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

For more detailed examples, refer to the `examples` directory in the repository.

### Installation

TorchFold can be easily installed using pip. Follow these steps to set up the library on your system:

1. **Install via pip** (recommended):
   ```bash
   pip install torchfold
   ```

2. **Dependencies**: TorchFold requires PyTorch to be installed. If it's not already installed, you can install it via pip as well. Visit the [PyTorch official website](https://pytorch.org/get-started/locally/) for platform-specific installation instructions.
   ```bash
   pip install torch
   ```

3. **Verify Installation**: After installation, you can verify it by importing the library in Python:
   ```python
   import torchfold
   print(torchfold.__version__)
   ```

### Setup for Development

If you want to contribute to TorchFold or run it from source, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nearai/torchfold.git
   cd torchfold
   ```

2. **Install in Development Mode**:
   Use pip to install the package in editable mode so changes to the source code are immediately reflected.
   ```bash
   pip install -e .
   ```

3. **Running Tests**: Currently, there are no specific test scripts provided in the repository. You can manually test the functionality using the provided example scripts in the `examples` directory.

### Building a Production Release

TorchFold is distributed as a pip package, and there are no specific build steps for production release beyond packaging it via `setup.py`. To create a distribution package:

1. **Ensure `setuptools` and `wheel` are installed**:
   ```bash
   pip install setuptools wheel
   ```

2. **Create a Distribution**:
   Run the following command in the root directory of the repository to create a source distribution and wheel:
   ```bash
   python setup.py sdist bdist_wheel
   ```

3. **Upload to PyPI** (for maintainers):
   If you have the necessary permissions, you can upload the package to PyPI using `twine`.
   ```bash
   pip install twine
   twine upload dist/*
   ```

### Platform-Specific Instructions

TorchFold is platform-agnostic as it is a Python library built on top of PyTorch. However, ensure that your PyTorch installation is compatible with your operating system and hardware (e.g., GPU support). Refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for detailed instructions on installing PyTorch for your specific platform (Windows, macOS, Linux).

## Project Structure

This section outlines the layout of the repository, detailing the key directories and files that constitute the project.

### Directory and File Layout
- **torchfold/**: The core package directory containing the main implementation of the TorchFold library.
  - `__init__.py`: Initialization file for the TorchFold package.
  - `torchfold.py`: Main source code file for the TorchFold library, likely containing the primary logic and functionality.
  - `torchfold_test.py`: Test file for the TorchFold library, used for validating the functionality of the implementation.
- **examples/**: Directory containing example usage of the library.
  - `snli/spinn-example.py`: An example script demonstrating the usage of TorchFold, possibly in the context of the Stanford Natural Language Inference (SNLI) dataset with a SPINN (Stack-augmented Parser-Interpreter Neural Network) model.
- **.gitignore**: Standard Git ignore file specifying intentionally untracked files to ignore.
- **LICENSE**: File containing the licensing information for the project.
- **README.md**: The main documentation file providing an overview and instructions for the project.
- **logo.jpg**: An image file, likely used for branding or illustrative purposes in the documentation.
- **setup.py**: A setup script for installing the TorchFold library as a Python package.

## Technologies Used

- **PyTorch**: A deep learning framework used for tensor computation and dynamic neural networks. This project, named TorchFold, focuses on dynamic batching with PyTorch, providing a simple interface for optimized computation.

## Additional Notes

### Performance Considerations

TorchFold is designed to optimize computations through dynamic batching. When using this library, be aware that the performance benefits are most noticeable when dealing with variable-sized or tree-structured data where traditional batching methods are inefficient. The dynamic batching mechanism groups operations to minimize redundant computations, which can significantly reduce training and inference times for certain models, such as those processing tree-structured data like in natural language processing tasks.

### Debugging with Unfold

For debugging purposes, TorchFold provides an `Unfold` class. This class serves as a replacement for `Fold` and performs computations immediately rather than batching them. This can be particularly useful when you need to step through the computation graph or inspect intermediate results without the complexity of dynamic batching. To use it, initialize `Unfold` with your neural network module and use the `add` method similarly to `Fold`.

### CUDA Support

TorchFold supports CUDA for GPU acceleration. You can enable CUDA by calling the `cuda()` method on a `Fold` or `Unfold` instance. Ensure that your PyTorch installation is configured for CUDA and that your hardware supports it to take advantage of faster computation speeds.

### Example Use Case

An example provided in the repository demonstrates the application of TorchFold in a SPINN (Stack-augmented Parser-Interpreter Neural Network) model for natural language inference tasks. This example illustrates how to structure tree-based data processing using dynamic batching to improve efficiency. While this is a simplified model, it showcases the potential of TorchFold in handling complex data structures.

### Citation

If you use TorchFold in your research or projects, please cite the repository as follows:

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

### Limitations

- **Argument Types**: All arguments passed to the `add` method in `Fold` must be of type `Tensor`, `Variable`, `int`, or `Node`. Mixing incompatible types will result in errors.
- **Batch Processing**: While dynamic batching is powerful, it requires careful design of your computation graph to ensure operations can be batched effectively. Incorrect usage might lead to inefficiencies or errors during execution.

This section provides supplementary information to help users maximize the utility of TorchFold in their projects. For further details or specific issues, consider exploring the codebase or reaching out to the community.

## Contributing

We welcome contributions from the community to help improve TorchFold. Whether it's bug fixes, new features, or documentation updates, your input is valuable.

### How to Contribute
1. **Fork the Repository**: Start by forking the repository to your own GitHub account.
2. **Clone the Repository**: Clone the forked repository to your local machine.
3. **Create a Branch**: Create a new branch for your changes. Use a descriptive name related to the feature or bug you're working on.
4. **Make Your Changes**: Implement your changes or fixes in the codebase.
5. **Test Your Changes**: Ensure your changes do not break existing functionality. Add tests if applicable.
6. **Commit Your Changes**: Commit your changes with a clear and descriptive commit message.
7. **Push to GitHub**: Push your branch to your forked repository on GitHub.
8. **Create a Pull Request**: Open a pull request from your branch to the main repository's `main` branch. Provide a detailed description of your changes.

### Contribution Guidelines
- **Code Style**: Follow Python PEP 8 style guidelines for code formatting and naming conventions.
- **Testing**: Ensure that new features or bug fixes include appropriate tests to maintain code reliability. Run existing tests to verify that your changes do not introduce regressions.
- **Documentation**: Update or add documentation for any new features or changes to existing functionality.
- **Issue Tracking**: If you're addressing a specific issue, reference the issue number in your pull request description.

Thank you for contributing to TorchFold and helping make it better!

## License

This project is licensed under the Apache License, Version 2.0. You can view the full license text in the [LICENSE](./LICENSE) file.
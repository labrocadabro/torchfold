# TorchFold: Dynamic Batching for PyTorch Neural Networks

## Project Overview

TorchFold is a PyTorch library designed to implement dynamic batching, providing a simple and efficient way to optimize computations in neural networks. Inspired by TensorFlow Fold, this library allows developers to handle variable-sized inputs by dynamically batching operations, which is particularly useful in scenarios involving recursive or tree-structured data.

### Main Purpose and Problems Solved
The primary purpose of TorchFold is to address the challenge of processing data with varying sizes or structures in neural network models. Traditional batching methods often require fixed-size inputs, which can be inefficient or impractical for tasks like natural language processing (NLP) or graph-based learning. TorchFold solves this by enabling dynamic batching, allowing computations to be grouped and executed efficiently without the need for padding or restructuring data.

### Key Features and Benefits
- **Dynamic Batching**: Automatically groups computations for efficiency, reducing memory usage and speeding up processing.
- **Simple Interface**: Integrates seamlessly with PyTorch by replacing direct calls to neural network modules with a straightforward `add` method.
- **Optimized Execution**: Constructs optimized computation graphs and executes them dynamically, ensuring minimal overhead.
- **Versatility**: Supports a wide range of applications, especially those involving recursive or hierarchical data structures, such as tree-based models in NLP.

TorchFold is a powerful tool for developers working with complex data structures in PyTorch, offering both performance improvements and ease of use.

## Getting Started, Installation, and Setup

### Quick Start Guide

To quickly get started with `torchfold`, follow these steps:

1. **Clone the Repository**: If you haven't already, clone this repository to your local machine.
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. **Install the Package**: Use pip to install the `torchfold` package locally.
   ```bash
   pip install .
   ```
3. **Run an Example**: Try running the provided example to see `torchfold` in action.
   ```bash
   python examples/snli/spinn-example.py
   ```

For detailed installation instructions and dependencies, refer to the section below.

### Installation Instructions

#### Prerequisites
- Python 3.x
- PyTorch (version compatible with your system)
- pip (Python package installer)

#### Steps to Install
1. **Clone the Repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. **Install Dependencies**: Ensure you have PyTorch installed. If not, install it via pip or refer to the official PyTorch website for platform-specific instructions.
   ```bash
   pip install torch
   ```
3. **Install `torchfold`**: Install the package locally using pip.
   ```bash
   pip install .
   ```

#### Platform-Specific Instructions
- **Linux/macOS/Windows**: The installation process is generally the same across platforms as long as Python and PyTorch are supported. For PyTorch installation, you might need to choose the correct version based on your CUDA version (if using GPU). Visit the [PyTorch official installation page](https://pytorch.org/get-started/locally/) for detailed instructions.

### Setup for Development

If you plan to contribute or modify the code, follow these additional steps:

1. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
2. **Install in Development Mode**: This allows changes to the code to be reflected without reinstalling.
   ```bash
   pip install -e .
   ```
3. **Run Tests**: Ensure everything is set up correctly by running the provided tests.
   ```bash
   python torchfold/torchfold_test.py
   ```

### Building for Production

Since `torchfold` is a Python package, building for production typically involves packaging it for distribution or direct installation on a production server. To create a distributable package:

1. **Build the Package**:
   ```bash
   python setup.py sdist bdist_wheel
   ```
2. **Distribute or Install**: Upload the generated files in the `dist/` folder to a package repository like PyPI, or install directly on your production environment using pip.
   ```bash
   pip install dist/torchfold-<version>.tar.gz
   ```

For deploying to a production environment, ensure that the target system has all dependencies (like PyTorch) installed and compatible with the package version.

## Project Structure

This section provides an overview of the key directories and files in the repository to help you navigate the codebase effectively.

#### Key Directories and Files

- **`torchfold/`**: The core directory containing the main implementation of the TorchFold library.
  - **`torchfold.py`**: The primary source file for the TorchFold module, which likely contains the core functionality for dynamic batching in PyTorch.
  - **`torchfold_test.py`**: Contains test cases for the TorchFold library to ensure its functionality.
  - **`__init__.py`**: Initializes the `torchfold` package, making it importable in Python.

- **`examples/`**: A directory with example usage of the library.
  - **`snli/spinn-example.py`**: An example script demonstrating the application of TorchFold, possibly for the Stanford Natural Language Inference (SNLI) dataset using a SPINN model.

- **`setup.py`**: A setup script for installing the TorchFold library as a Python package.

- **`.gitignore`**: Specifies intentionally untracked files to ignore in the repository.

- **`LICENSE`**: Contains the licensing information for the project.

- **`logo.jpg`**: A logo image file, likely used for branding or documentation purposes.

- **`README.md`**: The main documentation file providing an overview and instructions for the project.

## Additional Notes

### Key Features and Limitations

TorchFold is designed to simplify dynamic batching in PyTorch, inspired by TensorFlow Fold. Below are some key points to note about its functionality and constraints based on the current codebase:

- **Dynamic Batching**: TorchFold optimizes computation by dynamically batching operations, which can significantly improve performance when dealing with variable-sized inputs like trees or graphs. This is evident in the `Fold` class implementation, which constructs batched computations via the `add` method and executes them with `apply`.

- **Ease of Use**: The library provides a straightforward interface. Users can replace direct calls to neural network modules with `Fold.add()` calls to enable dynamic batching, as shown in the example code and the SNLI SPINN model in the examples directory.

- **Support for Complex Structures**: The codebase includes support for tree-like structures, demonstrated in the `examples/snli/spinn-example.py` file, which implements a TreeLSTM for natural language inference tasks using the SNLI dataset. This shows TorchFold's applicability to recursive and hierarchical data processing.

- **Debugging Mode with Unfold**: Alongside the `Fold` class, TorchFold offers an `Unfold` class for debugging purposes. This class performs computations immediately without batching, allowing developers to trace and verify the logic of their models step-by-step.

- **CUDA Support**: The library supports GPU acceleration, with options to enable CUDA processing in both `Fold` and `Unfold` classes, catering to high-performance computing needs.

- **Limitations in Batching**: While powerful, the batching mechanism has specific constraints. For instance, the `nobatch()` method indicates that not all operations can or should be batched, and developers must handle such cases explicitly. The test cases in `torchfold_test.py` highlight scenarios where batching is disabled for specific nodes.

- **Testing and Validation**: The repository includes unit tests (`torchfold_test.py`) that cover RNN batching optimizations and non-batched operations, ensuring the reliability of the core functionality. However, the test coverage appears limited to specific use cases, so users should validate performance in their specific contexts.

- **Example Availability**: The provided example (`spinn-example.py`) is explicitly marked as a demonstration and not a full implementation. Users looking to apply TorchFold to real-world problems may need to adapt or extend this code significantly.

### Usage Considerations

- **Target Audience**: This library is primarily aimed at researchers and developers working with PyTorch who need to handle dynamic, variable-sized data structures efficiently. Familiarity with PyTorch and neural network concepts is assumed.

- **Performance Optimization**: While TorchFold aims to optimize computation through dynamic batching, users should benchmark its performance against regular PyTorch implementations for their specific use cases, as the overhead of managing dynamic batches may vary depending on the data and model complexity.

- **Community and Support**: As an open-source project, TorchFold may rely on community contributions for updates and bug fixes. Users should be prepared to engage with the community or dive into the source code for troubleshooting or extending functionality.

This section aims to provide a clear understanding of what TorchFold offers and where caution or additional effort might be required when integrating it into your projects.

## Contributing

We welcome contributions from the community to help improve TorchFold. Whether it's bug fixes, feature enhancements, or documentation improvements, your input is valuable to us.

### How to Contribute

1. **Fork the Repository**: Start by forking the repository to your own GitHub account.
2. **Clone the Repository**: Clone the forked repository to your local machine for development.
3. **Make Your Changes**: Implement your changes or additions in your local copy. Ensure your code adheres to the general style and structure of the existing codebase.
4. **Test Your Changes**: Make sure to test your modifications to ensure they work as expected and do not introduce new issues.
5. **Commit Your Changes**: Commit your changes with a clear and descriptive commit message.
6. **Push to Your Fork**: Push your changes to your forked repository on GitHub.
7. **Submit a Pull Request**: Create a pull request from your fork to the main repository. Provide a detailed description of your changes and the motivation behind them.

### Contribution Guidelines

- **Code Style**: While there are no strict style guidelines defined, please follow the style and conventions used in the existing codebase for consistency.
- **Testing**: Ensure that your contributions include appropriate tests if applicable. Verify that existing tests pass before submitting a pull request.
- **Documentation**: Update or add documentation for any new features or changes to existing functionality.
- **Licensing**: By contributing to TorchFold, you agree that your contributions will be licensed under the Apache License, Version 2.0, as outlined in the LICENSE file.

If you have any questions or need assistance during the contribution process, feel free to reach out by opening an issue on GitHub. We appreciate your efforts to make TorchFold better!

## License

This project is licensed under the Apache License, Version 2.0. You can view the full license text in the [LICENSE](./LICENSE) file.
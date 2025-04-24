# TorchFold: Dynamic Batching Library for PyTorch

## Project Overview

This project, **TorchFold**, is a PyTorch library designed to implement dynamic batching with a simple and intuitive interface. Inspired by TensorFlow Fold, TorchFold enables efficient computation by optimizing batch operations dynamically.

### Purpose and Problems Solved
TorchFold addresses the challenge of handling variable-sized inputs in neural network computations, which is common in tasks like natural language processing or tree-structured data processing. Its main purpose is to facilitate dynamic batching, allowing users to process data more efficiently by grouping operations into optimized batches during execution. This reduces computational overhead and improves performance on hardware accelerators like GPUs.

### Key Features and Benefits
- **Dynamic Batching**: Automatically optimizes computations by batching operations dynamically, reducing memory usage and speeding up processing.
- **Simple Interface**: Users can easily integrate TorchFold into their PyTorch models by replacing standard neural network module calls with a straightforward `add` method.
- **Flexibility**: Supports complex, recursive computations such as tree traversals, making it ideal for structured data tasks.
- **Efficiency**: Leverages PyTorch's capabilities to execute batched operations efficiently, especially on CUDA-enabled devices.

TorchFold is particularly beneficial for developers and researchers working on machine learning models that require handling of hierarchical or irregular data structures, providing a powerful tool to streamline their workflows.

## Getting Started, Installation, and Setup

### Getting Started

To quickly get started with TorchFold, follow these steps for a basic usage example. This library provides dynamic batching for PyTorch, making it easier to handle variable-sized inputs in neural networks.

#### Quick Start Guide

1. **Install TorchFold**: Ensure you have the library installed (see the Installation and Setup section below for detailed instructions).
2. **Basic Usage**: Here's a simple example to demonstrate dynamic batching with TorchFold:

   ```python
   import torchfold
   import torch.nn as nn

   # Initialize Fold object
   f = torchfold.Fold()

   # Define a recursive function for tree traversal
   def dfs(node):
       if is_leaf(node):  # Replace with your leaf condition
           return f.add('leaf', node)
       else:
           prev = f.add('init')
           for child in children(node):  # Replace with your child accessor
               prev = f.add('child', prev, child)
           return prev

   # Define your model
   class Model(nn.Module):
       def __init__(self):
           super(Model, self).__init__()
           # Define necessary layers

       def leaf(self, leaf):
           # Process leaf node
           return leaf

       def child(self, prev, child):
           # Process child node
           return prev + child  # Example operation

   # Traverse your data structure
   res = dfs(my_tree)  # Replace with your data structure
   model = Model()
   result = f.apply(model, [[res]])
   ```

   This example shows how to use TorchFold to dynamically batch operations over a tree structure. Replace placeholders like `is_leaf`, `children`, and `my_tree` with your actual data logic.

### Installation and Setup

#### Prerequisites
- Python 3.x
- PyTorch (version compatible with your system)

#### Installation Steps
1. **Install via pip** (recommended):
   ```bash
   pip install torchfold
   ```

2. **Verify Installation**:
   After installation, you can verify it by importing the library in Python:
   ```python
   import torchfold
   print(torchfold.__version__)
   ```

#### Development Setup
If you want to work on the library or run examples:
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nearai/torchfold.git
   cd torchfold
   ```
2. **Install in Development Mode**:
   ```bash
   pip install -e .
   ```

#### Running Examples
An example implementation for a SPINN (Stack-augmented Parser-Interpreter Neural Network) model using TorchFold for dynamic batching is provided. To run the example:
1. Ensure dependencies like `torchtext` are installed:
   ```bash
   pip install torchtext
   ```
2. Navigate to the examples directory and run:
   ```bash
   python examples/snli/spinn-example.py
   ```
   Note: This is a demonstration script and not a fully functional model.

#### Platform-Specific Instructions
- **Linux/MacOS/Windows**: TorchFold is platform-agnostic as long as PyTorch is supported. Ensure you have the correct PyTorch version installed for your system. If using CUDA, verify that your PyTorch installation supports it.

#### Production Build
TorchFold is a library and does not require a separate production build. To use it in a production environment:
- Ensure dependencies are locked to specific versions in your `requirements.txt`.
- Deploy your application with the library installed via pip as shown above.

## Project Structure

This section provides an overview of the key directories and files in the repository to help you understand the structure and purpose of the codebase.

#### Key Directories
- **torchfold/**: Contains the core implementation of the TorchFold library, which is designed to enable dynamic batching in PyTorch by folding multiple computation graphs into a single graph.
- **examples/**: Includes sample scripts demonstrating the usage of TorchFold. Notably, it contains a subdirectory `snli/` with an example implementation using the SPINN model for the Stanford Natural Language Inference (SNLI) task.

#### Key Files
- **torchfold/torchfold.py**: The main module of the library, providing the `TorchFold` class and related functionalities for dynamic batching.
- **torchfold/torchfold_test.py**: Contains unit tests for the TorchFold library to ensure its functionality.
- **setup.py**: A setup script for installing the TorchFold library as a Python package.
- **README.md**: The main documentation file providing an overview and instructions for the repository.
- **LICENSE**: The licensing information for the repository.
- **.gitignore**: Specifies intentionally untracked files to ignore in the repository.
- **logo.jpg**: An image file, likely used for branding or documentation purposes.

## Additional Notes

### Compatibility and Requirements

TorchFold is designed to work seamlessly with PyTorch, providing dynamic batching capabilities to optimize computations. While specific version requirements are not explicitly stated in the codebase, it is recommended to use a recent version of PyTorch to ensure compatibility with the latest features and performance improvements.

### Performance Considerations

When using TorchFold, be aware that the dynamic batching process optimizes computations by grouping operations. However, the efficiency of batching can depend on the structure of your data and the complexity of your neural network model. For best results, ensure that your data structures (e.g., trees or graphs) are well-suited for dynamic batching, and monitor performance during development to adjust as needed.

### Debugging with Unfold

TorchFold includes an `Unfold` class, which serves as a debugging tool. Unlike the `Fold` class, which batches operations for efficiency, `Unfold` executes computations immediately without batching. This can be particularly useful for troubleshooting issues in your model or understanding the flow of data through your computations. To use it, simply replace `Fold` with `Unfold` in your code, passing the same neural network module.

### Limitations

- **Argument Types**: TorchFold requires all arguments passed to the `add` method to be of type `Tensor`, `Variable`, `int`, or `Node`. Mixing incompatible types may result in errors.
- **Batch Constraints**: When using the `nobatch` method on a node, ensure that only one such node is used per operation to avoid errors during computation.

### Community and Support

TorchFold is an open-source project hosted on GitHub. For additional resources, you can refer to the associated blog post linked in the project description for a deeper understanding of dynamic batching with PyTorch. If you encounter issues or have questions, consider contributing to the repository or reaching out to the community via the project's GitHub issues page.

### Citation

If you use TorchFold in your research or projects, please cite it as follows:

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

We welcome contributions from the community to help improve this project. Here's how you can get involved:

#### How to Contribute
1. **Fork the Repository**: Start by forking this repository to your own GitHub account.
2. **Clone the Repository**: Clone the forked repository to your local machine using `git clone`.
3. **Make Changes**: Create a new branch for your changes with a descriptive name related to the feature or bug fix you're working on. Make your changes in this branch.
4. **Test Your Changes**: Ensure that your code works as expected. If applicable, add or update tests to cover your changes.
5. **Commit Your Changes**: Commit your changes with a clear and descriptive commit message following conventional commit format if possible (e.g., `feat: add new feature`, `fix: resolve issue with XYZ`).
6. **Push Your Changes**: Push your branch to your forked repository on GitHub.
7. **Create a Pull Request**: Open a pull request from your branch to the main repository's `main` or `master` branch. Provide a detailed description of your changes and the motivation behind them.

#### Contribution Guidelines
- **Code Style**: Please follow PEP 8 for Python code. Use tools like `flake8` or `pylint` to ensure your code adheres to these standards.
- **Testing**: All contributions should include relevant tests or updates to existing tests. Ensure that all tests pass before submitting a pull request. If tests are not present, consider adding them as part of your contribution.
- **Documentation**: Update or add documentation for any new features or changes to existing functionality. This includes inline comments, docstrings, and updates to README or other documentation files if necessary.
- **Issue Tracking**: If your contribution addresses a specific issue, reference the issue number in your pull request description (e.g., `Fixes #123`).

We review all pull requests and may request changes or provide feedback before merging. Thank you for contributing to our project!

## License

This project is licensed under the Apache License, Version 2.0. You can view the full license text in the [LICENSE](./LICENSE) file in this repository.
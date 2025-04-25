# TorchFold: Dynamic Batching for PyTorch

## Project Overview

This project, **torchfold**, is a PyTorch library designed to enable dynamic batching, a technique that optimizes computational efficiency in neural network operations. Its primary purpose is to enhance the performance of deep learning models by intelligently grouping operations during computation, particularly for tasks involving variable-sized or tree-structured data.

### Key Purpose and Problems Solved
**torchfold** addresses the challenge of inefficient computation in neural networks when dealing with non-uniform data structures. Traditional batching methods often struggle with dynamic or recursive structures like trees or graphs, leading to suboptimal performance. This library solves these issues by allowing operations to be batched dynamically, reducing computational overhead and improving training and inference speeds for models in natural language processing (NLP) and other domains with complex data.

### Key Features and Benefits
- **Dynamic Batching**: Automatically groups operations for variable-sized inputs, maximizing GPU utilization and minimizing computation time.
- **Seamless Integration with PyTorch**: Designed to work natively with PyTorch, making it easy to incorporate into existing models and workflows.
- **Support for Tree-Structured Data**: Includes functionality to handle recursive data structures, as demonstrated in the included example for the Stanford Natural Language Inference (SNLI) dataset using a SPINN (Stack-augmented Parser-Interpreter Neural Network) model.
- **Performance Optimization**: By batching operations, it significantly reduces the number of individual computations, leading to faster training and inference phases.
- **Debugging Support**: Offers an `Unfold` class for immediate computation during debugging, providing flexibility in development.

With **torchfold**, developers can build more efficient deep learning models, particularly for applications in NLP and other fields requiring processing of complex, hierarchical data structures.

## Getting Started, Installation, and Setup

### Quick Start Guide

TorchFold is a library for dynamic batching in PyTorch, designed to optimize computations by dynamically batching operations. Here's how to quickly get started with using TorchFold in your project:

1. **Basic Usage**: Replace direct calls to `nn.Module` methods with `f.add('function_name', arguments)` to construct an optimized computation graph.
2. **Apply Computation**: Use `f.apply(model, [inputs])` to execute the computation on your neural network model with dynamic batching.

Here's a simple example to illustrate the usage:

```python
    import torchfold
    import torch.nn as nn

    # Initialize TorchFold
    f = torchfold.Fold()

    # Define a recursive function for tree traversal (example)
    def dfs(node):
        if is_leaf(node):
            return f.add('leaf', node)
        else:
            prev = f.add('init')
            for child in children(node):
                prev = f.add('child', prev, child)
            return prev

    # Define your model
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            # Define layers or components

        def leaf(self, leaf):
            # Process leaf node
            return leaf

        def child(self, prev, child):
            # Process child node
            return prev + child  # Example operation

    # Execute computation
    res = dfs(my_tree)  # Replace with your data structure
    model = Model()
    result = f.apply(model, [[res]])
```

For a more detailed example, check out the `examples/snli/spinn-example.py` file in the repository, which demonstrates a TreeLSTM implementation using TorchFold for dynamic batching.

### Installation

To install TorchFold, we recommend using the pip package manager. Follow these steps to get the library installed on your system:

1. Ensure you have Python and pip installed on your system.
2. Run the following command to install TorchFold:

    ```bash
    pip install torchfold
    ```

This will download and install the latest version of TorchFold (currently v0.1.0) along with any required dependencies.

#### Platform-Specific Instructions
- **Linux/macOS/Windows**: The installation process is platform-independent as long as Python and pip are available. No additional steps are required.
- **GPU Support**: TorchFold works with PyTorch, so if you intend to use GPU acceleration, ensure that you have the appropriate CUDA version installed and that PyTorch is configured to use CUDA.

### Setup

Once TorchFold is installed, there is minimal setup required to start using it in your project. Follow these steps to integrate it into your development environment:

1. **Import TorchFold**: In your Python script or notebook, import the necessary components:

    ```python
    from torchfold import Fold
    ```

2. **Integrate with PyTorch**: Ensure you have PyTorch installed. If not, install it via pip:

    ```bash
    pip install torch
    ```

    or for CUDA support:

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

    Replace `cu118` with the appropriate CUDA version for your system if needed.

#### Development Environment
- There are no specific steps for running TorchFold in a development environment beyond installing the package and integrating it into your PyTorch-based project.
- You can test your setup by running the provided example in `examples/snli/spinn-example.py`. This script demonstrates how to use TorchFold with a TreeLSTM model for natural language inference tasks.

#### Production Release
- TorchFold itself does not require a specific build process for production as it is a library used within PyTorch projects. For production deployment, follow standard practices for deploying PyTorch models, ensuring that all dependencies (including TorchFold) are installed in your production environment.
- Use the same `pip install torchfold` command to install the library on your production server or container.

## Project Structure

This section outlines the layout of the project, detailing the key directories and files that constitute the repository.

### Directory and File Overview

- **root directory**: Contains essential project files including:
  - **.gitignore**: Specifies intentionally untracked files to ignore.
  - **LICENSE**: Contains the licensing information for the project.
  - **README.md**: The main documentation file providing an overview and instructions for the project.
  - **setup.py**: A script for installing the project and its dependencies.
  - **logo.jpg**: A graphic file, likely used for branding or documentation purposes.

- **examples/**: A directory containing example implementations or usage scenarios.
  - **examples/snli/spinn-example.py**: An example script demonstrating the usage of the project, possibly related to the Stanford Natural Language Inference (SNLI) dataset with a specific model or algorithm (SPINN).

- **torchfold/**: The main source code directory for the project.
  - **torchfold/__init__.py**: Marks this directory as a Python package, potentially containing initialization code for the module.
  - **torchfold/torchfold.py**: The core module file, likely containing the primary functionality or implementation of the project.
  - **torchfold/torchfold_test.py**: Contains test cases for the `torchfold` module to ensure functionality and reliability.

This structure provides a clear organization of the project's components, separating core functionality, examples, and essential documentation.

## Technologies Used

- **PyTorch**: The primary deep learning framework used for dynamic batching and tensor computations. The project leverages PyTorch's `torch` and `torch.autograd` modules for building and managing neural network operations.
- **Python**: The core programming language used for implementing the project logic and functionality.

## Additional Notes

This section provides supplementary information about **TorchFold**, a library designed to facilitate dynamic batching in PyTorch, inspired by TensorFlow Fold. The goal is to offer a seamless way to optimize computations by dynamically batching operations, which is particularly useful for tasks involving variable-sized inputs such as tree-structured data or graphs.

### Compatibility and Requirements
TorchFold is built to work with PyTorch, and users are encouraged to ensure their PyTorch version is compatible with the library. While specific version requirements are not detailed in the codebase, checking the PyTorch documentation for updates and compatibility notes is recommended before installation. The library is distributed via PyPI, ensuring easy access and installation.

### Use Cases
TorchFold shines in scenarios where traditional static batching falls short, such as processing recursive or hierarchical data structures. Common use cases include natural language processing tasks (e.g., parsing tree structures in sentences) and graph-based machine learning models where input sizes vary. The provided example in the repository illustrates how to handle tree structures using a depth-first search approach with dynamic batching.

### Limitations and Considerations
While TorchFold offers significant advantages for dynamic batching, users should be aware of potential overhead in constructing the computation graph for very small datasets, where traditional batching might be more efficient. Additionally, the library requires a good understanding of PyTorch's `nn.Module` to define custom operations for batching, which might pose a learning curve for newcomers.

### Community and Support
TorchFold is an open-source project hosted on GitHub, and community contributions are welcome. For further reading and insights, refer to the associated blog post linked in the project description. For support or to report issues, users can reach out via the GitHub repository's issue tracker. Note that the project is maintained by NEAR Inc, and direct contact with the author is possible through the provided email in the setup configuration.

### Citation
If you use TorchFold in your research or projects, please consider citing it as outlined in the main README. This helps acknowledge the work of the contributors and supports the project's visibility in academic and professional communities.

## Contributing

We welcome contributions to the TorchFold project! If you are interested in helping improve dynamic batching with PyTorch, please follow these guidelines to ensure a smooth contribution process.

### How to Contribute
1. **Fork the Repository**: Start by forking the repository on GitHub and clone it to your local machine.
2. **Make Changes**: Implement your changes or additions in your local copy. Ensure your code aligns with the project's purpose and functionality.
3. **Test Your Changes**: Before submitting, make sure to test your modifications to avoid introducing bugs.
4. **Submit a Pull Request**: Push your changes to your fork and submit a pull request to the main repository. Provide a clear description of your changes and the motivation behind them.

### Contribution Guidelines
- **Code Style**: Follow Python PEP 8 style guidelines for code formatting and structure to maintain consistency across the codebase.
- **Documentation**: Update or add documentation for any new features or changes to existing functionality. Ensure clarity for other developers.
- **Testing**: Include tests for new features or bug fixes. Ensure that your code passes existing tests if applicable.
- **Commit Messages**: Write clear and descriptive commit messages that explain the purpose of the changes.

### Community
Join the conversation or seek help by opening an issue on GitHub for bugs, feature requests, or general discussions. We appreciate your input and collaboration in making TorchFold better!

Thank you for contributing to TorchFold and supporting dynamic batching with PyTorch!

## License

This project is licensed under the Apache License, Version 2.0. You can view the full license text in the [LICENSE](./LICENSE) file in this repository.
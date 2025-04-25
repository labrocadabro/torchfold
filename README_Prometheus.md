# TorchFold: Dynamic Batching Library for PyTorch

## Project Overview

This project, TorchFold, is a PyTorch library designed to implement dynamic batching with a simple and intuitive interface. It serves as an analog to TensorFlow Fold, providing a mechanism to optimize computations by dynamically batching operations, which is particularly useful for handling variable-sized or recursive data structures like trees or graphs in neural networks.

### Main Purpose and Problems Solved
TorchFold addresses the challenge of efficiently processing data with irregular structures in deep learning models. Traditional batching methods often struggle with non-uniform data, leading to inefficient computation or wasted resources. TorchFold solves this by allowing developers to construct optimized computations that adapt to the data's structure at runtime, improving performance and resource utilization in PyTorch-based projects.

### Key Features and Benefits
- **Dynamic Batching**: Automatically batches operations based on the input data structure, reducing computational overhead.
- **Simple Interface**: Easily integrate with existing PyTorch code by replacing direct calls to neural network modules with TorchFold's `add` method.
- **Optimized Computation**: Constructs and executes optimized computation graphs, enhancing performance for complex, recursive operations.
- **Flexibility**: Supports a variety of neural network architectures and data types, making it suitable for diverse deep learning tasks.

TorchFold is a powerful tool for researchers and developers working on natural language processing, graph neural networks, or any domain where data structures are not fixed or uniform, enabling more efficient and scalable model training and inference.

## Getting Started, Installation, and Setup

### Getting Started

**Quick Start Guide**

`torchfold` is a library for dynamic batching with PyTorch, designed to optimize computations by dynamically batching operations. Here's how to quickly use it in your project:

1. **Basic Usage**: Replace direct calls to your neural network modules with `f.add('function_name', arguments)` to construct an optimized computation graph.
2. **Apply Computation**: Use `f.apply(model, [inputs])` to execute the computation on your model with dynamic batching.

Here's a simple example of how to use `torchfold` with a tree structure:

```python
    import torchfold
    import torch.nn as nn

    # Initialize Fold
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
            # Initialize your model components
            pass

        def leaf(self, leaf):
            # Process leaf node
            pass

        def child(self, prev, child):
            # Process child node
            pass

    # Process a tree structure
    res = dfs(my_tree)
    model = Model(...)
    # Execute with dynamic batching
    result = f.apply(model, [[res]])
```

For a more detailed example, refer to the `examples/snli/spinn-example.py` file in the repository, which demonstrates usage in a tree-based neural network for natural language inference.

### Installation and Setup

#### Prerequisites
- Python 3.x
- PyTorch (version compatible with your system, typically the latest stable version is recommended)

#### Installation

To install `torchfold`, use pip for the easiest setup:

```bash
pip install torchfold
```

Alternatively, you can install from source by cloning the repository and running:

```bash
python setup.py install
```

#### Development Setup

If you plan to contribute or modify the library, follow these steps:

1. Clone the repository:
   ```bash
git clone https://github.com/nearai/torchfold.git
cd torchfold
   ```
2. Install in development mode:
   ```bash
python setup.py develop
   ```

This setup allows you to make changes to the codebase and have them reflected immediately without reinstalling.

#### Platform-Specific Instructions

- **Linux/MacOS/Windows**: The library is platform-independent and should work on any system with Python and PyTorch installed. Ensure your PyTorch installation matches your system's architecture (CPU/GPU support).
- **CUDA Support**: If you have a GPU and want to leverage CUDA for faster computation, ensure that your PyTorch installation includes CUDA support. You can check this with `torch.cuda.is_available()` in Python. The `torchfold` library itself does not require additional CUDA setup beyond PyTorch.

#### Running Examples

To run the provided example (as seen in `examples/snli/spinn-example.py`), ensure you have additional dependencies installed:

```bash
pip install torchtext
```

Then, run the example with or without fold optimizations:

```bash
python examples/snli/spinn-example.py --fold
```

Use `--no-cuda` flag if you do not have CUDA support or prefer CPU computation:

```bash
python examples/snli/spinn-example.py --fold --no-cuda
```

#### Production Build

Since `torchfold` is a library, there is no specific production build process. Ensure that your application using `torchfold` is optimized for production by:
- Using the latest stable version of PyTorch.
- Testing your model with dynamic batching on representative datasets to ensure performance.
- Packaging your application with dependencies pinned to specific versions for reproducibility.

## Project Structure

The project is organized into a clear and concise structure, with directories and files serving specific purposes. Below is an overview of the key components of the repository:

#### Key Directories and Files

- **`torchfold/`**: The core module directory containing the primary implementation of the dynamic batching functionality for PyTorch.
  - **`__init__.py`**: Initializes the `torchfold` module, exporting the `Fold` and `Unfold` classes.
  - **`torchfold.py`**: Contains the main logic for dynamic batching, including the `Fold` class for optimized computation and the `Unfold` class for debugging purposes.
  - **`torchfold_test.py`**: Includes unit tests for the `torchfold` module, covering functionality like RNN batching and node operations.

- **`examples/`**: Contains example implementations demonstrating the usage of `torchfold`.
  - **`snli/spinn-example.py`**: Provides an example of using `torchfold` with a SPINN (Stack-augmented Parser-Interpreter Neural Network) model for natural language inference tasks.

- **`setup.py`**: Configuration file for installing the `torchfold` package using setuptools. It defines metadata like version, description, and dependencies.

- **`.gitignore`**: Specifies files and directories to be ignored by Git, such as build artifacts and temporary files.

- **`LICENSE`**: Contains the licensing information for the project (Apache License, Version 2.0).

- **`logo.jpg`**: A logo image associated with the project, used in documentation.

- **`README.md`**: The main documentation file providing an overview, installation instructions, and usage examples for the project.

This structure ensures that the core functionality, testing, and examples are well-organized and accessible for developers looking to use or contribute to the `torchfold` library.

## Technologies Used

This project leverages the following major technologies, frameworks, and tools:

- **PyTorch**: A deep learning framework used for tensor computation and dynamic neural networks. TorchFold is built specifically to work with PyTorch for dynamic batching.
- **Python**: The primary programming language used for the implementation of this library.
- **setuptools**: Utilized for packaging and distributing the TorchFold library, as seen in the setup.py configuration.

These technologies form the core foundation of the TorchFold project, enabling dynamic batching capabilities for PyTorch-based applications.

## Additional Notes

### Performance Considerations

When using `torchfold` for dynamic batching in PyTorch, keep in mind that the efficiency of batching operations depends heavily on the structure of your data and the operations being performed. The `Fold` class is designed to optimize computation by grouping operations into batches dynamically. However, for very small batch sizes or highly irregular data structures, the overhead of managing the fold graph might outweigh the benefits of batching. In such cases, consider using the `Unfold` class for debugging or simpler computation without batching.

### Debugging with Unfold

The library provides an `Unfold` class as an alternative to `Fold`, primarily for debugging purposes. Unlike `Fold`, which batches operations for efficiency, `Unfold` performs computations immediately. This can be useful for tracing through the computation graph step-by-step to identify issues in your model or data processing pipeline. To switch to `Unfold`, initialize it with your neural network module and use it in place of `Fold`.

### CUDA Support

`torchfold` supports CUDA for GPU acceleration. To enable CUDA, call the `.cuda()` method on your `Fold` or `Unfold` instance before adding operations. Ensure that your PyTorch installation is configured for CUDA and that your model and data are also moved to the GPU if necessary.

### Limitations

- **Argument Types**: All arguments passed to the `add` method in `Fold` must be of type `Tensor`, `Variable`, `int`, or a `Fold.Node`. Mixing incompatible types will result in a `ValueError`.
- **Batch Consistency**: When using `nobatch()` on a node, ensure that only one such node is used per operation to avoid errors.
- **Dynamic Nature**: The dynamic batching approach may lead to varying memory usage during runtime, which could be a concern for resource-constrained environments.

### Community and Support

For further details on implementation or to discuss use cases, refer to the blog post linked in the project setup or visit the source repository on GitHub. Contributions, bug reports, and feature requests are welcome as per the contributing guidelines (see the relevant section of this README).

### Example Usage

An example script is available in the `examples/snli/` directory, demonstrating the application of `torchfold` in a natural language inference task using the SNLI dataset. This can serve as a starting point for understanding how to integrate dynamic batching into your own projects.

## Contributing

We welcome contributions from the community to help improve TorchFold. Whether it's bug fixes, feature enhancements, or documentation updates, your input is valuable to us.

### How to Contribute
1. **Fork the Repository**: Start by forking the repository to your own GitHub account.
2. **Clone the Repository**: Clone the forked repository to your local machine for development.
3. **Make Changes**: Implement your changes or additions in your local copy. Ensure your code aligns with the project's purpose and functionality.
4. **Test Your Changes**: Before submitting, test your changes to ensure they work as expected and do not introduce new issues.
5. **Commit Your Changes**: Commit your changes with clear, descriptive commit messages.
6. **Push to Your Fork**: Push your changes to your forked repository on GitHub.
7. **Submit a Pull Request**: Create a pull request from your fork to the main repository. Provide a detailed description of your changes and the motivation behind them.

### Contribution Guidelines
- **Code Style**: Please follow a consistent coding style. If you're unsure, refer to the existing codebase for guidance or adopt widely accepted Python style guidelines like PEP 8.
- **Testing**: Ensure that your contributions include appropriate tests if applicable. This helps maintain the stability of the project. If you're adding new functionality, consider adding test cases in a similar style to existing tests.
- **Documentation**: Update or add documentation for any new features or changes to existing functionality. Clear documentation helps other users understand and utilize the updates.
- **Issue Tracking**: If your contribution addresses a specific issue, reference the issue number in your pull request description.

We review all contributions and may request changes or provide feedback before merging. Thank you for taking the time to contribute to TorchFold!

## License

This project is licensed under the Apache License, Version 2.0. You can view the full license text in the [LICENSE](./LICENSE) file.
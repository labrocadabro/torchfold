# TorchFold: Dynamic Batching Library for PyTorch

## Project Overview

TorchFold is a Python library designed for dynamic batching in PyTorch, providing a streamlined interface to optimize computations in neural networks. Inspired by TensorFlow Fold, TorchFold enables developers to efficiently handle variable-sized inputs and complex recursive structures by dynamically batching operations, which significantly improves performance during training and inference.

### Purpose and Problems Solved
The primary purpose of TorchFold is to address the challenge of processing data with varying structures, such as trees or graphs, in deep learning models built with PyTorch. Traditional batching methods often struggle with non-uniform data, leading to inefficient computation and memory usage. TorchFold solves this by constructing optimized computation graphs and executing them in a batched manner, reducing overhead and enhancing scalability for recursive neural architectures.

### Key Features and Benefits
- **Dynamic Batching**: Automatically batches operations for inputs of varying sizes, optimizing resource utilization.
- **Simple Interface**: Seamlessly integrates with existing PyTorch code by replacing direct neural network module calls with a straightforward `add` method.
- **Performance Optimization**: Reduces computational overhead by constructing and executing optimized computation graphs.
- **Flexibility**: Supports complex data structures like trees, making it ideal for tasks such as natural language processing (e.g., parsing sentences with recursive structures).
- **Debugging Support**: Includes an `Unfold` class for immediate computation during debugging, bypassing dynamic batching when necessary.

TorchFold is particularly beneficial for researchers and developers working on models that require processing hierarchical or recursive data, offering a powerful tool to enhance efficiency and maintainability in PyTorch-based projects.

## Getting Started, Installation, and Setup

### Quick Start Guide

TorchFold is a library for dynamic batching in PyTorch, designed to optimize computations by batching operations dynamically. Here's how to quickly get started with using TorchFold:

1. **Initialize TorchFold**: Create an instance of `Fold` to start constructing your computation graph.
   ```python
   f = torchfold.Fold()
   ```
2. **Define Your Computation**: Replace direct calls to neural network modules with `f.add()` to build an optimized computation graph.
   ```python
   def dfs(node):
       if is_leaf(node):
           return f.add('leaf', node)
       else:
           prev = f.add('init')
           for child in children(node):
               prev = f.add('child', prev, child)
           return prev
   ```
3. **Create Your Model**: Define a model class inheriting from `nn.Module` with methods corresponding to the operations added in `f.add()`.
   ```python
   class Model(nn.Module):
       def __init__(self, ...):
           ...

       def leaf(self, leaf):
           ...

       def child(self, prev, child):
           ...
   ```
4. **Execute Computation**: Apply the computation graph to your model to dynamically batch and execute operations.
   ```python
   res = dfs(my_tree)
   model = Model(...)
   result = f.apply(model, [[res]])
   ```

This quick start guide covers the basic usage of TorchFold. For detailed installation instructions and advanced configurations, refer to the sections below.

### Installation and Setup

#### Prerequisites
- Python 3.x
- PyTorch (version compatible with your system, typically the latest stable version is recommended)

#### Installing TorchFold
TorchFold can be easily installed using pip, which will handle the necessary dependencies automatically:
```bash
pip install torchfold
```

#### Verifying Installation
After installation, you can verify that TorchFold is installed by running the following in a Python environment:
```python
import torchfold
print(torchfold.__version__)  # Should print '0.1.0'
```

#### Development Setup
If you wish to contribute to TorchFold or work with the latest source code, you can set up a development environment as follows:
1. Clone the repository:
   ```bash
   git clone https://github.com/nearai/torchfold.git
   cd torchfold
   ```
2. Install in development mode:
   ```bash
   pip install -e .
   ```
This will install TorchFold in editable mode, allowing you to make changes to the source code and have them reflected immediately without reinstalling.

#### Platform-Specific Instructions
- **Linux/macOS/Windows**: The installation process via pip works across all major platforms without additional configuration. Ensure that PyTorch is installed and compatible with your operating system and hardware (e.g., CUDA support for GPU acceleration).
- **GPU Support**: If you plan to use TorchFold with GPU acceleration, make sure to install the CUDA version of PyTorch before installing TorchFold. You can enable CUDA in your code by calling `f.cuda()` on your `Fold` instance.

#### Running in Development
To run TorchFold in a development environment after setting up as described above, simply import and use it in your Python scripts or notebooks. Changes to the source code will be reflected immediately due to the editable installation.

#### Building for Production
TorchFold is a library and does not require a separate build step for production use. Once installed via pip, it is ready to be used in production environments. Ensure that your application or model code is optimized and tested before deployment. If you are packaging an application that uses TorchFold, include it in your dependency list (e.g., in a `requirements.txt` file):
```text
torchfold==0.1.0
```

This ensures that the correct version of TorchFold is installed in your production environment.

## Features / Capabilities

This section outlines the core features and capabilities of **TorchFold**, a library designed to enhance PyTorch with dynamic batching capabilities.

### Core Features

- **Dynamic Batching**: TorchFold introduces a mechanism to dynamically batch operations in PyTorch, allowing for efficient processing of variable-sized inputs. This is particularly useful for tasks like natural language processing where input sizes can vary widely.
- **Fold Class**: The primary component of TorchFold, the `Fold` class, manages the computation graph by organizing operations into steps and nodes. It supports batching of tensors and variables for optimized computation.
- **Node Management**: Operations are encapsulated as nodes, which can be split for handling multiple return values and marked as non-batched if needed. This provides fine-grained control over computation.
- **CUDA Support**: TorchFold includes support for CUDA, enabling GPU acceleration for batched operations when available.
- **Unfold Class for Debugging**: An `Unfold` class is provided as a debugging tool, allowing for immediate computation without batching, which can help in tracing and verifying operations step-by-step.
- **Flexible Operation Application**: The library allows operations to be applied to a neural network module (`nn`) with automatic handling of batched arguments, simplifying the integration into existing PyTorch workflows.

### Example Use Case

While specific usage examples are detailed in the Usage Examples section, TorchFold is particularly suited for scenarios requiring dynamic computation graphs, such as processing recursive data structures or handling variable-length sequences in neural networks. An example script is available in the `examples/snli/` directory demonstrating its application in a natural language inference task.

## Technologies Used

- **Python**: The primary programming language used in this project.
- **PyTorch**: A deep learning framework used for tensor computations and neural network operations, central to the functionality of TorchFold for dynamic batching.
- **TorchText**: A library used for natural language processing tasks, as seen in the example implementation for handling datasets like SNLI.

## Usage Examples

### Basic Usage with SPINN Example

The repository includes an example implementation of a Stack-augmented Parser-Intepreter Neural Network (SPINN) for natural language inference using the SNLI dataset. This example demonstrates how to use `torchfold` for dynamic batching in tree-structured neural networks.

To run the SPINN example, follow these steps:

1. **Ensure Dependencies are Installed**: Make sure you have PyTorch and TorchText installed in your environment. If not, install them using pip:
   ```bash
   pip install torch torchtext
   ```

2. **Navigate to the Examples Directory**: Change to the directory containing the SPINN example script:
   ```bash
   cd examples/snli
   ```

3. **Run the SPINN Example**: Execute the script with default settings. By default, it uses a batch size of 128 and checks for CUDA availability:
   ```bash
   python spinn-example.py
   ```

   - To enable `torchfold` for dynamic batching, add the `--fold` flag:
     ```bash
     python spinn-example.py --fold
     ```

   - To disable CUDA (if you don't have a GPU or prefer CPU), use the `--no-cuda` flag:
     ```bash
     python spinn-example.py --no-cuda
     ```

   - To adjust the batch size, use the `--batch_size` argument (e.g., for a batch size of 64):
     ```bash
     python spinn-example.py --batch_size 64
     ```

4. **Monitor Training Progress**: The script will output the average time taken per iteration every 10 iterations, allowing you to monitor the training performance.

### Understanding the Example

The `spinn-example.py` script showcases two ways of encoding tree structures:
- **Regular Encoding**: Processes each tree node individually without batching.
- **Fold Encoding with `torchfold`**: Uses `torchfold.Fold` to batch operations dynamically, which can improve performance by grouping similar operations across multiple examples.

This example trains a model on the SNLI dataset for 10 epochs using the Adam optimizer and Cross Entropy Loss, demonstrating a practical application of `torchfold` in handling recursive neural network computations efficiently.

## Project Structure

This section outlines the structure of the repository, highlighting key directories and files that are essential for understanding the project's organization.

#### Key Directories and Files
- **examples/**: Contains example scripts demonstrating the usage of the project. Notably, it includes `snli/spinn-example.py`, which serves as a practical illustration of implementing the project in a specific context.
- **torchfold/**: The core directory for the project's source code. It includes:
  - `__init__.py`: Marks this directory as a Python package.
  - `torchfold.py`: The main implementation file for the project's functionality.
  - `torchfold_test.py`: Contains test cases for validating the functionality in `torchfold.py`.
- **.gitignore**: Specifies intentionally untracked files to ignore.
- **LICENSE**: Details the licensing terms under which the project is released.
- **README.md**: Provides an overview and basic documentation for the project.
- **README_Prometheus.md**: Additional documentation or notes specific to Prometheus-related content.
- **logo.jpg**: A graphical asset, likely used for branding or documentation purposes.
- **setup.py**: A setup script for installing the project as a Python package.

## Additional Notes

This section provides supplementary information about the `torchfold` library, which focuses on dynamic batching with PyTorch. Below are some key points and considerations for users and contributors.

#### Compatibility and Requirements
`torchfold` is designed to work seamlessly with PyTorch, leveraging its tensor operations and autograd system for dynamic batching. Ensure that you have a compatible version of PyTorch installed to avoid runtime issues. While specific version requirements are not explicitly defined in the codebase, checking the latest PyTorch documentation for compatibility with dynamic computation graphs is recommended.

#### Performance Considerations
Dynamic batching, as implemented in `torchfold`, optimizes computation by grouping operations into batches during the execution of neural network models. This can significantly improve performance on variable-sized inputs (e.g., trees or graphs). However, the efficiency depends on the structure of your data and model architecture. Users are encouraged to experiment with different batching strategies and monitor memory usage, especially when dealing with large datasets or complex models.

#### Debugging with Unfold
For debugging purposes, `torchfold` provides an `Unfold` class that performs computations immediately rather than batching them. This can be useful for identifying issues in your model without the complexity of dynamic batching. Switch to `Unfold` during development to isolate errors, then revert to `Fold` for optimized performance in production.

#### Use Case Example
An example script is provided in the `examples/snli/` directory, demonstrating how `torchfold` can be applied to the Stanford Natural Language Inference (SNLI) dataset using a SPINN (Stack-augmented Parser-Interpreter Neural Network) model. This example serves as a practical guide for implementing dynamic batching in natural language processing tasks. Review `spinn-example.py` to understand how to structure your data and model for use with `torchfold`.

#### Community and Support
`torchfold` is an open-source project under the Apache License, Version 2.0. For additional resources, refer to the blog post linked in the project metadata for a detailed explanation of dynamic batching with PyTorch. If you encounter issues or have questions, consider checking the GitHub repository for community discussions or opening an issue for direct support from maintainers.

#### Limitations
While `torchfold` excels at dynamic batching, it may not be suitable for all types of neural network architectures or data formats. Ensure your use case aligns with the library's capabilities, particularly if dealing with fixed-size inputs where traditional static batching might be more efficient. The library assumes a certain level of familiarity with PyTorch's computational graph and tensor operations.

This library is a specialized tool for advanced PyTorch users looking to optimize performance with dynamic data structures. If you are new to PyTorch, consider exploring foundational tutorials before integrating `torchfold` into your projects.

## Contributing

We welcome contributions from the community to help improve this project. Whether it's bug fixes, new features, or documentation improvements, your input is valuable to us. Follow the steps below to get started with contributing.

### How to Contribute
1. **Fork the Repository**: Start by forking the repository on GitHub and clone it to your local machine.
2. **Make Changes**: Implement your changes or additions in your forked repository. Ensure your code adheres to the existing style for consistency.
3. **Test Your Changes**: Make sure to test your modifications to ensure they work as expected and do not introduce new issues.
4. **Submit a Pull Request**: Once your changes are ready, submit a pull request to the main repository. Provide a clear description of the changes and the motivation behind them.

### Contribution Guidelines
- **Code Style**: We strive to maintain a consistent code style across the project. Please format your code to match the style used in the existing codebase. While specific style guides are not currently enforced, readability and consistency are key.
- **Testing**: All contributions should include relevant tests to cover new functionality or bug fixes. Ensure that existing tests pass before submitting your pull request.
- **Documentation**: If your contribution adds new features or changes existing functionality, please update the documentation accordingly to reflect these changes.
- **Issue Tracking**: If your contribution addresses a specific issue, reference the issue number in your pull request description.

### Community
Feel free to reach out with questions or for discussions regarding potential contributions. You can do this via GitHub issues or by joining discussions in relevant forums or channels if available.

Thank you for considering contributing to this project. Your efforts help make this project better for everyone!

## License

This project is licensed under the Apache License, Version 2.0. You can view the full license text in the [LICENSE](./LICENSE) file in this repository.
# TorchFold: Dynamic Batching for PyTorch Neural Networks

## Project Overview

TorchFold is a PyTorch library designed to enable dynamic batching for neural network computations. Inspired by TensorFlow Fold, it provides a simple and efficient interface to optimize computations by dynamically batching operations, which can significantly improve performance when processing variable-sized inputs such as trees or graphs.

### Main Purpose and Problems Solved
TorchFold addresses the challenge of efficiently processing data with irregular structures in deep learning models. Traditional batching methods often struggle with inputs of varying sizes, leading to inefficient computation or the need for extensive padding. TorchFold solves this by dynamically batching operations, ensuring optimal use of computational resources without requiring manual intervention for batch management.

### Key Features and Benefits
- **Dynamic Batching**: Automatically batches computations for inputs of different sizes, reducing the need for padding and improving efficiency.
- **Simple Interface**: Easily integrate with existing PyTorch models by replacing direct calls to neural network modules with a simple `add` method.
- **Performance Optimization**: Executes batched operations in an optimized manner, leveraging the full power of GPU acceleration when available.
- **Flexibility**: Supports complex, recursive data structures like trees, making it ideal for tasks in natural language processing and other domains requiring hierarchical data processing.
- **Debugging Support**: Includes an `Unfold` class for immediate computation during debugging, bypassing dynamic batching when needed.

## Getting Started, Installation, and Setup

### Quick Start Guide

TorchFold provides dynamic batching for PyTorch with a simple interface. Here's how to quickly get started with using TorchFold in your project:

1. **Basic Usage**: Replace direct calls to `nn` module functions with `f.add('function_name', arguments)` to enable dynamic batching.
2. **Example Code**:
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

### Installation and Setup

#### Installing TorchFold

TorchFold can be easily installed using the pip package manager. Run the following command to install the library:

```bash
pip install torchfold
```

#### Dependencies

TorchFold relies on PyTorch for its core functionality. Ensure you have PyTorch installed in your environment. If not, you can install it via pip:

```bash
pip install torch
```

#### Platform-Specific Instructions

TorchFold is platform-independent as long as PyTorch is supported on your system. Ensure that your operating system and hardware are compatible with PyTorch. For GPU support, make sure you have the appropriate CUDA version installed if you're using a GPU-enabled PyTorch build.

- **Linux/Windows/macOS**: No additional setup is required beyond installing PyTorch and TorchFold.

#### Development Setup

If you wish to contribute to TorchFold or work with the source code:

1. Clone the repository:
   ```bash
git clone https://github.com/nearai/torchfold.git
cd torchfold
   ```
2. Install in development mode:
   ```bash
pip install -e .
   ```

#### Production Build

For production use, installing via pip as described above is sufficient. There is no separate build process required for TorchFold since it is a library intended for integration into larger PyTorch projects. Ensure your application is optimized for production by following PyTorch best practices for deployment.

## API Reference

### Core Classes

#### `Fold`

- **Description**: A class for dynamic batching in PyTorch, allowing operations to be folded into batched computations for efficiency. It manages a computation graph of operations and supports batching of inputs and outputs.
- **Constructor Signature**:
  ```python
  Fold(volatile=False, cuda=False)
  ```
  - **Parameters**:
    - `volatile` (bool, optional): If True, the variables created will not track gradients. Defaults to `False`.
    - `cuda` (bool, optional): If True, tensors are created on CUDA devices. Defaults to `False`.
  - **Return Value**: An instance of `Fold`.
- **Example Usage**:
  ```python
  fold = Fold(volatile=False, cuda=True)
  # Add operations to the fold
  result = fold.add('some_operation', arg1, arg2)
  # Apply to a neural network module
  output = fold.apply(neural_module, [result])
  ```

##### Key Methods of `Fold`

- **Method: `cuda()`**
  - **Description**: Sets the fold to use CUDA for tensor operations.
  - **Signature**:
    ```python
    cuda()
    ```
  - **Parameters**: None
  - **Return Value**: `self` (for method chaining)
  - **Example Usage**:
    ```python
    fold = Fold()
    fold.cuda()
    ```

- **Method: `add(op, *args)`**
  - **Description**: Adds an operation to the fold computation graph. The operation is identified by a string `op` and takes variable arguments.
  - **Signature**:
    ```python
    add(op, *args)
    ```
  - **Parameters**:
    - `op` (str): The name of the operation to add.
    - `*args`: Variable arguments which can be of type `Fold.Node`, `int`, `torch.Tensor`, or `torch.autograd.Variable`.
  - **Return Value**: A `Fold.Node` object representing the operation node in the computation graph.
  - **Example Usage**:
    ```python
    fold = Fold()
    node = fold.add('linear', input_tensor, weight_tensor)
    ```

- **Method: `apply(nn, nodes)`**
  - **Description**: Applies the fold to a given neural network module, executing the operations defined in the fold.
  - **Signature**:
    ```python
    apply(nn, nodes)
    ```
  - **Parameters**:
    - `nn`: A neural network module with methods corresponding to the operations defined in the fold.
    - `nodes` (list): A list of `Fold.Node` objects or lists of nodes representing the outputs to retrieve.
  - **Return Value**: The computed results as a list of tensors.
  - **Example Usage**:
    ```python
    fold = Fold()
    node = fold.add('linear', input_tensor, weight_tensor)
    outputs = fold.apply(neural_module, [[node]])
    ```

#### `Unfold`

- **Description**: A debugging counterpart to `Fold`, performing computations immediately rather than batching them. Useful for debugging and verifying operations without batching overhead.
- **Constructor Signature**:
  ```python
  Unfold(nn, volatile=False, cuda=False)
  ```
  - **Parameters**:
    - `nn`: A neural network module to perform operations on.
    - `volatile` (bool, optional): If True, the variables created will not track gradients. Defaults to `False`.
    - `cuda` (bool, optional): If True, tensors are created on CUDA devices. Defaults to `False`.
  - **Return Value**: An instance of `Unfold`.
- **Example Usage**:
  ```python
  unfold = Unfold(neural_module, volatile=False, cuda=True)
  result = unfold.add('some_operation', arg1, arg2)
  output = unfold.apply(neural_module, [result])
  ```

##### Key Methods of `Unfold`

- **Method: `cuda()`**
  - **Description**: Sets the unfold to use CUDA for tensor operations.
  - **Signature**:
    ```python
    cuda()
    ```
  - **Parameters**: None
  - **Return Value**: `self` (for method chaining)
  - **Example Usage**:
    ```python
    unfold = Unfold(neural_module)
    unfold.cuda()
    ```

- **Method: `add(op, *args)`**
  - **Description**: Immediately performs an operation on the provided neural network module with the given arguments.
  - **Signature**:
    ```python
    add(op, *args)
    ```
  - **Parameters**:
    - `op` (str): The name of the operation to perform.
    - `*args`: Variable arguments which can be of type `Unfold.Node`, `int`, `torch.Tensor`, or `torch.autograd.Variable`.
  - **Return Value**: An `Unfold.Node` object containing the result of the operation.
  - **Example Usage**:
    ```python
    unfold = Unfold(neural_module)
    node = unfold.add('linear', input_tensor, weight_tensor)
    ```

- **Method: `apply(nn, nodes)`**
  - **Description**: Applies the unfold to the given neural network module, collecting results from the provided nodes.
  - **Signature**:
    ```python
    apply(nn, nodes)
    ```
  - **Parameters**:
    - `nn`: A neural network module (must match the one passed to the constructor).
    - `nodes` (list): A list of `Unfold.Node` objects or lists of nodes representing the outputs to retrieve.
  - **Return Value**: The computed results as a list of tensors.
  - **Example Usage**:
    ```python
    unfold = Unfold(neural_module)
    node = unfold.add('linear', input_tensor, weight_tensor)
    outputs = unfold.apply(neural_module, [[node]])
    ```

## Project Structure

This section provides an overview of the key directories and files in the TorchFold repository, which is designed to implement dynamic batching for PyTorch with a simple interface.

#### Key Directories and Files

- **`torchfold/`**: The core directory containing the primary implementation of the dynamic batching functionality.
  - **`torchfold/__init__.py`**: Initializes the TorchFold module by importing the main classes `Fold` and `Unfold` for use in dynamic batching and debugging, respectively.
  - **`torchfold/torchfold.py`**: Contains the main logic for dynamic batching. It defines the `Fold` class for constructing optimized computation graphs and batching operations, and the `Unfold` class for debugging by performing computations immediately without batching.

- **`examples/`**: Contains example implementations demonstrating the usage of TorchFold.
  - **`examples/snli/spinn-example.py`**: Provides an example of using TorchFold with a SPINN (Stack-augmented Parser-Interpreter Neural Network) model for natural language inference tasks. It includes both regular and folded implementations for tree encoding, showcasing the performance benefits of dynamic batching.

- **`setup.py`**: The setup script for installing TorchFold as a Python package. It defines the package metadata and dependencies.

- **`LICENSE`**: The file containing the licensing information for the repository, specifying the terms under which the code can be used and distributed.

- **`.gitignore`**: Specifies intentionally untracked files to ignore in the repository, such as temporary files or build artifacts.

- **`logo.jpg`**: A visual asset used in the README for branding purposes.

This structure encapsulates the essential components for implementing and demonstrating dynamic batching in PyTorch using TorchFold.

## Additional Notes

### Compatibility and Dependencies

TorchFold is designed to work seamlessly with PyTorch, providing dynamic batching capabilities. Ensure that you have PyTorch installed in your environment to utilize this library effectively. The library is compatible with Python and can be installed via pip as outlined in the Installation section.

### Use Cases

TorchFold is particularly useful for scenarios involving variable-sized or recursive data structures such as trees or graphs, where traditional batching methods fall short. It allows for efficient computation by dynamically batching operations, which can significantly improve performance in natural language processing tasks (e.g., SNLI with SPINN as shown in the examples) and other machine learning applications requiring hierarchical data processing.

### Citation

If you use TorchFold in your research or projects, please consider citing it as follows:

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

### Further Reading

For a deeper understanding of dynamic batching and the concepts behind TorchFold, refer to the associated blog post at [near.ai/articles/2017-09-06-PyTorch-Dynamic-Batching/](http://near.ai/articles/2017-09-06-PyTorch-Dynamic-Batching/). This resource provides additional context and detailed explanations of how TorchFold optimizes computations with PyTorch.

### Contact and Support

For inquiries or support, you can reach out to the author, Illia Polosukhin, at [illia@near.ai](mailto:illia@near.ai). Additionally, the source code and further project details are available on GitHub at [https://github.com/nearai/torchfold](https://github.com/nearai/torchfold).

## Contributing

Thank you for your interest in contributing to this project! We welcome contributions from the community to help improve and expand this library.

### How to Contribute

1. **Fork the Repository**: Start by forking the repository to your own GitHub account.
2. **Clone the Repository**: Clone the forked repository to your local machine for development.
3. **Create a Branch**: Create a new branch for your feature or bug fix. Use a descriptive name related to the change you're making.
4. **Make Your Changes**: Implement your changes in the codebase. Ensure your code adheres to the guidelines below.
5. **Test Your Changes**: Make sure to test your modifications. If applicable, add or update tests to cover your changes.
6. **Commit Your Changes**: Commit your changes with a clear and descriptive commit message.
7. **Push to Your Fork**: Push your branch to your forked repository.
8. **Submit a Pull Request**: Create a pull request from your branch to the main repository. Provide a detailed description of your changes and the motivation behind them.

### Contribution Guidelines

- **Code Style**: Follow PEP 8, the Python style guide, for all Python code. Use consistent indentation (4 spaces) and meaningful variable/function names.
- **Documentation**: Document your code where necessary. If you add new functionality, update or add relevant documentation.
- **Testing**: Ensure that your changes do not break existing functionality. Add tests for new features or bug fixes. Tests should be placed in the appropriate test files within the project structure.
- **Compatibility**: Ensure your code is compatible with the Python versions supported by the project (refer to setup.py for supported versions).
- **License**: By contributing, you agree that your contributions will be licensed under the same license as the project.

We review all contributions and may request changes before merging. Thank you for helping to make this project better!

## License

This project is licensed under the Apache License, Version 2.0. You can view the full license text in the [LICENSE](./LICENSE) file.
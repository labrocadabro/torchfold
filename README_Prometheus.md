# TorchFold: Dynamic Batching Library for PyTorch

## Project Overview

This repository contains **TorchFold**, a library for dynamic batching in PyTorch, inspired by TensorFlow Fold. TorchFold enables efficient computation by constructing optimized versions of neural network operations and dynamically batching them during execution. It provides a simple interface to replace direct calls to neural network modules with a folding mechanism that enhances performance for variable-sized inputs, such as tree-structured data.

### Main Purpose and Problems Solved
TorchFold addresses the challenge of handling variable-sized or recursive data structures (like trees or graphs) in neural networks. Traditional batching methods often struggle with such data due to inconsistent shapes or sizes. TorchFold solves this by dynamically batching operations, allowing for efficient processing of complex data structures without manual padding or restructuring.

### Key Features and Benefits
- **Dynamic Batching**: Automatically batches operations for inputs of varying sizes, optimizing computation on GPUs.
- **Simple Interface**: Easily integrate with existing PyTorch code by replacing direct neural network calls with `fold.add()` operations.
- **Optimized Execution**: Constructs and executes an optimized computation graph tailored to the input data.
- **Support for Recursive Structures**: Ideal for processing hierarchical or recursive data like syntax trees in natural language processing tasks.
- **Debugging Support**: Includes an `Unfold` class for immediate computation during debugging, bypassing dynamic batching when needed.

TorchFold is particularly beneficial for applications in natural language processing, graph neural networks, or any domain requiring efficient handling of non-uniform data structures.

## Getting Started, Installation, and Setup

### Quick Start Guide

TorchFold is a library for dynamic batching with PyTorch, providing a simple interface to optimize computations. Here's how to quickly get started:

1. **Install TorchFold**: Use pip to install the library (detailed instructions below).
   ```bash
   pip install torchfold
   ```
2. **Basic Usage**: Replace direct calls to `nn.Module` methods with `f.add('function_name', arguments)` to construct optimized computations. Use `f.apply` to execute dynamic batching.
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
       def __init__(self):
           super(Model, self).__init__()
           # Initialize your layers here

       def leaf(self, leaf):
           # Process leaf node
           pass

       def child(self, prev, child):
           # Process child node with previous state
           pass

   # Example tree processing
   res = dfs(my_tree)
   model = Model()
   f.apply(model, [[res]])
   ```

For more detailed examples, refer to the `examples` directory in the repository.

### Installation and Setup

#### Prerequisites
- Python 3.x
- PyTorch (version compatible with your system)

#### Installing TorchFold
TorchFold can be installed easily via pip:
```bash
pip install torchfold
```

Alternatively, if you want to install from the source for development purposes:
1. Clone the repository:
   ```bash
   git clone https://github.com/nearai/torchfold.git
   cd torchfold
   ```
2. Install the package:
   ```bash
   python setup.py install
   ```

#### Development Setup
To contribute or modify TorchFold:
1. Ensure you have the prerequisites installed.
2. Clone the repository as shown above.
3. Install in development mode:
   ```bash
   python setup.py develop
   ```

#### Platform-Specific Instructions
TorchFold is platform-agnostic as long as Python and PyTorch are supported on your system. Ensure that PyTorch is installed correctly for your platform (CPU or GPU setup) before using TorchFold.

#### Production Build
Since TorchFold is a library, there is no specific production build process. Once installed via pip or from source, it can be used in any production environment where PyTorch is supported.

## API Reference

### TorchFold Library API

The `torchfold` library provides tools for dynamic computation graph folding in PyTorch, enabling efficient batching of operations in neural networks with variable structure. Below is a detailed reference of the publicly exported classes and methods.

#### Fold Class

The `Fold` class is the core component for creating and managing dynamic computation graphs with batched operations.

- **Name**: `Fold`
  - **Description**: A class to manage a computation graph by folding operations into batched computations, optimizing performance for neural networks with dynamic structures.
  - **Constructor Signature**: `Fold(volatile=False, cuda=False)`
    - **Parameters**:
      - `volatile` (bool, optional): If True, disables gradient computation for inference mode. Defaults to False.
      - `cuda` (bool, optional): If True, moves tensors to CUDA. Defaults to False.
    - **Return Value**: A `Fold` instance.
  - **Example Usage**:
    ```python
    fold = Fold(volatile=True, cuda=True)
    ```

- **Method**: `cuda()`
  - **Description**: Enables CUDA support for the fold operations.
  - **Signature**: `cuda() -> Fold`
    - **Parameters**: None
    - **Return Value**: The `Fold` instance with CUDA enabled.
  - **Example Usage**:
    ```python
    fold = Fold().cuda()
    ```

- **Method**: `add(op, *args)`
  - **Description**: Adds an operation to the fold computation graph.
  - **Signature**: `add(op: str, *args: Union[Fold.Node, int, torch.Tensor, Variable]) -> Fold.Node`
    - **Parameters**:
      - `op` (str): The name of the operation (should correspond to a method in the neural network module).
      - `*args`: Variable arguments which can be `Fold.Node`, integers, PyTorch tensors, or autograd Variables.
    - **Return Value**: A `Fold.Node` representing the operation in the computation graph.
  - **Example Usage**:
    ```python
    node = fold.add('linear_layer', input_node, weight_tensor)
    ```

- **Method**: `apply(nn, nodes)`
  - **Description**: Applies the folded computation graph to a given neural network module.
  - **Signature**: `apply(nn: torch.nn.Module, nodes: List[Union[List[Fold.Node], Fold.Node]]) -> List[torch.Tensor]`
    - **Parameters**:
      - `nn` (torch.nn.Module): The neural network module to apply the operations to.
      - `nodes` (List[Union[List[Fold.Node], Fold.Node]]): The nodes or list of nodes representing the outputs to retrieve.
    - **Return Value**: A list of tensors representing the results of the computation.
  - **Example Usage**:
    ```python
    results = fold.apply(neural_net, [output_node])
    ```

#### Fold.Node Class

- **Name**: `Fold.Node`
  - **Description**: Represents a node in the computation graph of a `Fold` instance, corresponding to an operation at a specific step.
  - **Constructor Signature**: Internal use only, created via `Fold.add()`.
  - **Method**: `split(num)`
    - **Description**: Splits the node into multiple nodes if the operation returns multiple values.
    - **Signature**: `split(num: int) -> Tuple[Fold.Node, ...]`
      - **Parameters**:
        - `num` (int): Number of splits to create.
      - **Return Value**: A tuple of `Fold.Node` instances.
    - **Example Usage**:
      ```python
      node1, node2 = output_node.split(2)
      ```
  - **Method**: `nobatch()`
    - **Description**: Marks the node as not requiring batching, ensuring it is processed individually.
    - **Signature**: `nobatch() -> Fold.Node`
      - **Parameters**: None
      - **Return Value**: The same `Fold.Node` instance with batching disabled.
    - **Example Usage**:
      ```python
      single_node = node.nobatch()
      ```

#### Unfold Class

The `Unfold` class is a debugging tool that performs computations immediately without batching, useful for verifying the correctness of operations.

- **Name**: `Unfold`
  - **Description**: A debugging alternative to `Fold` that executes operations immediately without batching.
  - **Constructor Signature**: `Unfold(nn, volatile=False, cuda=False)`
    - **Parameters**:
      - `nn` (torch.nn.Module): The neural network module to apply operations to.
      - `volatile` (bool, optional): If True, disables gradient computation for inference mode. Defaults to False.
      - `cuda` (bool, optional): If True, moves tensors to CUDA. Defaults to False.
    - **Return Value**: An `Unfold` instance.
  - **Example Usage**:
    ```python
    unfold = Unfold(neural_net, volatile=True, cuda=True)
    ```

- **Method**: `cuda()`
  - **Description**: Enables CUDA support for the unfold operations.
  - **Signature**: `cuda() -> Unfold`
    - **Parameters**: None
    - **Return Value**: The `Unfold` instance with CUDA enabled.
  - **Example Usage**:
    ```python
    unfold = Unfold(neural_net).cuda()
    ```

- **Method**: `add(op, *args)`
  - **Description**: Adds and immediately executes an operation using the provided neural network module.
  - **Signature**: `add(op: str, *args: Union[Unfold.Node, int, torch.Tensor, Variable]) -> Unfold.Node`
    - **Parameters**:
      - `op` (str): The name of the operation (should correspond to a method in the neural network module).
      - `*args`: Variable arguments which can be `Unfold.Node`, integers, PyTorch tensors, or autograd Variables.
    - **Return Value**: An `Unfold.Node` representing the result of the operation.
  - **Example Usage**:
    ```python
    result_node = unfold.add('linear_layer', input_node, weight_tensor)
    ```

- **Method**: `apply(nn, nodes)`
  - **Description**: Applies the unfold operation to retrieve results from the given nodes. Ensures the neural network module matches the one provided at initialization.
  - **Signature**: `apply(nn: torch.nn.Module, nodes: List[Union[List[Unfold.Node], Unfold.Node]]) -> List[torch.Tensor]`
    - **Parameters**:
      - `nn` (torch.nn.Module): The neural network module (must match the one passed to constructor).
      - `nodes` (List[Union[List[Unfold.Node], Unfold.Node]]): The nodes or list of nodes representing the outputs to retrieve.
    - **Return Value**: A list of tensors representing the results of the computation.
  - **Example Usage**:
    ```python
    results = unfold.apply(neural_net, [output_node])
    ```

#### Unfold.Node Class

- **Name**: `Unfold.Node`
  - **Description**: Represents a node in the `Unfold` computation, holding the result tensor of an operation.
  - **Constructor Signature**: Internal use only, created via `Unfold.add()`.
  - **Method**: `split(num)`
    - **Description**: Splits the node's tensor into multiple nodes.
    - **Signature**: `split(num: int) -> List[Unfold.Node]`
      - **Parameters**:
        - `num` (int): Number of splits to create.
      - **Return Value**: A list of `Unfold.Node` instances.
    - **Example Usage**:
      ```python
      nodes = output_node.split(2)
      ```
  - **Method**: `nobatch()`
    - **Description**: Returns the node unchanged (no-op for compatibility with `Fold.Node`).
    - **Signature**: `nobatch() -> Unfold.Node`
      - **Parameters**: None
      - **Return Value**: The same `Unfold.Node` instance.
    - **Example Usage**:
      ```python
      single_node = node.nobatch()
      ```

## Project Structure

This section provides an overview of the key directories and files in the repository to help you navigate the codebase effectively.

#### Key Directories and Files

- **torchfold/**: The core directory containing the main implementation of the TorchFold library.
  - **torchfold.py**: The primary source file for the TorchFold library, which implements dynamic batching for dynamic computation graphs in PyTorch.
  - **torchfold_test.py**: Contains test cases for the TorchFold library to ensure its functionality.
  - **__init__.py**: Initializes the torchfold package, making it importable in Python.

- **examples/**: Contains example scripts demonstrating the usage of TorchFold.
  - **snli/spinn-example.py**: An example implementation of the SPINN (Stack-augmented Parser-Interpreter Neural Network) model for the Stanford Natural Language Inference (SNLI) dataset.

- **setup.py**: A setup script for installing the TorchFold library as a Python package.

- **.gitignore**: Specifies intentionally untracked files to ignore in the repository.

- **LICENSE**: The license file detailing the terms under which the software can be used and distributed.

- **logo.jpg**: A logo image associated with the project.

- **README.md**: The main documentation file providing an overview and instructions for the project.

## Additional Notes

### Compatibility and Dependencies

TorchFold is designed to work seamlessly with PyTorch, providing dynamic batching capabilities akin to TensorFlow Fold. Ensure that you have a compatible version of PyTorch installed in your environment to utilize this library effectively. The library does not explicitly list version dependencies in the codebase, so it is recommended to test with the latest stable version of PyTorch for optimal performance.

### Performance Considerations

When using TorchFold, be aware that the dynamic batching process optimizes computations by grouping operations. However, the efficiency can vary based on the structure of your data and the complexity of the neural network model. For recursive or tree-structured data, as shown in the examples, TorchFold can significantly reduce computation time by batching similar operations together. Experiment with different batching strategies using the `nobatch()` method on nodes where batching may not be beneficial.

### Debugging with Unfold

For debugging purposes, TorchFold provides an `Unfold` class that serves as a drop-in replacement for `Fold`. Unlike `Fold`, which batches computations, `Unfold` executes operations immediately without batching. This can be useful for tracing the flow of data through your model and identifying issues in the computation graph. Switch to `Unfold` by initializing it with your neural network module and using it in place of `Fold` during development.

### Use Cases

TorchFold is particularly suited for scenarios involving recursive neural networks or tree-structured data processing, such as natural language processing tasks (e.g., parsing sentences into tree structures as demonstrated in the provided SNLI example). If your project involves operations on variable-sized inputs or hierarchical data, TorchFold can help manage the complexity of batching these operations efficiently.

### Limitations

While TorchFold offers powerful dynamic batching, it requires all arguments to operations to be of specific types (`Tensor`, `Variable`, `int`, or `Node`). Mixing incompatible types or using unsupported data structures may result in errors. Additionally, the library assumes that the operations defined in your neural network module are compatible with batched inputs; ensure your model is designed accordingly to avoid runtime issues.

### Community and Support

TorchFold is an open-source project, and community contributions are welcome. If you encounter issues or have suggestions for improvements, consider checking the repository on GitHub for updates or opening an issue. The project includes a citation format for academic use, indicating its relevance in research contexts—please use the provided citation if you incorporate TorchFold into your published work.

## Contributing

We welcome contributions to TorchFold to enhance its functionality and improve its performance. If you'd like to contribute, please follow these guidelines to ensure a smooth collaboration process.

### How to Contribute
1. **Fork the Repository**: Start by forking the repository on GitHub and clone it to your local machine.
2. **Make Changes**: Implement your changes or additions in your forked repository. Ensure your code aligns with the project's purpose and structure.
3. **Test Your Changes**: Make sure to test your code to avoid introducing bugs. If possible, add tests to cover new functionality.
4. **Submit a Pull Request**: Once your changes are ready, submit a pull request to the main repository. Provide a clear description of your changes and the motivation behind them.

### Contribution Guidelines
- **Code Style**: Follow Python PEP 8 style guidelines for code formatting and structure to maintain consistency across the codebase.
- **Documentation**: Update or add documentation for any new features or changes to ensure users understand how to use them.
- **Testing**: Contributions should include relevant tests to verify functionality. Ensure existing tests pass before submitting your pull request.
- **Licensing**: By contributing, you agree that your contributions will be licensed under the Apache License, Version 2.0, as specified in the repository.

Thank you for considering contributing to TorchFold. Your efforts help make this project better for everyone!

## License

This project is licensed under the Apache License, Version 2.0. You can view the full license terms and conditions in the [LICENSE](./LICENSE) file.
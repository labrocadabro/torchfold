# TorchFold: Dynamic Batching for PyTorch Neural Networks

## Project Overview

This project, **TorchFold**, is a PyTorch library designed to enable dynamic batching for neural network computations. Its primary purpose is to optimize the processing of variable-sized inputs in deep learning models by grouping operations into batches dynamically during computation.

### Key Purpose and Problems Solved
TorchFold addresses the challenge of handling variable-sized data structures (like trees or graphs) in neural networks, which typically require fixed-size inputs for batch processing. By implementing a folding mechanism, it allows developers to define computations over recursive or irregular structures and batch them efficiently for GPU acceleration with PyTorch. This is particularly useful in natural language processing (NLP) tasks and other domains where data doesn't naturally fit into fixed-size tensors.

### Key Features and Benefits
- **Dynamic Batching**: Automatically groups operations into batches at runtime, optimizing computational efficiency on GPUs.
- **Support for Variable-Sized Inputs**: Facilitates the processing of complex, nested data structures without manual padding or reshaping.
- **Seamless PyTorch Integration**: Works directly with PyTorch tensors and variables, making it easy to incorporate into existing models.
- **Debugging Support**: Includes an `Unfold` class for immediate computation during debugging, bypassing batching for easier error tracking.
- **Performance Optimization**: Reduces computational overhead by minimizing redundant operations through intelligent batching.

TorchFold empowers developers to build more flexible and efficient deep learning models, especially for tasks involving sequential or hierarchical data, enhancing both development speed and model performance.

## API Reference

### TorchFold Library API

The TorchFold library provides tools for dynamic batching in PyTorch, enabling efficient computation over variable-sized structures like trees or graphs. Below is a comprehensive list of all publicly exported classes and their methods from the library. Each item includes its name, description, signature with parameter and return type details, and example usage.

#### Fold Class

- **Name**: `Fold`
- **Description**: A class for managing dynamic batching of operations in neural networks. It constructs a computation graph by adding operations and their dependencies, then applies these operations in a batched manner to a neural module.
- **Signature**:
  ```python
  Fold(volatile=False, cuda=False)
  ```
  - **Parameters**:
    - `volatile` (bool, optional): If True, disables gradient computation for the operations. Default is False.
    - `cuda` (bool, optional): If True, operations are performed on CUDA tensors. Default is False.
  - **Return Value**: An instance of the `Fold` class.
- **Example Usage**:
  ```python
  fold = Fold(volatile=True, cuda=True)
  ```

- **Method**: `cuda`
- **Description**: Enables CUDA support for the Fold instance.
- **Signature**:
  ```python
  cuda()
  ```
  - **Parameters**: None
  - **Return Value**: Self (the `Fold` instance with CUDA enabled).
- **Example Usage**:
  ```python
  fold = Fold().cuda()
  ```

- **Method**: `add`
- **Description**: Adds an operation to the fold computation graph along with its arguments. This method registers an operation to be executed later during the `apply` method.
- **Signature**:
  ```python
  add(op, *args)
  ```
  - **Parameters**:
    - `op` (str): The name of the operation (should correspond to a method in the neural module).
    - `*args`: Variable arguments which can be instances of `Fold.Node`, `int`, `torch.Tensor`, or `torch.autograd.Variable`.
  - **Return Value**: A `Fold.Node` object representing the node in the computation graph.
- **Example Usage**:
  ```python
  fold = Fold()
  node1 = fold.add('some_operation', 1, 2)
  node2 = fold.add('another_operation', node1)
  ```

- **Method**: `apply`
- **Description**: Applies the constructed computation graph to a given neural module, executing all operations in a batched manner.
- **Signature**:
  ```python
  apply(nn, nodes)
  ```
  - **Parameters**:
    - `nn` (object): A neural module with methods corresponding to the operations added via `add`.
    - `nodes` (list or tuple): A list or tuple of lists of `Fold.Node` objects or other arguments representing the final outputs to retrieve.
  - **Return Value**: The batched results of the computation as a list of tensors.
- **Example Usage**:
  ```python
  class NeuralModule:
      def some_operation(self, x, y):
          return x + y

  fold = Fold()
  node = fold.add('some_operation', 1, 2)
  result = fold.apply(NeuralModule(), [[node]])
  print(result)  # Outputs the result of the operation
  ```

#### Unfold Class

- **Name**: `Unfold`
- **Description**: A debugging alternative to `Fold`, performing computations immediately rather than batching them. Useful for testing and debugging neural network operations without dynamic batching.
- **Signature**:
  ```python
  Unfold(nn, volatile=False, cuda=False)
  ```
  - **Parameters**:
    - `nn` (object): A neural module with methods corresponding to the operations to be executed.
    - `volatile` (bool, optional): If True, disables gradient computation for the operations. Default is False.
    - `cuda` (bool, optional): If True, operations are performed on CUDA tensors. Default is False.
  - **Return Value**: An instance of the `Unfold` class.
- **Example Usage**:
  ```python
  class NeuralModule:
      def some_operation(self, x, y):
          return x + y

  unfold = Unfold(NeuralModule(), volatile=True, cuda=True)
  ```

- **Method**: `cuda`
- **Description**: Enables CUDA support for the Unfold instance.
- **Signature**:
  ```python
  cuda()
  ```
  - **Parameters**: None
  - **Return Value**: Self (the `Unfold` instance with CUDA enabled).
- **Example Usage**:
  ```python
  unfold = Unfold(NeuralModule()).cuda()
  ```

- **Method**: `add`
- **Description**: Immediately executes an operation on the provided neural module with the given arguments, returning the result wrapped in an `Unfold.Node`.
- **Signature**:
  ```python
  add(op, *args)
  ```
  - **Parameters**:
    - `op` (str): The name of the operation (should correspond to a method in the neural module).
    - `*args`: Variable arguments which can be instances of `Unfold.Node`, `int`, `torch.Tensor`, or `torch.autograd.Variable`.
  - **Return Value**: An `Unfold.Node` object containing the result of the operation.
- **Example Usage**:
  ```python
  class NeuralModule:
      def some_operation(self, x, y):
          return x + y

  unfold = Unfold(NeuralModule())
  node = unfold.add('some_operation', 1, 2)
  print(node.tensor)  # Outputs the result tensor
  ```

- **Method**: `apply`
- **Description**: Processes a list of nodes or arguments to return concatenated results. Primarily used to maintain API compatibility with `Fold`.
- **Signature**:
  ```python
  apply(nn, nodes)
  ```
  - **Parameters**:
    - `nn` (object): A neural module, expected to be the same as passed to the constructor.
    - `nodes` (list or tuple): A list or tuple of lists of `Unfold.Node` objects or other arguments.
  - **Return Value**: A list of concatenated tensors representing the results.
- **Example Usage**:
  ```python
  class NeuralModule:
      def some_operation(self, x, y):
          return x + y

  unfold = Unfold(NeuralModule())
  node = unfold.add('some_operation', 1, 2)
  result = unfold.apply(NeuralModule(), [[node]])
  print(result)  # Outputs the concatenated result
  ```


## Project Structure

This section outlines the organization of the TorchFold repository, detailing the key directories and files that constitute the project.

#### Directory and File Layout

- **torchfold/**: This directory contains the core implementation of the TorchFold library.
  - **__init__.py**: Initializes the TorchFold package and exposes the main classes, `Fold` and `Unfold`, for dynamic batching.
  - **torchfold.py**: The primary source file for the library, implementing the `Fold` class for dynamic batching and the `Unfold` class for debugging purposes.
  - **torchfold_test.py**: Contains test cases for the TorchFold library to ensure functionality and correctness.

- **examples/**: Houses example implementations demonstrating the usage of TorchFold.
  - **snli/spinn-example.py**: An example script showcasing the application of TorchFold in a TreeLSTM-based model for the Stanford Natural Language Inference (SNLI) task. It includes both regular and folded computation methods for tree-structured data.

- **Root Files**:
  - **setup.py**: The setup script for installing the TorchFold package, specifying metadata, dependencies, and installation instructions.
  - **.gitignore**: Specifies intentionally untracked files to ignore in the repository.
  - **LICENSE**: Contains the licensing information for the project (Apache License, Version 2.0).
  - **README.md**: The main documentation file providing an overview, installation instructions, and usage examples for TorchFold.
  - **logo.jpg**: An image file used in the README for branding purposes.

This structure ensures that the core functionality, examples, and supporting files are organized for ease of access and understanding.

## Technologies Used

- **PyTorch**: A deep learning framework used for dynamic batching and neural network computations in this project.
- **Python**: The programming language used for the implementation of the torchfold library.

## Additional Notes

### Key Features and Limitations

TorchFold is a library designed to enable dynamic batching in PyTorch, providing a mechanism to optimize computation graphs for neural networks dealing with variable-sized or tree-structured data. Below are some key aspects and considerations for users:

- **Dynamic Batching**: TorchFold allows for dynamic batching of operations, which is particularly useful when working with data structures like trees or graphs where the size and shape of data can vary. This is achieved through the `Fold` class, which constructs an optimized computation graph and batches operations dynamically during execution.
- **Ease of Use**: The library offers a simple interface where users replace direct calls to neural network modules with `f.add()` calls, enabling seamless integration with existing PyTorch code.
- **Debugging Support**: The `Unfold` class provides a way to perform computations immediately without batching, which is helpful for debugging purposes.
- **Performance Considerations**: While dynamic batching can significantly improve performance by reducing redundant computations, it may introduce complexity in understanding the batched operations. Users should be aware of potential overhead in constructing the computation graph for very small datasets or simple models.
- **Hardware Compatibility**: TorchFold supports CUDA for GPU acceleration, allowing users to leverage GPU hardware for faster computation if available.
- **Limitations**: The library requires that all arguments to operations be of specific types (`Tensor`, `Variable`, `int`, or `Node`), which might restrict flexibility in some use cases. Additionally, the current implementation might not handle extremely large graphs efficiently due to the overhead of graph construction.

### Use Case Example

An example provided in the repository demonstrates the application of TorchFold in processing tree-structured data using a SPINN (Stack-augmented Parser-Interpreter Neural Network) model for natural language inference tasks (SNLI dataset). This example illustrates how TorchFold can handle recursive tree structures efficiently with dynamic batching, reducing the computational overhead compared to processing each tree individually.

### Development and Testing

The repository includes test files that validate the functionality of TorchFold, particularly in scenarios involving RNNs and tree-structured data processing. These tests ensure that dynamic batching works as expected and that the library handles various edge cases, such as non-batched operations.

### Community and Contributions

TorchFold is an open-source project, and community contributions are encouraged. If you encounter issues or have ideas for enhancements, consider checking the repository for existing issues or submitting a pull request with your improvements. The project includes a citation reference for academic use, ensuring proper attribution if used in research.

### Contact and Support

For additional support or to connect with other users of TorchFold, refer to the blog post linked in the repository or explore related discussions in PyTorch community forums. If you have specific questions about the codebase, feel free to open an issue on the GitHub repository.

## Contributing

Thank you for your interest in contributing to TorchFold! We welcome contributions from the community to help improve and expand this project. Here's how you can get started with contributing:

### How to Contribute

1. **Fork the Repository**: Start by forking the repository to your own GitHub account.
2. **Clone the Repository**: Clone the forked repository to your local machine to work on the changes.
3. **Make Changes**: Implement your changes or additions to the codebase. Ensure that your code aligns with the project's purpose of providing dynamic batching with a simple interface.
4. **Test Your Changes**: Make sure to test your changes locally to verify that they work as expected and do not introduce any bugs.
5. **Commit Your Changes**: Commit your changes with a clear and descriptive commit message.
6. **Push to Your Fork**: Push your changes to your forked repository on GitHub.
7. **Submit a Pull Request**: Create a pull request from your fork to the main repository. Provide a detailed description of your changes and the motivation behind them.

### Contribution Guidelines

- **Code Style**: Please follow standard Python coding conventions (PEP 8) for consistency across the codebase. Use meaningful variable names and include comments where necessary to explain complex logic.
- **Testing**: Ensure that any new functionality or changes are accompanied by appropriate tests. If you're fixing a bug, include a test case that demonstrates the fix.
- **Documentation**: Update any relevant documentation if your changes affect the usage or behavior of the library. This includes inline code comments and any example scripts if applicable.
- **Scope**: Contributions should align with the project's focus on dynamic batching with PyTorch. If you're unsure about the relevance of your idea, feel free to open an issue to discuss it before submitting a pull request.

We review all contributions and may provide feedback or request changes before merging. Thank you for helping make TorchFold better!

## License

This project is licensed under the Apache License, Version 2.0. You can view the full license text in the [LICENSE](./LICENSE) file in this repository.
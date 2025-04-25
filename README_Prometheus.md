# TorchFold: Dynamic Batching Library for PyTorch

## Project Overview

This project, TorchFold, is a PyTorch library designed to implement dynamic batching with a simple and intuitive interface. Inspired by TensorFlow Fold, it optimizes computations by dynamically batching operations, which is particularly useful for handling variable-sized or recursive data structures like trees or graphs in neural networks.

### Main Purpose and Problems Solved
TorchFold addresses the challenge of efficiently processing data with irregular structures in deep learning models. Traditional batching methods often struggle with non-uniform data, leading to inefficient computation or wasted resources. TorchFold solves this by allowing developers to construct optimized computations that adapt to the data's structure at runtime, improving performance and resource utilization in PyTorch-based projects.

### Key Features and Benefits
- **Dynamic Batching**: Automatically batches operations based on the input data structure, reducing computational overhead.
- **Simple Interface**: Easily integrate with existing PyTorch code by replacing direct calls to neural network modules with TorchFold's `add` method.
- **Optimized Computation**: Constructs and executes optimized computation graphs, enhancing performance for complex, recursive operations.
- **Flexibility**: Supports a variety of neural network architectures and data types, making it suitable for diverse deep learning tasks.

TorchFold is a powerful tool for researchers and developers working on natural language processing, graph neural networks, or any domain where data structures are not fixed or uniform, enabling more efficient and scalable model training and inference.

## Getting Started, Installation, and Setup

### Quick Start Guide

To quickly get started with TorchFold, follow these steps:

1. Install TorchFold using pip:
   ```bash
   pip install torchfold
   ```
2. Use the dynamic batching functionality in your PyTorch project by integrating the `torchfold.Fold()` class. Here's a basic example of how to structure your code:
   ```python
   import torchfold
   import torch.nn as nn

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
           super(Model, self).__init__()
           # Define your model components

       def leaf(self, leaf):
           # Process leaf node
           pass

       def child(self, prev, child):
           # Process child node
           pass

   # Construct computation graph
   res = dfs(my_tree)
   model = Model(...)
   # Execute with dynamic batching
   f.apply(model, [[res]])
   ```

For a more detailed example, refer to the provided `examples/snli/spinn-example.py` script in the repository.

### Installation Instructions

TorchFold can be installed easily via pip, which will handle all necessary dependencies:

```bash
pip install torchfold
```

Ensure you have PyTorch installed in your environment as TorchFold relies on it for functionality. If PyTorch is not installed, you can install it by following the instructions on the [PyTorch official website](https://pytorch.org/get-started/locally/).

#### Platform-Specific Instructions
- **Linux/macOS/Windows**: The pip installation should work across all major platforms without additional setup. Ensure your system has Python 3 and pip installed.
- **CUDA Support**: If you plan to use GPU acceleration, make sure to install the CUDA version of PyTorch as per your CUDA version. The `spinn-example.py` script includes options to enable or disable CUDA (`--no-cuda` flag).

### Setup for Development

If you want to contribute to TorchFold or run it from source:

1. Clone the repository:
   ```bash
   git clone https://github.com/nearai/torchfold.git
   cd torchfold
   ```
2. Install in development mode:
   ```bash
   python setup.py develop
   ```
   This will link the local package to your Python environment, allowing changes to take effect without reinstalling.
3. Verify the installation by running the example:
   ```bash
   python examples/snli/spinn-example.py
   ```
   Note that running the example requires additional dependencies like `torchtext`. Install them via pip if needed:
   ```bash
   pip install torchtext
   ```

### Building for Production

TorchFold is primarily a library to be used within other projects, so there is no specific production build process. After installation via pip, it can be imported and used in any Python script or application. If you're integrating TorchFold into a larger project, ensure all dependencies (like PyTorch) are specified in your project's requirements.

For deployment, consider containerizing your application using Docker to ensure consistency across environments. Include the `pip install torchfold` command in your Dockerfile or requirements file.

## Features / Capabilities

### Core Features

**TorchFold** is a library designed to enable dynamic batching in PyTorch, allowing for efficient processing of variable-sized inputs such as trees or graphs. Below are the primary features and capabilities of this project:

- **Dynamic Batching**: TorchFold facilitates dynamic batching, which helps optimize computation by grouping operations of varying sizes into batches. This is particularly useful for neural network architectures dealing with recursive or hierarchical data structures.
- **Fold and Unfold Classes**: The library provides two main classes:
  - **`Fold`**: Used for batching operations dynamically. It allows operations to be grouped and computed in a batched manner, optimizing performance by reducing overhead from individual computations.
  - **`Unfold`**: A debugging counterpart to `Fold`, which performs computations immediately without batching, useful for verifying correctness during development.
- **Support for Complex Neural Architectures**: TorchFold can be integrated with neural network modules to handle complex operations, such as tree-structured data processing with models like TreeLSTM.
- **CUDA Support**: The library supports GPU acceleration with CUDA, enabling faster computation on compatible hardware.
- **Flexible Node Operations**: It provides mechanisms to split results from operations (for functions returning multiple values) and control batching behavior with options like `nobatch()`.

### Example Use Case

An example implementation is provided for the **SPINN (Stack-augmented Parser-Interpreter Neural Network)** model, demonstrating how TorchFold can be used for natural language inference tasks with tree-structured data:

- **SNLI Example**: Located in `examples/snli/spinn-example.py`, this script showcases the use of TorchFold to encode tree structures for the Stanford Natural Language Inference (SNLI) dataset. It compares regular tree encoding with folded (batched) encoding, highlighting performance improvements through dynamic batching.

This library is particularly suited for researchers and developers working on machine learning models that require efficient handling of non-uniform data structures.

## Usage Examples

### Basic Usage with SPINN Example

The repository includes an example implementation of a Stack-augmented Parser-Interpreter Neural Network (SPINN) for natural language inference using the SNLI dataset. This example demonstrates how to use `torchfold` for dynamic batching in PyTorch. Below are the steps to run the example:

1. **Ensure Dependencies are Installed**: Make sure you have PyTorch, torchtext, and other required libraries installed. You can install `torchfold` directly from this repository using `pip install .` from the root directory.

2. **Navigate to the Examples Directory**: The SPINN example is located in the `examples/snli/` folder.
   
   ```bash
   cd examples/snli
   ```

3. **Run the SPINN Example**: Execute the `spinn-example.py` script to train the model on the SNLI dataset. You can choose to enable dynamic batching with the `--fold` flag.
   
   - Run without dynamic batching:
     ```bash
     python spinn-example.py
     ```
   
   - Run with dynamic batching using `torchfold`:
     ```bash
     python spinn-example.py --fold
     ```

4. **Adjust Batch Size (Optional)**: You can modify the batch size using the `--batch_size` argument. For example, to set a batch size of 64:
   
   ```bash
   python spinn-example.py --fold --batch_size 64
   ```

5. **Disable CUDA (Optional)**: If you do not have a CUDA-enabled GPU or prefer to run on CPU, use the `--no-cuda` flag:
   
   ```bash
   python spinn-example.py --fold --no-cuda
   ```

### Understanding the Output

While running the script, the program will output the average time taken per iteration every 10 iterations. This can help you monitor the training performance, especially to compare the efficiency of dynamic batching with `torchfold` versus regular processing.

### Notes

- The provided `spinn-example.py` is a demonstration and not a full implementation of the SPINN model.
- Ensure you have the SNLI dataset accessible through `torchtext.datasets.SNLI` as the script downloads and processes it automatically during execution.

## Project Structure

This section outlines the layout of the project, highlighting the purpose of key directories and files.

### Directory and File Overview

- **Root Directory**: Contains essential project files such as:
  - `.gitignore`: Specifies intentionally untracked files to ignore.
  - `LICENSE`: Contains the licensing information for the project.
  - `README.md`: The main documentation file for the project.
  - `README_Prometheus.md`: Additional documentation, possibly specific to Prometheus integration or context.
  - `logo.jpg`: A logo image associated with the project.
  - `setup.py`: A setup script for installing the project as a Python package.

- **examples/**: A directory for example scripts or usage demonstrations.
  - `examples/snli/spinn-example.py`: Likely an example script related to the SNLI dataset or SPINN model, demonstrating how to use the project's functionality.

- **torchfold/**: The core package directory, containing the main codebase.
  - `torchfold/__init__.py`: Marks this directory as a Python package.
  - `torchfold/torchfold.py`: The primary source code file for the TorchFold library, likely containing the main implementation.
  - `torchfold/torchfold_test.py`: Contains test cases for the TorchFold library to ensure functionality and correctness.

## Technologies Used

This project leverages the following major technologies, frameworks, and libraries:

- **PyTorch**: An open-source machine learning library for Python, used for dynamic batching and neural network operations in this project. TorchFold is built on top of PyTorch to facilitate dynamic computation graphs.

No other significant frameworks, SDKs, or tools were identified in the current codebase.

## Additional Notes

### Performance Considerations

When using `torchfold` for dynamic batching in PyTorch, be aware that the efficiency of batching operations largely depends on the structure of your data and the nature of the operations being performed. The `Fold` class is optimized to group operations into batches dynamically, which can significantly improve performance for recursive or variable-sized data structures like trees or graphs. However, for very small batch sizes or highly irregular data, the overhead of managing the computation graph might outweigh the benefits of batching. In such scenarios, consider benchmarking both with and without dynamic batching to determine the best approach for your use case.

### Debugging with Unfold

The `torchfold` library includes an `Unfold` class as an alternative to `Fold`, which is particularly useful for debugging. Unlike `Fold`, which batches operations for efficiency, `Unfold` executes computations immediately without batching. This allows you to step through the computation graph to identify issues in your model or data processing pipeline. To use it, simply initialize `Unfold` with your neural network module and replace `Fold` in your code for debugging purposes.

### CUDA Support

`torchfold` supports GPU acceleration via CUDA, provided your PyTorch installation is configured for it. To enable CUDA, call the `.cuda()` method on your `Fold` or `Unfold` instance before adding operations. Ensure that your model and input data are also moved to the GPU if necessary. This can significantly speed up computations for large datasets or complex models.

### Limitations

- **Argument Types**: All arguments passed to the `add` method in `Fold` must be of type `Tensor`, `Variable`, `int`, or a `Fold.Node`. Passing incompatible types will result in a `ValueError`.
- **Batch Consistency**: When using `nobatch()` on a node, ensure that only one such node is used per operation to avoid runtime errors.
- **Dynamic Memory Usage**: The dynamic nature of batching may lead to fluctuating memory usage during runtime, which could be a concern in resource-constrained environments. Monitor memory usage during development to avoid unexpected issues.

### Community and Support

For more in-depth information on implementation details or to discuss specific use cases, refer to the blog post linked in the original project documentation or visit the repository on GitHub. The community welcomes bug reports, feature requests, and contributions. Feel free to open issues or submit pull requests as outlined in the contributing guidelines.

### Example Usage

A practical example is provided in the `examples/snli/` directory, showcasing `torchfold` in a natural language inference task using a SPINN (Stack-augmented Parser-Interpreter Neural Network) model. This example can serve as a starting point for integrating dynamic batching into your own projects, especially if you're working with tree-structured data or recursive neural networks.

## Contributing

We welcome contributions from the community to help improve TorchFold. Whether you’re fixing bugs, adding new features, or improving documentation, your efforts are appreciated.

### How to Contribute
1. **Fork the Repository**: Start by forking the repository and cloning it to your local machine.
2. **Make Changes**: Implement your changes or additions in your forked repository. Ensure your code adheres to the existing style and structure.
3. **Test Your Changes**: Make sure to test your modifications. If you're adding new functionality, include appropriate test cases. You can refer to the existing test file `torchfold_test.py` for guidance on how tests are structured.
4. **Commit Your Changes**: Write clear, concise commit messages that describe the purpose of your changes.
5. **Push to Your Fork**: Push your changes to your forked repository.
6. **Submit a Pull Request**: Create a pull request from your fork to the main repository. Provide a detailed description of your changes and the motivation behind them.

### Contribution Guidelines
- **Code Style**: Follow the coding style used in the existing codebase. Ensure your code is clean, well-documented, and follows Python best practices (e.g., PEP 8).
- **Testing**: All contributions should include or update tests to cover the new or modified functionality. Tests should pass before submitting a pull request.
- **Documentation**: If your contribution adds or modifies functionality, update any relevant documentation to reflect these changes.
- **Issue Tracking**: If your contribution addresses a specific issue, reference the issue number in your pull request description.

Thank you for contributing to TorchFold and helping make it better!

## License

This project is licensed under the Apache License Version 2.0. You can view the full license terms and conditions in the [LICENSE](./LICENSE) file.
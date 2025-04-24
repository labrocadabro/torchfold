# TorchFold: Dynamic Batching Library for PyTorch

## Project Overview

This repository contains **TorchFold**, a library for dynamic batching in PyTorch, inspired by TensorFlow Fold. The main purpose of TorchFold is to optimize neural network computations by dynamically batching operations, which helps improve performance when processing variable-sized or tree-structured data, such as in natural language processing tasks or recursive neural networks.

### Key Features
- **Dynamic Batching**: Automatically batches computations for efficiency, even with irregular data structures.
- **Simple Interface**: Easily integrate with existing PyTorch code by replacing direct calls to neural network modules with a simple `add` method.
- **Optimized Execution**: Constructs and executes an optimized computation graph tailored to the input data.
- **Debugging Support**: Includes an `Unfold` class for immediate computation during debugging, bypassing batching.

### Benefits
- **Performance Improvement**: Reduces computational overhead by batching operations, making it ideal for processing complex, non-uniform data.
- **Ease of Use**: Minimal code changes are required to implement dynamic batching in PyTorch models.
- **Flexibility**: Supports a wide range of applications, including tree-based models and recursive structures, with seamless integration into PyTorch workflows.

## Getting Started, Installation, and Setup

### Quick Start Guide

To quickly get started with TorchFold, follow these steps for a basic usage example. This guide assumes you have already installed the package (see Installation and Setup below for detailed instructions).

1. **Import TorchFold**: Start by importing the `torchfold` module in your Python script.
   ```python
   import torchfold
   import torch.nn as nn
   ```

2. **Define Your Computation**: Use `torchfold.Fold()` to dynamically batch operations. Replace direct calls to your neural network module with `f.add()` calls to construct an optimized computation graph.
   ```python
   f = torchfold.Fold()
   
   def dfs(node):
       if is_leaf(node):
           return f.add('leaf', node)
       else:
           prev = f.add('init')
           for child in children(node):
               prev = f.add('child', prev, child)
           return prev
   ```

3. **Define Your Model**: Create a model class inheriting from `nn.Module` with methods corresponding to the operations defined in `f.add()`.
   ```python
   class Model(nn.Module):
       def __init__(self, ...):
           super(Model, self).__init__()
           # Initialize layers here

       def leaf(self, leaf):
           # Process leaf node
           pass

       def child(self, prev, child):
           # Process child node
           pass
   ```

4. **Execute the Computation**: Apply the computation on your model with input data.
   ```python
   res = dfs(my_tree)
   model = Model(...)
   f.apply(model, [[res]])
   ```

This quick start demonstrates the basic usage of TorchFold for dynamic batching in PyTorch. For more complex examples, such as implementing a SPINN model, refer to the `examples` directory in the repository.

---

### Installation and Setup

#### Prerequisites
- Python 3.x
- PyTorch (compatible version based on your system)
- pip (Python package manager)

#### Installation Steps
TorchFold can be easily installed using pip. Follow these steps to set up the library:

1. **Install TorchFold**:
   Run the following command to install TorchFold directly from PyPI:
   ```bash
   pip install torchfold
   ```

2. **Verify Installation**:
   After installation, you can verify that TorchFold is installed by running:
   ```python
   import torchfold
   print(torchfold.__version__)
   ```

#### Development Setup
If you wish to work on the TorchFold codebase or run examples, follow these steps to set up a development environment:

1. **Clone the Repository**:
   Clone the repository to your local machine if you haven't already:
   ```bash
   git clone https://github.com/nearai/torchfold.git
   cd torchfold
   ```

2. **Install in Development Mode**:
   Use pip to install the package in editable mode, which allows changes to the code to take effect without reinstalling:
   ```bash
   pip install -e .
   ```

3. **Running Examples**:
   Navigate to the `examples` directory to explore sample implementations, such as the SPINN model for natural language inference:
   ```bash
   cd examples/snli
   python spinn-example.py
   ```
   Note: Ensure you have additional dependencies like `torchtext` installed to run the examples. You can install them via pip:
   ```bash
   pip install torchtext
   ```

#### Platform-Specific Instructions
- **Linux/macOS/Windows**: The installation process is platform-agnostic as long as Python and PyTorch are properly set up. Ensure that your PyTorch installation matches your system's architecture (CPU/GPU support).
- **CUDA Support**: If you plan to use GPU acceleration, ensure that your PyTorch installation supports CUDA and that the appropriate drivers are installed on your system. You can check CUDA availability in the SPINN example by running with the default arguments (CUDA is enabled by default unless `--no-cuda` is specified).

#### Building for Production
TorchFold is a library intended for integration into larger PyTorch projects. There is no separate production build process for TorchFold itself. To use it in a production environment:
- Ensure that your application code is optimized and tested with TorchFold.
- Package your application using standard Python deployment tools (e.g., Docker, PyInstaller) if needed, including TorchFold as a dependency in your `requirements.txt` or similar file:
  ```
  torchfold==0.1.0
  ```

For further assistance or to report issues, refer to the repository's issue tracker on GitHub.

## Project Structure

This section outlines the organization of the repository, highlighting key directories and files that are essential for understanding the project structure.

### Key Directories and Files

- **torchfold/**: The core directory containing the main implementation of the TorchFold library.
  - **torchfold.py**: The primary source file for the TorchFold library, which likely contains the core functionality for dynamic batching and folding operations in PyTorch.
  - **torchfold_test.py**: Contains test cases for the TorchFold library, ensuring the functionality works as expected.
  - **__init__.py**: Marks the directory as a Python package, potentially including initialization code for the library.

- **examples/**: A directory with example usage of the TorchFold library.
  - **snli/spinn-example.py**: An example script demonstrating the application of TorchFold, possibly for the Stanford Natural Language Inference (SNLI) dataset using a SPINN (Stack-augmented Parser-Interpreter Neural Network) model.

- **setup.py**: A setup script for installing the TorchFold library as a Python package, facilitating easy distribution and installation.

- **LICENSE**: The file specifying the licensing terms under which the TorchFold library is distributed.

- **README.md**: The main documentation file providing an overview, installation instructions, and other relevant information about the project.

- **.gitignore**: Specifies intentionally untracked files to ignore in version control, such as temporary files or build artifacts.

- **logo.jpg**: A graphical asset, likely used for branding or visual representation of the project in documentation or presentations.

## Additional Notes

### Compatibility and Dependencies

TorchFold is designed to work seamlessly with PyTorch, leveraging its capabilities for dynamic batching. Ensure that you have a compatible version of PyTorch installed before using this library. The library is implemented in Python and is available for installation via pip.

### Usage Tips

- **Dynamic Batching**: TorchFold simplifies dynamic batching by allowing you to replace direct calls to neural network modules with `f.add()` calls. This constructs an optimized computation graph that can be executed with dynamic batching on `f.apply()`.
- **Debugging with Unfold**: For debugging purposes, use the `Unfold` class which performs computations immediately without batching, helping to isolate issues in your model logic.
- **CUDA Support**: If you're using GPU acceleration, make sure to enable CUDA support by calling `.cuda()` on your `Fold` instance.

### Example Usage

An example implementation can be found in the `examples/snli/spinn-example.py` file, which demonstrates how to use TorchFold for processing tree-structured data with a SPINN (Stack-augmented Parser-Interpreter Neural Network) model for natural language inference tasks. This example includes both regular and folded computation methods to showcase the performance benefits of dynamic batching.

### Limitations

- **Argument Types**: All arguments passed to `Fold.add()` must be of type `Tensor`, `Variable`, `int`, or `Node`. Mixing incompatible types may result in errors.
- **Batching Constraints**: When using non-batched nodes, ensure that only one instance is used per operation to avoid errors during computation.

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

### Contact and Support

For issues, questions, or contributions, please refer to the GitHub repository at [https://github.com/nearai/torchfold](https://github.com/nearai/torchfold) or contact the author at [illia@near.ai](mailto:illia@near.ai).

### Further Reading

For a deeper understanding of dynamic batching and TorchFold's implementation, check out the associated blog post at [http://near.ai/articles/2017-09-06-PyTorch-Dynamic-Batching/](http://near.ai/articles/2017-09-06-PyTorch-Dynamic-Batching/).

## Contributing

Thank you for your interest in contributing to TorchFold! We welcome contributions from the community to help improve and expand this project. Here's how you can get involved:

### How to Contribute
1. **Fork the Repository**: Start by forking the repository on GitHub and cloning it to your local machine.
2. **Make Changes**: Implement your changes or new features in your fork. Ensure your code aligns with the project's purpose of providing dynamic batching for PyTorch.
3. **Test Your Changes**: Make sure to test your modifications to ensure they work as expected and do not introduce bugs. Currently, there are no specific testing frameworks or guidelines provided in the repository, so use your best judgment to validate your code.
4. **Submit a Pull Request**: Once your changes are ready, submit a pull request to the main repository. Provide a clear description of your changes and the motivation behind them.

### Contribution Guidelines
- **Code Style**: There are no explicit code style guidelines mentioned in the repository. However, we encourage you to follow standard Python and PyTorch best practices for readability and maintainability.
- **Documentation**: If your contribution adds new features or modifies existing ones, please update any relevant documentation or examples to reflect these changes.
- **Compatibility**: Ensure that your code is compatible with the latest version of PyTorch and does not break existing functionality.

We appreciate all contributions, whether they are bug fixes, feature additions, or improvements to documentation. If you have any questions or need assistance, feel free to reach out by opening an issue on GitHub.

## License

This project is licensed under the Apache License Version 2.0. For full details, please see the [LICENSE](LICENSE) file in the repository.
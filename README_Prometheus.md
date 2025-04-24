# TorchFold: Dynamic Batching Library for PyTorch

## Project Overview

TorchFold is a PyTorch library designed to implement dynamic batching with a simple and intuitive interface. Inspired by TensorFlow Fold, it optimizes computations by dynamically batching operations, which is particularly useful for handling variable-sized inputs such as trees or graphs in neural networks.

### Main Purpose and Problems Solved
The primary purpose of TorchFold is to facilitate efficient computation in PyTorch when dealing with data structures that do not naturally fit into fixed-size batches. Traditional batching methods often struggle with nested or hierarchical data, leading to inefficient processing or the need for extensive padding. TorchFold solves this by allowing developers to define computations over such structures and automatically handling the batching process, thereby reducing memory usage and speeding up training and inference.

### Key Features and Benefits
- **Dynamic Batching**: Automatically batches operations for variable-sized inputs, optimizing performance without manual intervention.
- **Simple Interface**: Integrates seamlessly with PyTorch by replacing direct calls to neural network modules with a straightforward `add` method, making it easy to adopt in existing projects.
- **Optimized Computation**: Constructs an optimized computation graph tailored to the input structure, ensuring efficient execution.
- **Flexibility**: Supports a wide range of neural network architectures and data types, especially those involving recursive or hierarchical structures.
- **Debugging Support**: Includes an `Unfold` class for immediate computation during debugging, allowing developers to bypass dynamic batching when necessary.

TorchFold is an essential tool for researchers and developers working on tasks involving complex data structures in deep learning, offering both performance improvements and ease of use.

## Getting Started, Installation, and Setup

### Quick Start Guide

TorchFold is a library for dynamic batching in PyTorch, designed to optimize computation by grouping operations. Here's how to quickly get started:

1. **Install TorchFold**: Follow the installation instructions below to set up the library.
2. **Basic Usage**: Import TorchFold in your Python script to enable dynamic batching for your PyTorch models. Here's a simple usage example:
   ```python
   import torch
   import torchfold

   # Define your model or computations
   def compute_example(inputs):
       # Example computation
       return torch.sum(inputs)

   # Use TorchFold for dynamic batching
   folded_computation = torchfold.Fold()
   # Add your computations to the fold
   # Execute with dynamic batching
   ```
3. **Explore Examples**: Check the `examples/` directory for more detailed use cases, such as the SNLI example with SPINN.

For comprehensive instructions on installation and advanced setup, refer to the sections below.

### Installation

TorchFold can be installed easily using pip. Follow these steps to set up the library on your system:

1. **Prerequisites**: Ensure you have Python 3.x installed along with PyTorch. If PyTorch is not installed, you can install it via pip:
   ```bash
   pip install torch
   ```
2. **Install TorchFold**: Clone the repository and install the library using the following commands:
   ```bash
   git clone https://github.com/nearai/torchfold.git
   cd torchfold
   pip install .
   ```
   Alternatively, if you prefer not to clone the repository, you can install directly from the source (if available on PyPI in the future):
   ```bash
   pip install torchfold
   ```

**Platform-Specific Instructions**: TorchFold is platform-agnostic as long as Python and PyTorch are supported on your system. Ensure your PyTorch installation matches your operating system and hardware (CPU/GPU support).

### Setup for Development and Production

#### Development Setup
If you plan to contribute to TorchFold or modify the source code:
1. Clone the repository as shown above.
2. Install in development mode to make changes to the code:
   ```bash
   pip install -e .
   ```
3. Make your changes and test locally. You can run tests (if available) or use the provided examples in the `examples/` directory to validate your modifications.

#### Production Build
TorchFold does not require a separate production build step as it is a Python library. Once installed via pip, it is ready for production use. Ensure that your application environment has all dependencies (like PyTorch) installed with compatible versions:
1. Use a virtual environment or container (e.g., Docker) to isolate dependencies.
2. Install the library as described in the Installation section.
3. Integrate TorchFold into your production PyTorch models for optimized dynamic batching.

For additional details or troubleshooting, refer to the project repository or related blog posts linked in the setup.py metadata.

## Project Structure

This section provides an overview of the key directories and files within the TorchFold repository, which implements dynamic batching for PyTorch with a simple interface.

#### Key Directories and Files

- **`torchfold/`**: Core package directory containing the main implementation of dynamic batching.
  - **`torchfold/__init__.py`**: Entry point for the package, exposing `Fold` and `Unfold` classes for dynamic batching and debugging, respectively.
  - **`torchfold/torchfold.py`**: Main implementation file with the `Fold` class for constructing optimized computations and the `Unfold` class for immediate computation during debugging.
  - **`torchfold/torchfold_test.py`**: Contains test cases for validating the functionality of the TorchFold library.

- **`examples/`**: Directory with practical usage examples of TorchFold.
  - **`examples/snli/spinn-example.py`**: Example implementation of a SPINN (Stack-augmented Parser-Interpreter Neural Network) model using TorchFold for tree-structured data processing, specifically for the SNLI dataset.

- **`setup.py`**: Configuration file for installing the TorchFold package, specifying metadata like version, description, and dependencies.

- **`.gitignore`**: Specifies intentionally untracked files to ignore in the repository.

- **`LICENSE`**: Contains the licensing information for the project (Apache License, Version 2.0).

- **`logo.jpg`**: Image file used in the README for branding purposes.

- **`README.md`**: The main documentation file providing an overview, installation instructions, and usage examples for TorchFold.

## Additional Notes

### Key Features and Limitations

TorchFold is a library designed to enable dynamic batching in PyTorch, inspired by TensorFlow Fold. Its primary goal is to optimize computations by batching operations dynamically, which can significantly improve performance when working with variable-sized inputs such as trees or graphs. Here's a breakdown of important aspects to consider:

- **Dynamic Batching**: TorchFold allows for efficient processing of variable-sized data structures by batching operations on-the-fly. This is particularly useful in scenarios like natural language processing (NLP) tasks where input sizes can vary (e.g., sentence lengths or tree depths).
- **Ease of Use**: By simply replacing direct calls to neural network modules with `fold.add()` calls, users can integrate dynamic batching into their existing PyTorch models with minimal code changes.
- **Performance Optimization**: The library optimizes computation graphs to reduce overhead, as seen in the test cases where chunking operations are minimized to enhance efficiency.
- **Limitations**: While powerful for specific use cases, TorchFold requires a good understanding of the underlying computation graph to fully leverage its capabilities. It may not be suitable for all types of neural network architectures or data processing tasks. Additionally, debugging complex graphs can be challenging due to the abstraction introduced by dynamic batching.

### Use Cases

TorchFold is particularly suited for projects involving:

- **Natural Language Processing (NLP)**: As demonstrated in the provided example with the SPINN (Stack-augmented Parser-Interpreter Neural Network) model for the Stanford Natural Language Inference (SNLI) dataset, TorchFold excels in handling tree-structured data common in NLP tasks.
- **Graph-Based Learning**: Any application where data is represented as graphs or trees can benefit from TorchFold's dynamic batching to process nodes efficiently.
- **Custom Neural Architectures**: For researchers and developers designing custom neural network architectures with variable input sizes, TorchFold provides a way to maintain performance without manual batching.

### Compatibility

- **PyTorch Dependency**: TorchFold is built specifically for PyTorch, and users should ensure compatibility with their PyTorch version when integrating this library.
- **CUDA Support**: The library supports CUDA for GPU acceleration, which can be enabled as needed for performance boosts on compatible hardware.

### Community and Support

As an open-source project, TorchFold benefits from community contributions. Users are encouraged to report issues, suggest improvements, or contribute code via the repository. For further reading and background, a blog post is available at [near.ai/articles/2017-09-06-PyTorch-Dynamic-Batching/](http://near.ai/articles/2017-09-06-PyTorch-Dynamic-Batching/), which provides deeper insights into the motivations and technical details behind TorchFold.

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

## Contributing

We welcome contributions from the community to help improve TorchFold. Whether it's bug fixes, new features, or documentation improvements, your input is valuable to us. Follow the steps below to get started with contributing.

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

Thank you for considering contributing to TorchFold. Your efforts help make this project better for everyone!

## License

This project is licensed under the Apache License, Version 2.0. You can view the full license text in the [LICENSE](./LICENSE) file in this repository.
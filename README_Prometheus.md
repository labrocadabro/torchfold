# TorchFold: Dynamic Batching Library for Efficient PyTorch Computational Graphs

## Project Overview

TorchFold is a dynamic batching library for PyTorch that simplifies and optimizes computational graph execution for complex nested data structures like trees. It provides a powerful abstraction that allows developers to write dynamic computational graphs with automatic, efficient batching.

### Key Features

- **Dynamic Batching**: Automatically optimizes computational graph execution by batching similar operations
- **Simple Interface**: Replaces direct neural network module calls with a straightforward `add()` method
- **Flexible Computation**: Supports complex nested computational graphs, particularly useful for tree-structured data
- **PyTorch Integration**: Seamlessly works with PyTorch neural network modules

### Main Purpose

The primary goal of TorchFold is to solve performance bottlenecks in deep learning models that work with irregular or hierarchical data structures. By providing dynamic batching capabilities, it allows researchers and developers to:

- Improve computational efficiency for tree-based neural network architectures
- Reduce manual overhead in managing batched computations
- Enable more natural implementation of recursive or graph-based neural network models

### Benefits

- Enhanced performance through automatic batching
- More readable and maintainable code
- Reduced complexity in handling different sized inputs
- Supports both CUDA and CPU computations
- Compatible with standard PyTorch neural network modules

## Getting Started, Installation, and Setup

### Prerequisites

- Python 3.6+
- PyTorch (recommended latest stable version)
- pip package manager

### Installation

You can install TorchFold directly using pip:

```bash
pip install torchfold
```

### Quick Start

TorchFold provides a simple interface for dynamic batching in PyTorch. Here's a basic example of how to use it:

```python
import torchfold
import torch.nn as nn

# Create a Fold object
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
    def leaf(self, leaf):
        # Process leaf nodes
        pass

    def child(self, prev, child):
        # Process child nodes
        pass

# Usage
model = Model(...)
res = dfs(my_tree)
f.apply(model, [[res]])
```

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/nearai/torchfold.git
cd torchfold
```

2. Create a virtual environment (optional but recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install development dependencies:
```bash
pip install -e .
```

### Running Tests

To run the project tests:
```bash
# Assuming pytest is installed
pytest torchfold/torchfold_test.py
```

### Build and Distribution

To build a distribution package:
```bash
python setup.py sdist bdist_wheel
```

### Project Version

Current version: 0.1.0

## API Reference

### Classes

#### `Fold`
A class for batching operations in PyTorch computational graphs.

##### Constructor
```python
Fold(volatile=False, cuda=False)
```
- `volatile` (bool, optional): Controls variable volatility. Defaults to `False`.
- `cuda` (bool, optional): Enables CUDA acceleration. Defaults to `False`.

##### Methods
- `cuda()`: Enable CUDA acceleration for the Fold instance.
  - Returns: `self`

- `add(op, *args)`: Add an operation to the fold.
  - `op` (str): Name of the operation/method to add
  - `*args`: Arguments for the operation (can be Nodes, integers, tensors)
  - Returns: A `Fold.Node` representing the added operation
  - Raises `ValueError` if arguments are of invalid types

- `apply(nn, nodes)`: Apply the accumulated operations to a neural network module.
  - `nn`: Neural network module
  - `nodes`: Nodes to process
  - Returns: Processed node results

#### `Fold.Node`
A nested class representing a computational node within the Fold.

##### Methods
- `split(num)`: Split a node into multiple nodes if an operation returns multiple values.
  - `num` (int): Number of splits
  - Returns: Tuple of split nodes

- `nobatch()`: Disable batching for this node.
  - Returns: The node with batching disabled

#### `Unfold`
A debugging replacement for `Fold` that performs computations immediately.

##### Constructor
```python
Unfold(nn, volatile=False, cuda=False)
```
- `nn`: Neural network module
- `volatile` (bool, optional): Controls variable volatility. Defaults to `False`.
- `cuda` (bool, optional): Enables CUDA acceleration. Defaults to `False`.

##### Methods
- `cuda()`: Enable CUDA acceleration for the Unfold instance.
  - Returns: `self`

- `add(op, *args)`: Add and immediately compute an operation.
  - `op` (str): Name of the operation/method to add
  - `*args`: Arguments for the operation
  - Returns: An `Unfold.Node` with the computed result

- `apply(nn, nodes)`: Process nodes with the neural network.
  - `nn`: Neural network module (must match the one used in constructor)
  - `nodes`: Nodes to process
  - Returns: Processed node results

### Example Usage

```python
# Basic Fold usage
fold = Fold()
fold.cuda()  # Enable CUDA

# Add operations
node1 = fold.add('embedding', input_tensor)
node2 = fold.add('linear', node1)

# Apply to a neural network
result = fold.apply(neural_net, [node1, node2])
```

This library provides a flexible mechanism for batching operations in PyTorch, optimizing computational graph construction and execution.

## Project Structure

The project is organized into the following key directories and files:

### Main Package
- `torchfold/`: Core package directory
  - `__init__.py`: Exports main classes `Fold` and `Unfold`
  - `torchfold.py`: Contains the primary implementation of dynamic batching functionality
  - `torchfold_test.py`: Unit tests for the package

### Supporting Files
- `setup.py`: Package configuration and installation script
- `LICENSE`: Apache License, Version 2.0
- `.gitignore`: Specifies intentionally untracked files to ignore
- `logo.jpg`: Project logo image

### Examples
- `examples/snli/`: Contains example implementations
  - `spinn-example.py`: Demonstration of using the library with SNLI dataset

### Key Configuration
The project uses a simple, flat structure with the main implementation in the `torchfold` directory and minimal supporting files at the root level. The package is designed for dynamic batching with PyTorch, with a focus on clean, straightforward organization.

## Technologies Used

### Programming Languages
- Python

### Frameworks and Libraries
- PyTorch - Primary deep learning framework used for computational graph manipulation
- torch.autograd - Used for automatic differentiation

### Core Technologies
- Dynamic Batching - A key technique implemented by this library for efficient neural network computation
- Computational Graph Manipulation - The core functionality of the torchfold library

### Development Tools
- setuptools - Used for package setup and distribution
- PyTest (implied by presence of test files) - Likely used for unit testing

### Compatibility
- CUDA Support - Built-in support for GPU acceleration
- Cross-platform - Compatible with standard PyTorch environments

### Version
- Current version: 0.1.0 (as specified in setup.py)

## Additional Notes

### Performance Considerations
TorchFold is designed to optimize dynamic computational graphs by enabling dynamic batching. This can significantly improve computational efficiency, especially for tree-based or recursive neural network architectures.

### Memory Management
- The library supports both CPU and CUDA tensors via the `.cuda()` method
- Be mindful of memory usage when working with large computational graphs

### Debugging and Development
- An `Unfold` class is provided for easier debugging, which performs computations immediately instead of batching
- Use `torch.set_printoptions()` to control tensor display when investigating computational results

### Compatibility
- Compatible with PyTorch dynamic computation graphs
- Works best with custom neural network modules that implement specific operation methods

### Known Limitations
- Requires careful design of neural network modules to work effectively
- Complex computational graphs might require careful node sequencing
- Not suitable for completely static computational graphs where traditional batching is more efficient

### Version and Citation
If you use TorchFold in academic research, please cite the project using the DOI: 10.5281/zenodo.1299387

### Related Resources
- Original blog post explaining the dynamic batching concept: http://near.ai/articles/2017-09-06-PyTorch-Dynamic-Batching/
- Inspired by [TensorFlow Fold](https://github.com/tensorflow/fold)

## Contributing

We welcome contributions to TorchFold! By contributing, you can help improve this dynamic batching library for PyTorch.

### How to Contribute

1. **Fork the Repository**: Create a fork of the project on GitHub.

2. **Clone Your Fork**:
   ```
   git clone https://github.com/your-username/torchfold.git
   cd torchfold
   ```

3. **Create a Branch**:
   ```
   git checkout -b feature/your-feature-name
   ```

### Contribution Guidelines

#### Code Contributions
- Ensure your code follows Python best practices
- Write clear, concise, and descriptive commit messages
- Include tests for new functionality
- Update documentation if you modify existing features

#### Testing
- Run existing tests before submitting a pull request
- Add tests for any new functionality
- Ensure all tests pass before submitting your contribution

#### Pull Request Process
1. Update the README.md or documentation if your changes require it
2. Ensure your code passes all existing tests
3. Submit a pull request with a clear description of your changes

### Code of Conduct
- Be respectful and considerate of other contributors
- Collaborate constructively
- Help maintain a positive and inclusive community

### Reporting Issues
- Use the GitHub Issues section to report bugs
- Provide a clear description of the issue
- Include steps to reproduce the problem
- If possible, include a minimal code example demonstrating the issue

### Questions?
If you have any questions about contributing, please open an issue or contact the maintainers directly.

**Note**: By contributing, you agree that your contributions will be licensed under the project's Apache License, Version 2.0.

## License

This project is licensed under the Apache License, Version 2.0. 

#### Full License Details
The complete license text is available in the [LICENSE](LICENSE) file. Key points include:

- You are free to use, modify, and distribute this software
- You must give appropriate credit to the original authors
- Modifications must be clearly marked
- There is no warranty provided with this software

#### License Link
For the full license text, please see: [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

#### Copyright
Copyright 2018, NEAR Inc
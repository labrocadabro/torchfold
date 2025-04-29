# TorchFold: Dynamic Batching for Efficient PyTorch Computational Graphs

## Project Overview

TorchFold is a powerful PyTorch utility that provides dynamic batching capabilities for deep learning computation graphs, specifically designed to optimize performance and flexibility in neural network architectures.

### Core Purpose
TorchFold enables efficient batching of computational graph operations, allowing developers to dynamically batch tensors across different computational steps and network modules. This addresses a common challenge in deep learning: managing computational efficiency when working with variable-length or dynamically structured inputs.

### Key Features
- **Dynamic Batching**: Intelligently combines tensor computations across different computational steps
- **Flexible Node Management**: Supports creating computational nodes with complex dependencies
- **CUDA Support**: Seamless integration with GPU-accelerated computations
- **Debugging Mode**: Includes an `Unfold` class for step-by-step computation verification

### Benefits
- Improved computational efficiency by reducing redundant computations
- Enhanced flexibility in designing complex neural network architectures
- Simplified management of batched and non-batched computational graph components
- Reduced memory overhead in large-scale deep learning models

## Getting Started, Installation, and Setup

### Prerequisites

- Python 3.6+
- PyTorch (latest stable version recommended)

### Installation

You can install TorchFold using pip:

```bash
pip install torchfold
```

### Quick Start

TorchFold provides a simple interface for dynamic batching in PyTorch. Here's a basic example:

```python
import torchfold
import torch.nn as nn

# Create a Fold instance
f = torchfold.Fold()

# Define a recursive function to build computation
def dfs(node):
    if is_leaf(node):
        return f.add('leaf', node)
    else:
        prev = f.add('init')
        for child in children(node):
            prev = f.add('child', prev, child)
        return prev

# Define a PyTorch model with Fold-compatible methods
class Model(nn.Module):
    def leaf(self, leaf):
        # Process leaf node
        pass

    def child(self, prev, child):
        # Process child nodes
        pass

    def init(self):
        # Initialize computation
        pass

# Apply the Fold to your computation
res = dfs(my_tree)
model = Model(...)
results = f.apply(model, [[res]])
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
python -m unittest torchfold/torchfold_test.py
```

### Build and Distribution

The package is configured for distribution via PyPI. To build a distribution package:
```bash
python setup.py sdist bdist_wheel
```

### Compatibility

- Supports Python 3.6+
- Compatible with PyTorch (check `setup.py` for specific version recommendations)

## API Reference

### Main Classes

#### `Fold`
A class for batching computation in PyTorch with dynamic computation graphs.

##### Constructor
```python
Fold(volatile=False, cuda=False)
```
- `volatile`: If True, creates computations with gradient tracking disabled
- `cuda`: If True, moves computations to CUDA

##### Methods
- `.cuda()`: Enable CUDA computation for the fold
  ```python
  fold.cuda()
  ```

- `.add(op, *args)`: Add an operation to the fold
  ```python
  fold.add(operation_name, *arguments)
  ```
  - `op`: Name of the operation/method to apply
  - `*args`: Arguments for the operation (can be Nodes, tensors, or integers)
  - Returns a `Node` representing the computation

- `.apply(nn, nodes)`: Apply the accumulated computations to a neural network
  ```python
  fold.apply(neural_network, nodes_to_retrieve)
  ```
  - `nn`: Neural network module to apply computations
  - `nodes`: Nodes to retrieve from the computation graph

##### Inner Classes
- `Fold.Node`: Represents a node in the dynamic computation graph
  - Methods:
    - `.split(num)`: Split node if operation returns multiple values
    - `.nobatch()`: Disable batching for this node

- `Fold.ComputedResult`: Manages batched computation results

### Utility Classes

#### `Unfold`
A debugging replacement for `Fold` that performs computations immediately.

##### Constructor
```python
Unfold(nn, volatile=False, cuda=False)
```
- `nn`: Neural network module
- `volatile`: If True, creates computations with gradient tracking disabled
- `cuda`: If True, moves computations to CUDA

##### Methods
- `.cuda()`: Enable CUDA computation
- `.add(op, *args)`: Add and immediately execute an operation
- `.apply(nn, nodes)`: Retrieve computed nodes

### Usage Example
```python
# Create a Fold instance
fold = Fold(cuda=True)

# Add operations
node1 = fold.add('linear', input_tensor)
node2 = fold.add('relu', node1)

# Apply computations
result = fold.apply(neural_network, [node2])
```

## Project Structure

The project is organized into the following key directories and files:

### Main Package
- `torchfold/`: Core package directory
  - `__init__.py`: Exports main classes `Fold` and `Unfold`
  - `torchfold.py`: Contains the primary implementation of dynamic batching functionality
  - `torchfold_test.py`: Contains unit tests for the package

### Project Metadata and Configuration
- `setup.py`: Configures package metadata, dependencies, and installation settings
- `LICENSE`: Apache License, Version 2.0
- `.gitignore`: Specifies files and directories to be ignored by version control

### Additional Resources
- `README.md`: Project documentation and overview
- `logo.jpg`: Project logo or branding image

### Examples
- `examples/snli/`: Contains example implementation 
  - `spinn-example.py`: Demonstrates usage in a specific context (likely SNLI dataset)

The project follows a standard Python package structure, with the main implementation in the `torchfold` directory and supporting files in the root and `examples` directories.

## Technologies Used

### PyTorch Ecosystem
- **PyTorch**: Primary deep learning framework used for dynamic computation graphs
- **torch.autograd**: Used for automatic differentiation
- **torch.tensor**: Core tensor manipulation library

### Python Libraries
- **collections**: Used for data structure management (defaultdict)

### Development Tools
- **setuptools**: Used for package configuration and distribution
- **Python**: Primary programming language

### Key Project Characteristics
- Dynamic batching library for neural network computations
- Supports CUDA acceleration for GPU computations
- Supports both batched and non-batched tensor operations

## Additional Notes

### Performance Considerations

TorchFold is designed to optimize dynamic computation graphs by implementing dynamic batching. This can significantly improve computational efficiency, especially for tree-structured or recursive neural network architectures.

### Debugging and Development

For debugging purposes, the library includes an `Unfold` class that provides an alternative implementation for immediate computation. This can be useful when troubleshooting complex computational graphs.

### Compatibility

- Supports both CPU and CUDA tensor computations
- Compatible with PyTorch dynamic computation graph
- Works with various neural network architectures that require dynamic batching

### Limitations

- Requires careful implementation of operation methods in the neural network module
- Performance gains depend on the specific structure of the computational graph
- Not suitable for computational graphs with highly irregular or unpredictable structures

### Citation

If you use TorchFold in your research, please cite the project using the following BibTeX entry:

```bibtex
@misc{illia_polosukhin_2018_1299387,
  author       = {Illia Polosukhin and Maksym Zavershynskyi},
  title        = {nearai/torchfold: v0.1.0},
  month        = jun,
  year         = 2018,
  doi          = {10.5281/zenodo.1299387},
  url          = {https://doi.org/10.5281/zenodo.1299387}
}
```

### Additional Resources

- Original blog post: http://near.ai/articles/2017-09-06-PyTorch-Dynamic-Batching/
- Inspiration: [TensorFlow Fold](https://github.com/tensorflow/fold)

## Contributing

We welcome contributions to TorchFold! By contributing, you can help improve this dynamic batching library for PyTorch.

### How to Contribute

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes
4. Write or update tests to cover your changes
5. Ensure all tests pass
6. Submit a pull request

### Contribution Guidelines

#### Code Style
- Follow standard Python coding conventions
- Use clear, descriptive variable and function names
- Include docstrings and comments for complex logic

#### Testing
- All contributions must include unit tests
- Use the existing test framework in `torchfold/torchfold_test.py`
- Ensure 100% test coverage for new code
- Run existing tests before submitting a pull request

#### Reporting Issues
- Use GitHub Issues to report bugs or suggest improvements
- Provide a clear description of the issue
- Include minimal reproducible code examples when possible

### Pull Request Process
- Ensure your code passes all existing tests
- Update the README or documentation if your changes require it
- Your pull request will be reviewed by the maintainers
- Be prepared to make revisions based on feedback

### Citing the Project
If you use TorchFold in your research, please cite the project using the DOI and reference provided in the README.

## License

This project is licensed under the Apache License, Version 2.0. 

### Full License Details

The full text of the license can be found in the [LICENSE](LICENSE) file in the project root directory. 

#### Key Highlights

- Perpetual, worldwide, non-exclusive license
- Allows reproduction, modification, and distribution
- Requires preservation of copyright and license notices
- Provides a patent license
- Comes with no warranty or liability

#### License Link

For the complete license text, please visit: [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

#### Copyright

Copyright 2018, NEAR Inc
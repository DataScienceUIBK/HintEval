# Contribution Guide for HintEval

This guide provides instructions for setting up your environment, following best practices, and contributing to the HintEval project.

## Setting Up the Development Environment

1. **Fork the Repository**  
   Fork the [HintEval repository](https://github.com/DataScienceUIBK/HintEval) on GitHub.

2. **Clone Your Fork**  
   Clone your forked repository and navigate to the project directory:
   ```bash
   git clone https://github.com/YOUR_USERNAME/HintEval.git
   cd HintEval
   ```

3. **Create a Virtual Environment**  
   Set up a virtual environment using Conda:
   ```bash
   conda create -n hinteval_env python=3.11.9 --no-default-packages
   conda activate hinteval_env
   ```

4. **Install Dependencies**  
   Install PyTorch 2.4.0 following platform-specific instructions from the [PyTorch installation page](https://pytorch.org/get-started/previous-versions/). If you have a GPU, install the CUDA version for optimal performance.

   After installing PyTorch, install HintEval in editable mode:
   ```bash
   pip install -e .
   ```

## Development Workflow

1. **Create a New Branch**  
   Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes and Commit**  
   Stage and commit your changes:
   ```bash
   git add .
   git commit -m "Your descriptive commit message"
   ```

3. **Push Changes to Your Fork**  
   Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Open a Pull Request**  
   In the original HintEval repository, open a new pull request for your feature branch.

## Coding Standards

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines.
- Include type hints wherever applicable.
- Document functions, classes, and modules with clear docstrings.
- Ensure all tests pass before submitting a pull request.

## Documentation

- Update documentation for new features or changes.
- Use the [NumPy style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy) for docstrings.

## Submitting Pull Requests

1. Ensure code meets coding standards.
2. Include tests for new functionality.
3. Update relevant documentation.
4. Provide a concise description of the changes in the pull request.

Thank you for contributing to HintEval!

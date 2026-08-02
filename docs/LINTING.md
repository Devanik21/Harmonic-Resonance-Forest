# Linting and Code Styling Guidelines

To ensure a uniform codebase and maintain the scientific integrity of the Harmonic Resonance Forest (HRF) project, we adhere to strict Python code formatting and linting standards. All contributors (including GSSoC contributors) must ensure their code meets these standards before submitting a Pull Request.

We use the following tools to enforce our code style:
- **Black**: The uncompromising Python code formatter.
- **Flake8**: A tool to check the style and quality of some Python code.
- **isort**: A Python utility / library to sort imports.

## Prerequisites

Ensure you have the linting tools installed. You can install them using pip:

```bash
pip install black flake8 isort
```

## 1. Code Formatting with Black

We use **Black** to format our Python code. Black reformats entire files in place and ensures consistency across the codebase.

To format your code with Black, run:

```bash
black .
```

This will format all Python files in the current directory and its subdirectories. We use the default Black settings.

## 2. Linting with Flake8

We use **Flake8** to check for style guide enforcement (PEP 8) and common programming errors.

To run Flake8, execute:

```bash
flake8 .
```

Please resolve any errors or warnings reported by Flake8 before committing your code.

## 3. Sorting Imports with isort

We use **isort** to automatically sort and group imports in Python files.

To sort your imports, run:

```bash
isort .
```

This will reorder the imports to comply with PEP 8 and our project standards.

## Pre-commit Checklist

Before you create a pull request, please ensure you have run all the styling tools:
1. Run `isort .` to organize imports.
2. Run `black .` to format the code.
3. Run `flake8 .` to check for any remaining linting issues.

Thank you for contributing to Harmonic Resonance Forest and helping us maintain a clean, readable, and uniform codebase!

# ONNX-Metadata-Extractor

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)
![ONNX Version](https://img.shields.io/badge/ONNX-1.8.0%2B-blue.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Script Explanation](#script-explanation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

**ONNX-Metadata-Extractor** is a Python tool designed to extract comprehensive metadata and model properties from ONNX (Open Neural Network Exchange) models. It parses ONNX models to retrieve detailed information about the model's structure, inputs, outputs, custom metadata, and more, organizing the data into a structured JSON format for easy analysis and utilization.

## Features

- **Model Information**: Extracts model name, version, producer details, description, and domain.
- **Input Specifications**: Retrieves input names, data types, shapes, dimensions, and documentation.
- **Output Specifications**: Retrieves output names, data types, shapes, dimensions, and documentation.
- **Custom Metadata**: Extracts user-defined metadata fields and custom attributes.
- **Additional Attributes**: Gathers opset versions, IR version, license information, and training framework details.
- **Graph Structure**: Details nodes (operations), initializers (weights), and the overall computational graph structure.
- **Structured Output**: Organizes all extracted information into a clear and structured JSON file.

## Installation

### Prerequisites

- **Python 3.7 or higher**

### Install Required Packages

Use `pip` to install the necessary Python packages:

```bash
pip install onnx
```
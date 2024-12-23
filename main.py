import onnx
import json
from typing import Any, Dict

def get_tensor_type(tensor_type) -> str:
    """
    Convert ONNX tensor type to a readable string.
    """
    type_mapping = {
        1: "FLOAT",
        2: "UINT8",
        3: "INT8",
        4: "UINT16",
        5: "INT16",
        6: "INT32",
        7: "INT64",
        8: "STRING",
        9: "BOOL",
        10: "FLOAT16",
        11: "DOUBLE",
        12: "UINT32",
        13: "UINT64",
        14: "COMPLEX64",
        15: "COMPLEX128",
        16: "BFLOAT16"
    }
    return type_mapping.get(tensor_type, "UNKNOWN")

def extract_model_info(model: onnx.ModelProto) -> Dict[str, Any]:
    """
    Extracts general model information.
    """
    model_info = {
        "Model Information": {
            "Model Name": model.graph.name,
            "Version": model.ir_version,
            "Producer Name": model.producer_name,
            "Producer Version": model.producer_version,
            "Description": model.doc_string,
            "Domain": model.domain
        }
    }
    return model_info

def extract_io_specs(graph_io, io_type: str) -> Dict[str, Any]:
    """
    Extracts input or output specifications.
    """
    specs = {}
    for io in graph_io:
        name = io.name
        # Determine the type of the tensor
        tensor_type = io.type.tensor_type
        data_type = get_tensor_type(tensor_type.elem_type)
        # Extract shape information
        shape = []
        for dim in tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append("?")  # Unknown dimension
        # Extract documentation string if available
        doc = io.doc_string if io.doc_string else ""
        specs[name] = {
            "Data Type": data_type,
            "Shape": shape,
            "Documentation": doc
        }
    return {f"{io_type} Specifications": specs}

def extract_custom_metadata(model: onnx.ModelProto) -> Dict[str, Any]:
    """
    Extracts custom metadata from the model.
    """
    custom_meta = {}
    for prop in model.metadata_props:
        custom_meta[prop.key] = prop.value
    return {"Custom Metadata": custom_meta}

def extract_additional_attributes(model: onnx.ModelProto) -> Dict[str, Any]:
    """
    Extracts additional attributes like opset versions, IR version, license, and training framework.
    """
    additional = {
        "Opset Versions": {opset.domain if opset.domain else "ai.onnx": opset.version for opset in model.opset_import},
        "IR Version": model.ir_version,
        "Producer Name": model.producer_name,
        "Producer Version": model.producer_version
    }

    # Attempt to extract license and training framework from custom metadata
    license_info = ""
    training_framework = ""
    for prop in model.metadata_props:
        if prop.key.lower() == "license":
            license_info = prop.value
        elif "framework" in prop.key.lower():
            training_framework = prop.value
    additional["License Information"] = license_info
    additional["Training Framework"] = training_framework

    return {"Additional Attributes": additional}

def extract_graph_structure(model: onnx.ModelProto) -> Dict[str, Any]:
    """
    Extracts the graph structure including nodes, initializers, inputs, and outputs.
    """
    graph = model.graph
    nodes = []
    for node in graph.node:
        node_info = {
            "Name": node.name,
            "Op Type": node.op_type,
            "Inputs": list(node.input),
            "Outputs": list(node.output),
            "Attributes": {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        }
        nodes.append(node_info)

    initializers = {}
    for initializer in graph.initializer:
        initializers[initializer.name] = {
            "Data Type": get_tensor_type(initializer.data_type),
            "Shape": list(initializer.dims),
            "Values": initializer.float_data if initializer.data_type == 1 else list(initializer.int64_data)
        }

    graph_structure = {
        "Graph Structure": {
            "Nodes": nodes,
            "Initializers": initializers,
            "Inputs": [io.name for io in graph.input],
            "Outputs": [io.name for io in graph.output]
        }
    }

    return graph_structure

def extract_onnx_metadata(model_path: str) -> Dict[str, Any]:
    """
    Main function to extract all required metadata from an ONNX model.
    """
    # Load the ONNX model
    model = onnx.load(model_path)

    # Initialize the metadata dictionary
    metadata = {}

    # Extract different parts of the metadata
    metadata.update(extract_model_info(model))
    metadata.update(extract_io_specs(model.graph.input, "Input"))
    metadata.update(extract_io_specs(model.graph.output, "Output"))
    metadata.update(extract_custom_metadata(model))
    metadata.update(extract_additional_attributes(model))
    metadata.update(extract_graph_structure(model))

    return metadata

if __name__ == "__main__":
    import argparse

    # Set up argument parsing to accept the model path and output file
    parser = argparse.ArgumentParser(description="Extract metadata from an ONNX model.")
    parser.add_argument("model_path", type=str, help="Path to the ONNX model file.")
    parser.add_argument("-o", "--output", type=str, default="model_metadata.json", help="Output JSON file name.")
    args = parser.parse_args()

    # Extract metadata
    metadata = extract_onnx_metadata(args.model_path)

    # Write the metadata to a JSON file
    with open(args.output, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata extracted and saved to {args.output}")

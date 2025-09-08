#!/usr/bin/env python3
"""
Script to extract evaluation metrics from trained_models.pkl file.
"""

import pickle
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np

def load_pickled_models(file_path):
    """Load the pickled models file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded pickle file: {file_path}")
        return data
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None

def extract_metrics(data):
    """Extract all available metrics from the loaded data."""
    metrics = {}
    
    print("\nExploring data structure:")
    print(f"Data type: {type(data)}")
    print(f"Data class: {data.__class__.__name__}")
    print(f"Data module: {data.__class__.__module__}")
    
    # Get all attributes of the object
    print("\nAvailable attributes:")
    attributes = []
    for attr_name in dir(data):
        if not attr_name.startswith('_'):  # Skip private attributes
            try:
                attr_value = getattr(data, attr_name)
                if not callable(attr_value):  # Skip methods
                    attributes.append((attr_name, attr_value))
                    print(f"  - {attr_name}: {type(attr_value)}")
                else:
                    print(f"  - {attr_name}: {type(attr_value)} (method)")
            except Exception as e:
                print(f"  - {attr_name}: Error accessing - {e}")
    
    # Extract metrics from attributes
    for attr_name, attr_value in attributes:
        try:
            if isinstance(attr_value, dict):
                metrics[attr_name] = attr_value
                print(f"\nFound dictionary attribute '{attr_name}' with keys: {list(attr_value.keys())}")
            elif isinstance(attr_value, (list, tuple)) and len(attr_value) > 0:
                # Check if it's a list of metrics/results
                if isinstance(attr_value[0], dict):
                    metrics[attr_name] = attr_value
                    print(f"\nFound list attribute '{attr_name}' with {len(attr_value)} dictionary items")
                else:
                    metrics[attr_name] = attr_value
                    print(f"\nFound list attribute '{attr_name}' with {len(attr_value)} items of type {type(attr_value[0])}")
            elif isinstance(attr_value, (int, float, str, bool)):
                metrics[attr_name] = attr_value
                print(f"Found simple attribute '{attr_name}': {attr_value}")
            else:
                # For complex objects, try to extract their attributes too
                if hasattr(attr_value, '__dict__'):
                    try:
                        obj_dict = attr_value.__dict__
                        metrics[f'{attr_name}_attributes'] = obj_dict
                        print(f"Found object attribute '{attr_name}' with attributes: {list(obj_dict.keys())}")
                    except:
                        metrics[f'{attr_name}_summary'] = str(attr_value)[:500]
                        print(f"Found complex attribute '{attr_name}': {type(attr_value)}")
                else:
                    metrics[f'{attr_name}_summary'] = str(attr_value)[:500]
                    print(f"Found attribute '{attr_name}': {type(attr_value)}")
        except Exception as e:
            print(f"Error processing attribute '{attr_name}': {e}")
    
    # Try specific method calls that might return metrics
    method_names = ['get_metrics', 'get_evaluation', 'get_results', 'get_scores', 'evaluate']
    for method_name in method_names:
        if hasattr(data, method_name):
            try:
                method = getattr(data, method_name)
                if callable(method):
                    result = method()
                    metrics[f'method_{method_name}'] = result
                    print(f"Successfully called method '{method_name}': {type(result)}")
            except Exception as e:
                print(f"Error calling method '{method_name}': {e}")
    
    return metrics

def save_metrics_to_json(metrics, output_path):
    """Save metrics to JSON file."""
    try:
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        converted_metrics = convert_numpy_types(metrics)
        
        with open(output_path, 'w') as f:
            json.dump(converted_metrics, f, indent=2, default=str)
        
        print(f"\nMetrics saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving metrics to JSON: {e}")
        return False

def print_metrics_summary(metrics):
    """Print a structured summary of the metrics."""
    print("\n" + "="*60)
    print("EXTRACTED METRICS SUMMARY")
    print("="*60)
    
    for key, value in metrics.items():
        print(f"\n{key.upper()}:")
        print("-" * len(key))
        
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        elif isinstance(value, (list, tuple)):
            if len(value) <= 10:
                for i, item in enumerate(value):
                    print(f"  [{i}]: {item}")
            else:
                print(f"  List with {len(value)} items")
                print(f"  First 5: {value[:5]}")
                print(f"  Last 5: {value[-5:]}")
        else:
            print(f"  {value}")

def main():
    # File paths
    pickle_file = "/Users/avivnahon/ds-agent-projects/session_0f4f0dd5-122d-4338-af13-0f967752758c/experiments/experiment_1/output/model_artifacts/trained_models.pkl"
    output_dir = "/Users/avivnahon/ds-agent-projects/session_0f4f0dd5-122d-4338-af13-0f967752758c/experiments/experiment_1/output/model_artifacts/"
    json_output = os.path.join(output_dir, "extracted_metrics.json")
    
    print("Loading trained models and extracting metrics...")
    print(f"Input file: {pickle_file}")
    print(f"Output file: {json_output}")
    
    # Load the pickled data
    data = load_pickled_models(pickle_file)
    if data is None:
        return
    
    # Extract metrics
    metrics = extract_metrics(data)
    
    # Print summary
    print_metrics_summary(metrics)
    
    # Save to JSON
    success = save_metrics_to_json(metrics, json_output)
    
    if success:
        print(f"\n✓ Successfully extracted and saved metrics to {json_output}")
    else:
        print(f"\n✗ Failed to save metrics to JSON file")

if __name__ == "__main__":
    main()
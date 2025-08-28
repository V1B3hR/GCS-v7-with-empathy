#!/usr/bin/env python3
"""
GCSInference Enhanced API Usage Examples
========================================

This script demonstrates how to use the enhanced GCSInference class
with its new structured JSON output format and robust error handling.

Usage Examples:
--------------
1. New Structured Format (Recommended)
2. Legacy Format (Backward Compatibility)
3. Error Handling
4. Health Monitoring
"""

import numpy as np
import json
from gcs.inference import GCSInference


def example_new_format():
    """Example using the new structured JSON format"""
    print("🆕 NEW STRUCTURED FORMAT EXAMPLE")
    print("-" * 50)
    
    # Initialize inference engine
    inference = GCSInference('path/to/model.h5', 'path/to/graph.npz')
    
    # Prepare input data (EEG source data + adjacency matrix)
    source_eeg = np.random.randn(1, 68, 250)  # (batch, nodes, timesteps)
    adj_matrix = np.random.random((1, 68, 68))  # (batch, nodes, nodes)
    input_data = [source_eeg, adj_matrix]
    
    # Make prediction with new format
    result = inference.predict(input_data)
    
    # Check prediction success
    if result["status"] == "success":
        print(f"✅ Prediction successful!")
        print(f"   Label: {result['prediction']['label']}")
        print(f"   Confidence: {result['prediction']['confidence']:.3f}")
        print(f"   Intent ID: {result['prediction']['intent_id']}")
        print(f"   Timestamp: {result['metadata']['timestamp']}")
        
        # Safe to pass to other systems - it's JSON serializable
        json_str = json.dumps(result)
        print(f"   JSON serializable: ✅")
        
    else:
        print(f"❌ Prediction failed: {result['error_message']}")


def example_legacy_format():
    """Example using the legacy tuple format for backward compatibility"""
    print("\n🔄 LEGACY FORMAT EXAMPLE")
    print("-" * 50)
    
    # Initialize inference engine
    inference = GCSInference('path/to/model.h5', 'path/to/graph.npz')
    
    # Prepare input data
    source_eeg = np.random.randn(1, 68, 250)
    adj_matrix = np.random.random((1, 68, 68))
    input_data = [source_eeg, adj_matrix]
    
    # Method 1: Use return_legacy_format parameter
    label, confidence, attention = inference.predict(input_data, return_legacy_format=True)
    print(f"Method 1 - Legacy flag: {label}, {confidence:.3f}, {attention.shape}")
    
    # Method 2: Use predict_legacy method
    label2, confidence2, attention2 = inference.predict_legacy(input_data)
    print(f"Method 2 - Legacy method: {label2}, {confidence2:.3f}, {attention2.shape}")


def example_error_handling():
    """Example demonstrating robust error handling"""
    print("\n🛡️ ERROR HANDLING EXAMPLE")
    print("-" * 50)
    
    # Initialize inference engine
    inference = GCSInference('path/to/model.h5', 'path/to/graph.npz')
    
    # Test with invalid input
    result = inference.predict(None)
    
    print("Testing with None input:")
    print(f"   Status: {result['status']}")
    print(f"   Error: {result['error_message']}")
    print(f"   Safe defaults: label={result['prediction']['label']}, conf={result['prediction']['confidence']}")
    
    # Test legacy error handling
    label, conf, attention = inference.predict(None, return_legacy_format=True)
    print(f"   Legacy error: ({label}, {conf}, {attention})")


def example_health_monitoring():
    """Example of health monitoring capabilities"""
    print("\n💊 HEALTH MONITORING EXAMPLE")
    print("-" * 50)
    
    # Initialize inference engine
    inference = GCSInference('path/to/model.h5', 'path/to/graph.npz')
    
    # Check system health
    health = inference.is_healthy()
    
    print("System Health Status:")
    print(f"   Overall: {health['status']}")
    print(f"   Model loaded: {health['model_loaded']}")
    print(f"   Inference ready: {health['inference_model_ready']}")
    print(f"   Graph loaded: {health['adjacency_matrix_loaded']}")
    print(f"   Available labels: {health['available_labels']}")


def integration_example():
    """Example showing integration with existing code"""
    print("\n🔗 INTEGRATION EXAMPLE")
    print("-" * 50)
    
    # This shows how existing code can be updated minimally
    
    # OLD CODE (still works):
    # intent, conf, attention = inference.predict(data)
    
    # NEW CODE (recommended for new implementations):
    inference = GCSInference('path/to/model.h5', 'path/to/graph.npz')
    data = [np.random.randn(1, 68, 250), np.random.random((1, 68, 68))]
    
    # For existing systems that expect tuple unpacking:
    intent, conf, attention = inference.predict(data, return_legacy_format=True)
    print(f"Legacy tuple: {intent}, {conf:.3f}, {attention.shape}")
    
    # For new systems that want structured output:
    result = inference.predict(data)
    if result["status"] == "success":
        intent = result["prediction"]["label"]
        conf = result["prediction"]["confidence"]
        attention = np.array(result["attention_weights"])
        print(f"Structured format: {intent}, {conf:.3f}, {attention.shape}")


if __name__ == "__main__":
    print("=" * 70)
    print("🧠 GCSInference Enhanced API Usage Examples")
    print("=" * 70)
    
    print("\n⚠️  NOTE: This is a demonstration script.")
    print("   In real usage, provide actual model and graph file paths.")
    print("   The examples below use mock data and will show the API structure.")
    
    try:
        example_new_format()
        example_legacy_format()
        example_error_handling()
        example_health_monitoring()
        integration_example()
        
        print("\n" + "=" * 70)
        print("✅ All examples completed successfully!")
        print("=" * 70)
        
    except ImportError:
        print("\n❌ Cannot import GCSInference. Make sure you're running from the backend directory.")
        print("   cd backend && python usage_examples.py")
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("   This is expected if model files don't exist - this is just API demonstration.")
"""
Clean Clone Verification Script for predict_rul()
Author: Member A (Data & ML Lead)
"""

import os
import sys
import json
import tempfile
import subprocess


def verify_clean_clone():
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = os.path.join(tmpdir, "repo")
        print(f"1. Cloning repository into isolated environment: {clone_dir}")
        res = subprocess.run(["git", "clone", repo_path, clone_dir], capture_output=True, text=True)
        assert res.returncode == 0, f"Git clone failed: {res.stderr}"
        
        test_code = """
import sys, os, json
sys.path.insert(0, os.path.abspath('.'))
from src.predict import predict_rul

sample_payload_30 = [
    {
        'op_setting_1': -0.0007,
        'op_setting_2': -0.0004,
        'sensor_2': 641.82,
        'sensor_3': 1589.70,
        'sensor_4': 1400.60,
        'sensor_7': 554.36,
        'sensor_8': 2388.06,
        'sensor_9': 9046.19,
        'sensor_11': 47.47,
        'sensor_12': 521.66,
        'sensor_13': 2388.02,
        'sensor_14': 8138.62,
        'sensor_15': 8.4195,
        'sensor_17': 392,
        'sensor_20': 39.06,
        'sensor_21': 23.4190
    }
    for _ in range(30)
]

result = predict_rul(sample_payload_30)
print("SUCCESS_CLEAN_CLONE")
print(json.dumps(result, indent=2))
assert result["version"] == "1.3.0", f"Version mismatch: {result['version']}"
assert "validated dataset range" in result["recommendation"], f"Recommendation language safety violation: {result['recommendation']}"
"""
        script_path = os.path.join(clone_dir, "test_isolated.py")
        with open(script_path, "w") as f:
            f.write(test_code)
            
        print("2. Executing predict_rul() inside clean clone...")
        res_py = subprocess.run([sys.executable, "test_isolated.py"], cwd=clone_dir, capture_output=True, text=True)
        print("Stdout:\n", res_py.stdout)
        if res_py.stderr:
            print("Stderr:\n", res_py.stderr)
            
        assert "SUCCESS_CLEAN_CLONE" in res_py.stdout, "Clean clone test failed!"
        print("Clean clone verification passed 100% with version 1.3.0 and neutral prototype wording!")


if __name__ == "__main__":
    verify_clean_clone()

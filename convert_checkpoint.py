"""
One-time script to convert old Keras 2 checkpoint to Keras 3 format.

Run this ONCE after setting up the venv:
    python convert_checkpoint.py

It reads the old checkpoint with tf_keras compatibility mode
and re-saves it in Keras 3 format.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import h5py

OLD_CHECKPOINT = "EarthRemoteSensingRapidResponse/tmp/checkpoint.weights.h5"
NEW_CHECKPOINT = "EarthRemoteSensingRapidResponse/tmp/checkpoint_k3.weights.h5"

def inspect_and_convert():
    print(f"Reading: {OLD_CHECKPOINT}")
    
    # Read weight names and arrays from the H5 file
    weights = {}
    with h5py.File(OLD_CHECKPOINT, "r") as f:
        # Keras 2 saves weights under model_weights/<layer_name>/<layer_name>/kernel, bias etc.
        def visitor(name, obj):
            if not isinstance(obj, h5py.Group):
                weights[name] = np.array(obj)
        f.visititems(visitor)
    
    print(f"Found {len(weights)} weight tensors")
    for name in sorted(weights.keys())[:10]:
        print(f"  {name}: shape={weights[name].shape}")
    if len(weights) > 10:
        print(f"  ... and {len(weights) - 10} more")
    
    # Now load with Keras 3 (no legacy mode) and build the model
    os.environ.pop("TF_USE_LEGACY_KERAS", None)
    
    import keras
    print(f"\nKeras version: {keras.__version__}")
    
    # We need to import model definition - add the project to path
    import sys
    sys.path.insert(0, "EarthRemoteSensingRapidResponse")
    from ERSRR_Model import create_vit_encoder_decoder
    
    model = create_vit_encoder_decoder()
    
    # Try direct load first (might work if format is compatible)
    try:
        model.load_weights(OLD_CHECKPOINT)
        print("\nDirect load succeeded! Re-saving in current Keras format...")
        model.save_weights(NEW_CHECKPOINT)
        print(f"Saved to: {NEW_CHECKPOINT}")
        print("Update ERSRR_Model.py checkpoint_dir to use the new file.")
        return
    except Exception as e:
        print(f"\nDirect load failed: {e}")
        print("\nManual weight mapping required.")
        print("The old checkpoint was saved with a different Keras version.")
        print("You will need to retrain the model with the new environment.")
        print("\nTo train: change MODEL_SETTING to 'train' in ERSRR_Model.py, then run it.")

if __name__ == "__main__":
    inspect_and_convert()

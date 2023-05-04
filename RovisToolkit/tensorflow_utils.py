"""
Copyright (c) RovisLab
RovisDojo: RovisLab neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (s.grigorescu@unitbv.ro)
"""

import os
import tensorflow as tf

def restore_weights(model_signature, model, checkpoint_dir):
    model_signature_name = model_signature.split(".")[0]
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)

        # Check if model names are the same
        model_checkpoint_name = latest_checkpoint.split(".")[1].split("/")[2]

        if model_signature_name != model_checkpoint_name:
            print("WARNING: Model and checkpoint names do not match. Could not load weights.")
        else:
            print("Restoring weights from", latest_checkpoint)
            #return tf.keras.models.load_model(latest_checkpoint)
            model.load_weights(latest_checkpoint)
    else:
        print("WARNING: No checkpoint file available. Could not load checkpoint.")

    return model

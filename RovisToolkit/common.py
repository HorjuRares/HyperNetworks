"""
Copyright (c) RovisLab
RovisDojo: RovisLab neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (s.grigorescu@unitbv.ro)
"""

"""
 * common.py
 *
 *  Created on: 11.02.2021
 *      Author: Sorin Grigorescu
"""

def listit(t):
    return list(map(listit, t)) if isinstance(t, (list, tuple)) else t

def get_cached_layers(dnn_blocks):
    """
    :param dnn_blocks:  blocks of DNN layers
    :return:            list of IDs of the layers which have to be cached
    """
    ch = []
    # Parse DNN blocks
    for block in dnn_blocks:
        # Parse layers
        for layer in block:
            if isinstance(layer[0], list):
                for channel in layer[0]:
                    if channel >= 0:
                        ch.append(channel)
    return ch

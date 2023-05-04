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
import logging
import torch
from collections import OrderedDict
# from Filters_DNNs.DNN_ObjectDetection2D.models.YoloV5.common import *

# Initialize logger
logger = logging.getLogger(__name__)

def get_last_checkpoint(checkpoint_dir, extention='.pth'):
    if not os.path.exists(checkpoint_dir):
        print("WARNING: Checkpoint folder does not exist. Could not load checkpoint.")
        return None

    if os.path.isfile(checkpoint_dir) and checkpoint_dir.endswith(extention):
        return checkpoint_dir

    checkpoints = [os.path.join(checkpoint_dir, file) for file in os.listdir(checkpoint_dir) if file.endswith(extention)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)

        # Check if model names are the same
        if not latest_checkpoint.endswith(extention):
            print("WARNING: No checkpoint file available. Could not load checkpoint.")
            return None
        return latest_checkpoint
    else:
        print("WARNING: No checkpoint file available. Could not load checkpoint.")
        return None

def load_last_checkpoint(ckpts_folder_path, model) -> int:
    ckpt_path = get_last_checkpoint(ckpts_folder_path)
    start_epoch = load_weights(model, ckpt_path)
    return start_epoch

def _check_num_output_objects(model, weights):
    # no = na * (nc + 5)
    for k, v in model.named_parameters():
        pass
    no_model = v.shape[0]
    for k in weights:
        pass
    no_weights = weights[k].shape[0]
    return no_weights == no_model

def load_weights(model, pretrained_weights) -> int:
    if not os.path.exists(pretrained_weights):
        logging.info("Weights file '{}' does not exist".format(pretrained_weights))
        return
    else:
        logging.info("Loading '{}' weights from '{}'".format(model.__class__.__name__, pretrained_weights))
    
    ckpt = torch.load(pretrained_weights) #, map_location=device)  # load checkpoint
    start_epoch = 0

    if 'epoch' in ckpt:
        start_epoch = ckpt['epoch'] + 1

    if 'model' in ckpt:
        if not isinstance(ckpt['model'], OrderedDict):
            state_dict = ckpt['model'].state_dict()  # to FP32
        else:
            state_dict = ckpt['model']
    else:
        state_dict = ckpt

    # TODO copy / move intersect_dicts
    #state_dict = intersect_dicts(state_dict, model.state_dict())  # intersect

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)

    return start_epoch

def save_checkpoint(model, model_name, optimizer, lr_scheduler, epoch, loss, save_path):
    if isinstance(model, torch.nn.DataParallel):
        md = model.module.state_dict()
    else:
        md = model.state_dict()

    opt = optimizer.state_dict()
    lrs = lr_scheduler.state_dict()
    save_dict = {
        "epoch": epoch,
        "model": md,
        "optim": opt,
        "lr_scheduler": lrs
    }

    # Define pth and onnx save paths
    file_path_pth = os.path.join(save_path, model_name + '-epoch' +
                                 '-{epoch:06d}-loss-{loss:2f}'.format(epoch=epoch, loss=loss))
    file_path_pth += '.pth'

    torch.save(save_dict, file_path_pth)
    logger.info("'{}' model checkpoint saved as: {}".format(model_name, file_path_pth))

def save_submodel(model, min_layer, max_layer, file_path):
    """
    Saves the layers found between the min and max layers
    :param model:       pytorch model file
    :param min_layer:   min layer to save
    :param max_layer:   max layer to save
    :param file_path:   pth file for saving
    """
    state_dict_to_save = OrderedDict()
    state_dict_model = model.state_dict()
    for k in state_dict_model:
        layer_id = int(k.split(".")[1])
        if (layer_id >= min_layer) and (layer_id <= max_layer):
            name_head = k.split(".")
            name_head[1] = str(int(name_head[1]) - min_layer)
            name_head = '.'.join(name_head[1:len(name_head)])
            state_dict_to_save[name_head] = state_dict_model[k]

    torch.save(state_dict_to_save, file_path)
    #torch.save(model_params, file_path)

    logging.info("Sub-model of layers {} -- {} saved to '{}'".format(min_layer, max_layer, file_path))

def save_submodel(model, submodel_name, file_path):
    state_dict_to_save = OrderedDict()
    state_dict_model = model.state_dict()
    for k in state_dict_model:
        if submodel_name in k:
            new_name_layer = k.replace(submodel_name + '.', '')
            state_dict_to_save[new_name_layer] = state_dict_model[k]

    torch.save(state_dict_to_save, file_path)

    #model_params = {
    #    "model": deepcopy(model.module if is_parallel(model) else model).half()
    #}
    #torch.save(model_params, file_path)

    logging.info("Sub-model {} saved to '{}'".format(submodel_name, file_path))

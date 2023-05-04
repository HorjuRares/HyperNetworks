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
import shutil
import time
import csv
import cv2
import numpy as np
import torch
import onnx
import onnxruntime
from data.rovis_db_interface import RovisDatabaseParser
import experimental.realtime_panoptic_segmentation.realtime_panoptic.data.panoptic_transform as P
from experimental.realtime_panoptic_segmentation.realtime_panoptic.utils.visualization import visualize_segmentation_image, visualize_detection_image, draw_mask
from data.types_ROVIS_TYPES import RovisFilterType, RovisDataType



########################################################################################################################
# PARAMETERS TO MODIFY
########################################################################################################################
show_output = True  # Imshow the semseg and detections images

# !!!!!!!!
# if set to True, this will delete folders from database
# proceed with caution
overwrite_data = True

write_instance_image = True

# Onnx model must be without batch processing
#model_path = "C:/dev/TorchTest/model/GPU_320_240.onnx"
#rovis_db_path = "C:/data/scout_data/rovis_test_3"
model_path = "C:/dev/src/RovisLab/RovisVision/models/GPU_320_240.onnx"
rovis_db_path = "C:/data/RovisDatabases/Scout/08"

"""
Mapping Scout cameras - obj det - sem seg in scount.conf
1 - Back - 12 - 22
2 - Left - 13 - 19
3 - Front - 11 - 21
4 - Right - 14 - 20
"""


image_keys = [(1, 1), (1, 2), (1, 3), (1, 4)]
objdet_keys = [(1, 12), (1, 13), (1, 11), (1, 14)]
semseg_keys = [(1, 22), (1, 19), (1, 21), (1, 20)]

confidence_threshold = 0.2

tensor_size = (320, 240)  # Must be the same as the model input size

cityscapes_colormap = np.array([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]])

cityscapes_instance_label_name = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

# Add custom cuda provider
CUSTOM_PROVIDERS = [
    {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'cuda_mem_limit': 4 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True
    },
    {}
]


def write_rois(stream, frame_id, predictions, w, h, scale_factor_x, scale_factor_y):
    boxes = predictions[0].cpu().numpy()
    labels = predictions[2].cpu().numpy()

    if len(predictions[1]) > 0:
        scores = predictions[1].tolist()
    else:
        scores = [1.0] * len(boxes)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cls = labels[i]
        if scores[i] < confidence_threshold:
            continue
        if 0 <= x1 <= w and 0 <= y1 <= h:
            stream.write("{},{},{},{},{},{},{}\n".format(
                frame_id,
                i,
                cls,
                scale_factor_x * x1,
                scale_factor_y * y1,
                scale_factor_x * (x2-x1),
                scale_factor_y * (y2-y1)
            ))


def draw_instance_segmentation_image(original_image, predictions):
    visualized_image = np.zeros_like(original_image)
    boxes = predictions[0]
    if len(predictions[3]) > 0:
        masks = predictions[3]
    else:
        masks = [None] * len(boxes)
    if len(predictions[1]) > 0:
        scores = predictions[1].tolist()
    else:
        scores = [1.0] * len(boxes)

    color_idx = 1
    for mask, score in zip(masks, scores):
        if score < 0.5:
            continue
        color = (color_idx, color_idx, color_idx)
        color = tuple(map(int, color))
        color_idx += 1

        if mask is not None:
            thresh = (mask > 0.5).cpu().numpy().astype('uint8')
            visualized_image, color = draw_mask(visualized_image, thresh, 1, color)
    return visualized_image


def check_directories(semseg_filter_id, objdet_filter_id):
    if not os.path.exists(model_path):
        print("Model not found at path ", model_path)
        exit(-1)
    if not os.path.exists(rovis_db_path):
        print("Rovis database not found at path ", rovis_db_path)
        exit(-2)

    semseg_path_exists = os.path.exists(
        os.path.join(rovis_db_path, "datastream_{}".format(semseg_filter_id))
    )

    if overwrite_data is False:
        if semseg_path_exists:
            print(
                "Specified datastream with filter id {} already exists and overwrite option is False".format(semseg_filter_id))
            exit(-3)

    if semseg_path_exists:
        shutil.rmtree(os.path.join(rovis_db_path, "datastream_{}".format(semseg_filter_id)))

    objdet_path_exists = os.path.exists(
        os.path.join(rovis_db_path, "datastream_{}".format(objdet_filter_id))
    )

    if overwrite_data is False:
        if objdet_path_exists:
            print("Specified datastream with filter id {} already exists and overwrite option is False".format(objdet_filter_id))
            exit(-4)

    if objdet_path_exists:
        shutil.rmtree(os.path.join(rovis_db_path, "datastream_{}".format(objdet_filter_id)))


def get_onnx_session():
    # Check model
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    #
    session_options = onnxruntime.SessionOptions()
    onnxrt_session = onnxruntime.InferenceSession(onnx_model.SerializeToString(), session_options)
    onnxrt_session.set_providers(['CUDAExecutionProvider', 'CPUExecutionProvider'], CUSTOM_PROVIDERS)
    return onnxrt_session


def create_folders(semseg_filter_id, objdet_filter_id):
    os.mkdir(os.path.join(rovis_db_path, "datastream_{}".format(semseg_filter_id)))

    os.mkdir(os.path.join(rovis_db_path, "datastream_{}".format(semseg_filter_id), "samples"))
    os.mkdir(os.path.join(rovis_db_path, "datastream_{}".format(semseg_filter_id), "samples", "0"))
    os.mkdir(os.path.join(rovis_db_path,
                          "datastream_{}".format(semseg_filter_id),
                          "samples",
                          "0",
                          "left"))

    os.mkdir(os.path.join(rovis_db_path, "datastream_{}".format(semseg_filter_id), "samples", "1"))
    os.mkdir(os.path.join(rovis_db_path,
                          "datastream_{}".format(semseg_filter_id),
                          "samples",
                          "1",
                          "left"))
    os.mkdir(os.path.join(rovis_db_path, "datastream_{}".format(objdet_filter_id)))


def save_semseg(semseg_prob, instance_detection, semseg_filter_id, semseg_f, ts_start, ts_stop, sampling_time, img_l):
    if show_output:
        seg_vis = visualize_segmentation_image(semseg_prob, img_l, cityscapes_colormap)
        cv2.imshow("Semantic", seg_vis)
        cv2.waitKey(1)

    semseg_image_name = "{}.png".format(ts_stop)
    instance_image_name = "{}.png".format(ts_stop)
    output_image_path = os.path.join(rovis_db_path,
                                     "datastream_{}".format(semseg_filter_id),
                                     "samples",
                                     "0",
                                     "left",
                                     semseg_image_name)

    instance_image_path = os.path.join(rovis_db_path,
                                       "datastream_{}".format(semseg_filter_id),
                                       "samples",
                                       "1",
                                       "left",
                                       instance_image_name)

    if write_instance_image:
        inst_path = "samples/1/left/" + instance_image_name
    else:
        inst_path = ""

    semseg_f.write("{},{},{},{},{}\n".format(
        ts_start,
        ts_stop,
        sampling_time,
        "samples/0/left/" + semseg_image_name,
        inst_path
    ))

    # Save semantic segmentation image
    semseg_img = semseg_prob.cpu().numpy().astype('uint8')
    cv2.imwrite(output_image_path, semseg_img)

    # Save instance segmentation image
    if write_instance_image:
        instance_img = draw_instance_segmentation_image(img_l, instance_detection)
        cv2.imwrite(instance_image_path, instance_img)


def save_objdet(instance_detection, img_l, obj_f, obj_frame_f, ts_start, ts_stop, sampling_time, frame_index, scale_factor_x, scale_factor_y):
    if show_output:
        det_vis = visualize_detection_image(instance_detection, img_l,
                                            cityscapes_instance_label_name)
        det_vis = cv2.cvtColor(det_vis, cv2.COLOR_BGR2RGB)
        cv2.imshow("Instance", det_vis)
        cv2.waitKey(1)

    obj_f.write("{},{},{},{}\n".format(
        ts_start,
        ts_stop,
        sampling_time,
        frame_index))
    write_rois(obj_frame_f,
               frame_index,
               instance_detection,
               tensor_size[0], tensor_size[1],
               scale_factor_x, scale_factor_y)


def add_panoptic_output(semseg_filter_id, objdet_filter_id, image_filter_id):
    rovis_db = RovisDatabaseParser(rovis_db_path)

    print("Creating directory structure for Semantic Segmentation datastream")
    create_folders(semseg_filter_id, objdet_filter_id)

    image_parser = rovis_db.get_data_by_id(filter_id=image_filter_id)

    ts_file = rovis_db.get_timesync_files()[0]

    with open(ts_file, "r") as timesync_f:
        ts_lines = timesync_f.readlines()

    semseg_desc_f = os.path.join(rovis_db_path,
                                 "datastream_{}".format(semseg_filter_id),
                                 "data_descriptor.csv")
    objdet_desc_f = os.path.join(rovis_db_path,
                                 "datastream_{}".format(objdet_filter_id),
                                 "data_descriptor.csv")
    objdet_framebased_f = os.path.join(rovis_db_path,
                                       "datastream_{}".format(objdet_filter_id),
                                       "framebased_data_descriptor.csv")

    image_index = ts_lines[0].strip("\n").strip(",").split(",").index("datastream_{}".format(image_filter_id))

    s_ts_lines = ts_lines[0].strip("\n").strip(",")
    ts_lines[0] = s_ts_lines + ",datastream_{},datastream_{}\n".format(semseg_filter_id, objdet_filter_id)

    onnxrt_session = get_onnx_session()
    device = "cuda"

    with open(semseg_desc_f, "w") as sem_f:
        with open(objdet_desc_f, "w") as obj_f:
            with open(objdet_framebased_f, "w") as obj_frame_f:
                sem_f.write("timestamp_start,timestamp_stop,sampling_time,semseg_img,instance_img\n")
                obj_f.write("timestamp_start,timestamp_stop,sampling_time,frame_id\n")
                obj_frame_f.write("frame_id,roi_id,cls,x,y,width,height\n")
                sem_f.flush()
                obj_f.flush()
                obj_frame_f.flush()

                frame_index = 0
                for i in range(1, len(ts_lines)):
                    s_line = ts_lines[i].strip("\n").strip(",").split(",")
                    ts_stop = int(s_line[image_index])
                    if ts_stop != -1:
                        ts_start, ts_stop, sampling_time, img_l, _ = next(image_parser)
                        scale_factor_x = img_l.shape[1] / tensor_size[0]
                        scale_factor_y = img_l.shape[0] / tensor_size[1]

                        ###################### Inference ###########################
                        img_l = cv2.resize(img_l, tensor_size)
                        data = {'image': img_l}

                        # data pre-processing
                        pixel_mean = [102.9801, 115.9465, 122.7717]
                        pixel_std = [1., 1., 1.]
                        normalize_transform = P.Normalize(mean=pixel_mean, std=pixel_std,
                                                          to_bgr255=True)
                        transform = P.Compose([
                            P.ToTensor(),
                            normalize_transform,
                        ])
                        x_1 = transform(data)
                        onnxrt_inputs = {
                            onnxrt_session.get_inputs()[0].name: x_1
                        }
                        try:
                            prev_time = time.time()
                            onnxrt_outs = onnxrt_session.run(None, onnxrt_inputs)
                            inference_time = time.time() - prev_time
                            print("Inference time: ", inference_time)

                            instance_detection = [torch.from_numpy(o).to(device) for o in onnxrt_outs[0:4]]
                            semseg_logics = [torch.from_numpy(o).to(device) for o in onnxrt_outs[4]]
                            semseg_prob = [torch.argmax(semantic_logit, dim=0) for semantic_logit in semseg_logics]

                            print("Saving semseg: /samples/0/left/{}.png, instance: /samples/1/left/{}.png, objdet {}".format(
                                ts_stop, ts_stop, frame_index))

                            save_semseg(semseg_prob[0],
                                        instance_detection,
                                        semseg_filter_id,
                                        sem_f,
                                        ts_start,
                                        ts_stop,
                                        sampling_time,
                                        img_l)

                            save_objdet(instance_detection,
                                        img_l,
                                        obj_f,
                                        obj_frame_f,
                                        ts_start,
                                        ts_stop,
                                        sampling_time,
                                        frame_index,
                                        scale_factor_x,
                                        scale_factor_y)
                            sem_f.flush()
                            obj_f.flush()
                            obj_frame_f.flush()
                            frame_index += 1

                            ts_line = ts_lines[i].strip("\n").strip(",")
                            ts_line += ",{},{}\n".format(
                                ts_stop, ts_stop
                            )
                            ts_lines[i] = ts_line

                        except onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument:
                            print("Exception occured. Skipping image...")
                            ts_line = ts_lines[i].strip("\n").strip(",")
                            ts_lines[i] = ts_line + ",-1,-1\n"
                        ###########################################################

                    else:
                        ts_line = ts_lines[i].strip("\n").strip(",")
                        ts_lines[i] = ts_line + ",-1,-1\n"

    # Update blockchain file (semseg)
    blockchain_file = os.path.join(rovis_db_path, "datablock_descriptor.csv")
    bk_file = blockchain_file + ".backup"
    if not os.path.exists(bk_file):
        shutil.copy(blockchain_file, bk_file)

    with open(blockchain_file, "r") as b_file:
        lines = b_file.readlines()

    # Check if semseg already present in blockchain file
    write_semseg = True
    for line in lines:
        try:
            l_split = line.strip("\n").strip(",").split(",")
            c_id = int(l_split[0])
            f_id = int(l_split[1])
            if c_id == semseg_core_id and f_id == semseg_filter_id:
                print(
                    "Datastream {} already present in blockchain file {}. Changes will not be made to the file".format(
                        semseg_filter_id, blockchain_file
                    ))
                write_semseg = False
        except ValueError:
            continue
    if write_semseg:
        if "\n" not in lines[-1]:
            lines[-1] += "\n"
        lines.append("{0},{1},{2},{3},{4},{{{5}-{6}}}".format(
            1,  # TODO fix this
            semseg_filter_id,
            "SemanticSegmentation",
            RovisFilterType.ROVIS_SEMANTIC_SEGMENTATION_FILTER_TYPE,
            RovisDataType.ROVIS_IMAGE,
            1,  # TODO fix this
            image_filter_id)
        )
        with open(blockchain_file, "w") as b_file:
            for line in lines:
                b_file.write(line)

    # Write objdet blockchain file
    write_objdet = True
    for line in lines:
        try:
            l_split = line.strip("\n").strip(",").split(",")
            c_id = int(l_split[0])
            f_id = int(l_split[1])
            if f_id == objdet_filter_id:
                print(
                    "Datastream {} already present in blockchain file {}. Changes will not be made to the file".format(
                        objdet_filter_id, blockchain_file
                    ))
                write_objdet = False
        except ValueError:
            continue

    if write_objdet:
        if "\n" not in lines[-1]:
            lines[-1] += "\n"
        lines.append("{0},{1},{2},{3},{4},{{{5}-{6}}}".format(
            1, # TODO fix this
            objdet_filter_id,
            "ObjectDetection2D",
            RovisFilterType.ROVIS_OBJECT_DETECTOR_2D_FILTER_TYPE,
            RovisDataType.ROVIS_2D_ROIS,
            1,  # TODO fix this
            image_filter_id)
        )
        with open(blockchain_file, "w") as b_file:
            for line in lines:
                b_file.write(line)

    # Write timesync file
        # Create backup file
        bk_ts_file = ts_file + ".backup"
        if not os.path.exists(bk_ts_file):
            shutil.copy(ts_file, bk_ts_file)

        with open(ts_file, "w") as tss:
            for line in ts_lines:
                tss.write(line)


if __name__ == "__main__":
    if len(semseg_keys) != len(image_keys):
        print("Images keys must be same size as semseg keys")
        exit(-12)
    if len(objdet_keys) != len(image_keys):
        print("Images keys must be same size as objdet keys")
        exit(-13)

    for semseg_filter_id, objdet_filter_id, image_filter_id in zip(semseg_keys, objdet_keys, image_keys):
        print("Processing semseg: {}, objdet: {}, image: {}".format(
            semseg_filter_id,
            objdet_filter_id,
            image_filter_id
        ))
        check_directories(semseg_filter_id, objdet_filter_id)

        add_panoptic_output(semseg_filter_id, objdet_filter_id, image_filter_id)

    print("Finished")

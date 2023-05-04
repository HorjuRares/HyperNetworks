"""
Copyright (c) RovisLab
RovisDojo: RovisLab neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (s.grigorescu@unitbv.ro)
"""

from absl import app, flags
import os
import io, libconf
import numpy as np


class ObjectClasses(object):
    def __init__(self, object_classes_file):
        assert object_classes_file is not None

        object_classes_file = os.path.join(os.getcwd(), object_classes_file)
        assert os.path.exists(object_classes_file), 'Path to object classes file {} invalid.'.format(object_classes_file)
        self.object_classes_file = object_classes_file

        self.num_classes = 0
        self.background_class = -1
        self.object_classes = []
        self.countable_objects = dict()
        with io.open(object_classes_file) as f:
            classes = libconf.load(f)

            if 'ObjectClasses' in classes:
                for obj_class in classes['ObjectClasses']:
                    self.object_classes.append([classes['ObjectClasses'][obj_class]['ID'], obj_class])
                    self.countable_objects[classes['ObjectClasses'][obj_class]['ID']] = classes['ObjectClasses'][obj_class]['Countable']
                    self.num_classes += 1

            elif 'LocClasses' in classes:
                for obj_class in classes['LocClasses']:
                    self.object_classes.append([classes['LocClasses'][obj_class]['ID'], obj_class])
                    # self.countable_objects[classes['LocClasses'][obj_class]['ID']] = classes['LocClasses'][obj_class]['Countable']
                    self.num_classes += 1

            elif 'EgoClasses' in classes:
                for obj_class in classes['EgoClasses']:
                    self.object_classes.append([classes['EgoClasses'][obj_class]['ID'], obj_class])
                    #self.countable_objects[classes['EgoClasses'][obj_class]['ID']] = classes['EgoClasses'][obj_class]['Countable']
                    self.num_classes += 1

            elif 'ActionClasses' in classes:
                for obj_class in classes['ActionClasses']:
                    self.object_classes.append([classes['ActionClasses'][obj_class]['ID'], obj_class])
                    #self.countable_objects[classes['ActionClasses'][obj_class]['ID']] = classes['ActionClasses'][obj_class]['Countable']
                    self.num_classes += 1

        for cls_id, cls_name in self.object_classes:
            if "background" in cls_name.lower() or "void" or "unlabeled" in cls_name.lower():
                self.background_class = cls_id

    def get_name_by_index(self, index):
        for object_class in self.object_classes:
            if object_class[0] == index:
                return object_class[1]

    def get_class_max_index(self):
        max_idx = 0
        for obj in self.object_classes:
            if obj[0] > max_idx:
                max_idx = obj[0]
        return max_idx

    def get_num_countable_classes(self):
        return self.get_class_max_index() + 1

    def colormap(self):
        return np.array(
            [
                [0, 0, 0],  # First color must be black (used in semantic images)
                [215, 0, 220],
                [165, 42, 42],
                [31, 170, 250],
                [15, 15, 255],
                [65, 90, 210],
                [125, 255, 125],
                [75, 75, 75],
                [0, 255, 255],
                [250, 170, 34],
                [0, 200, 0],
                [0, 150, 20],
                [102, 102, 156],
                [128, 64, 255],
                [140, 140, 200],
                [170, 170, 170],
                [250, 170, 36],
                [250, 170, 160],
                [55, 250, 37],      # road
                [96, 96, 96],
                [230, 150, 140],
                [128, 64, 128],
                [50, 110, 200],     # sky
                [110, 110, 110],
                [244, 35, 232],
                [128, 196, 128],
                [150, 100, 100],
                [70, 70, 70],
                [150, 150, 150],
                [150, 120, 90],
                [220, 20, 60],
                [220, 20, 60],
                [255, 0, 0],
                [255, 0, 100],
                [255, 0, 200],
                [255, 255, 255],
                [255, 255, 255],
                [250, 170, 29],
                [250, 170, 28],
                [250, 170, 26],
                [250, 170, 25],
                [250, 170, 24],
                [250, 170, 22],
                [250, 170, 21],
                [250, 170, 20],
                [255, 255, 255],
                [250, 170, 19],
                [250, 170, 18],
                [250, 170, 12],
                [250, 170, 11],
                [255, 255, 255],
                [255, 255, 255],
                [250, 170, 16],
                [250, 170, 15],
                [250, 170, 15],
                [255, 255, 255],
                [255, 255, 255],
                [0, 255, 0],
                [255, 255, 255],
                [64, 170, 64],
                [230, 160, 50],
                [70, 130, 180],
                [190, 255, 255],
                [152, 251, 152],
                [107, 142, 35],
                [0, 170, 30],
                [255, 255, 128],
                [250, 0, 30],
                [100, 140, 180],
                [220, 128, 128],
                [222, 40, 40],
                [100, 170, 30],
                [40, 40, 40],
                [33, 33, 33],
                [100, 128, 160],
                [20, 20, 255],
                [142, 0, 0],
                [70, 100, 150],
                [250, 171, 30],
                [250, 172, 30],
                [250, 173, 30],
                [250, 174, 30],
                [250, 175, 30],
                [250, 176, 30],
                [210, 170, 100],
                [153, 153, 153],
                [153, 153, 153],
                [128, 128, 128],
                [0, 0, 80],
                [210, 60, 60],
                [250, 170, 30],
                [250, 170, 30],
                [250, 170, 30],
                [250, 170, 30],
                [250, 170, 30],
                [250, 170, 30],
                [192, 192, 192],
                [192, 192, 192],
                [192, 192, 192],
                [220, 220, 0],
                [220, 220, 0],
                [0, 0, 196],
                [192, 192, 192],
                [220, 220, 0],
                [140, 140, 20],
                [119, 11, 32],
                [150, 0, 255],
                [0, 60, 100],
                [0, 0, 142],
                [0, 0, 90],
                [0, 0, 230],
                [0, 80, 100],
                [128, 64, 64],
                [0, 0, 110],
                [0, 0, 70],
                [0, 0, 142],
                [0, 0, 192],
                [170, 170, 170],
                [32, 32, 32],
                [111, 74, 0],
                [120, 10, 10],
                [81, 0, 81],
                [111, 111, 0],
                [0, 0, 0],
                [150, 150, 150],
                [150, 120, 90],
                [220, 20, 60],
                [220, 20, 60],
                [255, 0, 0],
                [255, 0, 100],
                [255, 0, 200],
                [255, 255, 255],
                [255, 255, 255],
                [250, 170, 29],
                [250, 170, 28],
                [250, 170, 26],
                [250, 170, 25],
                [250, 170, 24],
                [250, 170, 22],
                [250, 170, 21],
                [250, 170, 20],
                [255, 255, 255],
                [250, 170, 19],
                [250, 170, 18],
                [250, 170, 12],
                [250, 170, 11],
                [255, 255, 255],
                [255, 255, 255],
                [250, 170, 16],
                [250, 170, 15],
                [250, 170, 15],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [250, 170, 15],
                [150, 120, 90],
                [220, 20, 60],
                [220, 20, 60],
                [255, 0, 0],
                [255, 0, 100],
                [255, 0, 200],
                [255, 255, 255],
                [255, 255, 255],
                [250, 170, 29],
                [250, 170, 28],
                [250, 170, 26],
                [250, 170, 25],
                [250, 170, 24],
                [250, 170, 22],
                [250, 170, 21],
                [250, 170, 20],
                [255, 255, 255],
                [250, 170, 19],
                [250, 170, 18],
                [250, 170, 12],
                [250, 170, 11],
                [255, 255, 255],
                [255, 255, 255],
                [250, 170, 16],
                [250, 170, 15],
                [250, 170, 15],
                [255, 255, 255],
                [255, 255, 255],
                [0, 255, 0],
                [255, 255, 255],
                [64, 170, 64],
                [230, 160, 50],
                [70, 130, 180],
                [190, 255, 255],
                [152, 251, 152],
                [107, 142, 35],
                [0, 170, 30],
                [255, 255, 128],
                [250, 0, 30],
                [100, 140, 180],
                [220, 128, 128],
                [222, 40, 40],
                [100, 170, 30],
                [40, 40, 40],
                [33, 33, 33],
                [100, 128, 160],
                [20, 20, 255],
                [142, 0, 0],
                [70, 100, 150],
                [250, 171, 30],
                [250, 172, 30],
                [250, 173, 30],
                [250, 174, 30],
                [250, 175, 30],
                [250, 176, 30],
                [210, 170, 100],
                [153, 153, 153],
            ]
        )

    def get_countable_classes(self):
        countable = list()

        for k in self.countable_objects:
            if self.countable_objects[k] is True:
                countable.append(k)
        return countable

    def is_countable(self, index):
        for obj in self.object_classes:
            if obj[0] == index:
                return self.countable_objects[obj[0]]
        return False


# def tu_object_classes(_argv):
#     object_classes = ObjectClasses(CFG.ROVIS_VISION.OBJECT_CLASSES_PATH)
#
#     print("Num object classes:", object_classes.num_classes)
#     for obj in object_classes.object_classes:
#         print(obj[0], ":", obj[1])


def get_countable_classes(object_classes_path=None):
    assert object_classes_path is not None

    countable_classes = list()
    object_classes = ObjectClasses(object_classes_path)
    for idx in range(len(object_classes.countable_objects)):
        if object_classes.countable_objects[idx] is True:
            countable_classes.append(idx)
    return countable_classes


def is_countable(index):
    countable_classes = get_countable_classes()
    for countable_index in countable_classes:
        if countable_index == index:
            return True
    return False


# def get_name_by_index(index):
#     object_classes = ObjectClasses(CFG.ROVIS_VISION.OBJECT_CLASSES_PATH)
#     if 0 <= index < len(object_classes.object_classes):
#         return object_classes.object_classes[index][1]
#     return "background"


# def get_background_index():
#     object_classes = ObjectClasses(CFG.ROVIS_VISION.OBJECT_CLASSES_PATH)
#     return object_classes.background_class
#
#
# if __name__ == '__main__':
#     obj_classes = ObjectClasses(CFG.ROVIS_VISION.OBJECT_CLASSES_PATH)
#
#     for obj in obj_classes.object_classes:
#         print(obj[0], obj[1], obj_classes.is_countable(obj[0]))
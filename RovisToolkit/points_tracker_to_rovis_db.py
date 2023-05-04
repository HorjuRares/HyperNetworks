"""
Copyright (c) RovisLab
RovisDojo: RovisLab neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (s.grigorescu@unitbv.ro)
"""

from data.types_ROVIS_TYPES import RovisFilterType, RovisDataType
import os
import shutil

rovis_db_path = "C:/data/RovisDatabases/Scout/08"


def writing_timestamps_sync_csv_file(img_core_id, img_filter_id,
                                     filter_core_id, filter_id):
    # Create timestamps sync backup file
    timestamps_sync_file = os.path.join(rovis_db_path, "sampling_timestamps_sync.csv")
    print(timestamps_sync_file)
    bk_timestamps_file = timestamps_sync_file + ".backup"
    if not os.path.exists(bk_timestamps_file):
        shutil.copy(timestamps_sync_file, bk_timestamps_file)

    # Write csv file
    with open(timestamps_sync_file, "r") as f_ts_sync:
        lines = f_ts_sync.readlines()

    line_elements = lines[0].strip("\n").strip(",").split(",")

    index_of_image_stream = line_elements.index("datastream_{}_{}".format(img_core_id,
                                                                          img_filter_id))
    for i in range(1, len(lines)):
        s_line_elements = lines[i].strip("\n").strip(",").split(",")
        ts_stop_image = int(s_line_elements[index_of_image_stream])
        lines[i] = lines[i].strip("\n").strip(",") + ",{}\n".format(ts_stop_image)

    lines[0] = lines[0].strip("\n").strip(",") + ",datastream_{}_{}\n".format(filter_core_id, filter_id)

    with open(timestamps_sync_file, "w") as f_ts_sync:
        for line in lines:
            f_ts_sync.write(line)


def writing_blockchain_csv_file(img_core_id, img_filter_id,
                                filter_core_id, filter_id):
    # Create blockchain backup file
    blockchain_file = os.path.join(rovis_db_path, "datablock_descriptor.csv")
    bk_blockchain_file = blockchain_file + ".backup"
    if not os.path.exists(bk_blockchain_file):
        shutil.copy(blockchain_file, bk_blockchain_file)

    with open(blockchain_file, "r") as b_file:
        lines = b_file.readlines()

    # Write csv file
    write_points_tracker = True
    for line in lines:
        try:
            l_split = line.strip("\n").strip(",").split(",")
            c_id = int(l_split[0])
            f_id = int(l_split[1])
            if c_id == filter_core_id and f_id == filter_id:
                print("Datastream {}:{} already present in blockchain file {}."
                        "Changes will not be made to the file".format(
                            filter_core_id, filter_id, blockchain_file))
                write_points_tracker = False
        except ValueError:
            continue
    if write_points_tracker:
        if "\n" not in lines[-1]:
            lines[-1] += "\n"
        lines.append("{0},{1},{2},{3},{4},{{{5}-{6}}}".format(
            filter_core_id,
            filter_id,
            "PointsTracker",
            RovisFilterType.ROVIS_POINTS_TRACKER_FILTER_TYPE,
            RovisDataType.ROVIS_POINTS,
            img_core_id,
            img_filter_id)
        )
        with open(blockchain_file, "w") as b_file:
            for line in lines:
                b_file.write(line)


if __name__ == "__main__":
    image_core_id = 1
    image_filter_id = 4

    points_tracker_core_id = 1
    points_tracker_filter_id = 26

    writing_timestamps_sync_csv_file(image_core_id, image_filter_id,
                                     points_tracker_core_id, points_tracker_filter_id)
    writing_blockchain_csv_file(image_core_id, image_filter_id,
                                points_tracker_core_id, points_tracker_filter_id)

    print("Finished")
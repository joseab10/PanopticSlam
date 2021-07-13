from argparse import ArgumentParser
from collections import defaultdict
from os import path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import yaml

from panoptic_slam.kitti.data_loaders import KittiOdomDataYielder
from panoptic_slam.kitti.utils.utils import has_gt
from panoptic_slam.kitti.utils.config import SEMANTIC_LABELS
from panoptic_slam.io.utils import parse_path, mkdir


if __name__ == "__main__":

    parser = ArgumentParser(description="Script for counting the number of different instances for each "
                            "semantic class in KITTI.")

    parser.add_argument("-d", "--kitti_dir", required=True, type=parse_path,
                        help="Path to the root of the KITTI dataset (Parent of the sequence/ and raw/ directories).")

    parser.add_argument("-o", "--output_dir", required=True, type=parse_path,
                        help="Path to the directory where the results will be saved.")

    parser.add_argument("-s", "--fs", default=" ", help="Field separator for when saving as csv.")

    parser.add_argument("-l", "--ls", default="\n", help="Line separator for when saving as csv.")

    args = parser.parse_args()

    kitti_dir = args.kitti_dir

    instance_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    unique_instance_dict = defaultdict(lambda: defaultdict(set))

    sequences = [s for s in range(22) if has_gt(s)]

    print("Parsing Label files.\n")
    for seq in sequences:

        loader = KittiOdomDataYielder(kitti_dir, seq)

        for f, labels in tqdm(enumerate(loader.yield_labels()),
                                          desc="Sequence {:02d}".format(seq),
                                          total=len(loader.get_timestamps(None))):

            _, classes, instances = labels
            classes = np.array(classes["class"].tolist())
            instances = np.array(instances["instance"].tolist())

            unique_classes = np.unique(classes)

            unique_instances = {c: set(np.unique(instances[classes == c]).tolist()) for c in unique_classes}

            for k, v in unique_instances.items():
                unique_instance_dict[seq][k] = unique_instance_dict[seq][k].union(v)
                instance_dict[seq][f][int(k)].extend(list(v))

    sequences = np.array(sorted(unique_instance_dict.keys()))
    classes = sorted(SEMANTIC_LABELS.keys())
    class_labels = [SEMANTIC_LABELS[c] for c in classes]

    instance_counts = np.array([[len(unique_instance_dict[s][c]) for c in classes] for s in sequences]).T

    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(np.clip(instance_counts, 0, 2), cmap="plasma")
    ax.set_xticks(np.arange(len(sequences)))
    ax.set_xticklabels(sequences)
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(class_labels)
    cbar = plt.colorbar(img, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Unseen (0)", "Stuff? (1)", "Things (>1)"])

    mkdir(args.output_dir)
    print("\n Done parsing Label files.\n")

    img_file = path.join(args.output_dir, "kitti_instance_counts.svg")
    fig.savefig(img_file)
    print("Kitti instance counts saved as image to {}.".format(img_file))

    csv_file = path.join(args.output_dir, "kitti_instance_counts.csv")
    np.savetxt(csv_file,
               np.hstack([sequences.reshape((-1, 1)), instance_counts.T]),
               header="seq" + args.fs + args.fs.join([l.replace(" ", "_") for l in class_labels]),
               comments="", delimiter=args.fs, newline=args.ls)
    print("Kitti Instance counts saved as csv to {}.".format(csv_file))

    yaml.add_representer(defaultdict, yaml.representer.Representer.represent_dict)
    yaml.add_representer(np.ndarray, yaml.representer.Representer.represent_list)
    yaml_file = path.join(args.output_dir, "kitti_instance_index.yaml")
    with open(yaml_file, "w") as f:
        yaml.dump(instance_dict, f)
    print("Kitti Instance index (SEQ:FRAME:CLASS:[Instances]) saved to {}.".format(yaml_file))


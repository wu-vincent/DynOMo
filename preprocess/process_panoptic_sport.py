import pickle
import os
import shutil
import glob

sequences = [
    "data/panoptic_sport/boxes/ims/27",
    "data/panoptic_sport/softball/ims/27",
    "data/panoptic_sport/basketball/ims/21",
    "data/panoptic_sport/football/ims/18",
    "data/panoptic_sport/juggle/ims/14",
    "data/panoptic_sport/tennis/ims/8"]


def main():
    for seq in glob.glob("data/panoptic_sport/*/ims/*"):
        seq = seq.replace('\\', '/')
        if seq not in sequences:
            shutil.rmtree(seq)
            if os.path.isdir(seq.replace("ims", "seg")):
                shutil.rmtree(seq.replace("ims", "seg"))


if __name__ == "__main__":
    main()

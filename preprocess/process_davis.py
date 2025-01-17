import pickle
import os
import shutil


def main():
    with open("data/tapvid_davis/tapvid_davis.pkl", 'rb') as pf:
        data = pickle.load(pf)
    
    sequences = data.keys()

    shutil.rmtree("data/DAVIS/ImageSets/")
    for seq in os.listdir("data/DAVIS/Annotations/480p"):
        if seq not in sequences:
            shutil.rmtree(os.path.join(f"data/DAVIS/Annotations/480p/{seq}"))
            shutil.rmtree(os.path.join(f"data/DAVIS/JPEGImages/480p/{seq}"))

if __name__ == "__main__":
    main()
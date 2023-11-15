import os
import shutil


DATASET_PATH = os.path.join("..", "AGH_przetworzone")
DESTINATION_PATH = os.path.join("..", "plane_data")


if __name__ == '__main__':
    classes = {}
    for file in os.listdir(DATASET_PATH):
        filename = os.path.join(DATASET_PATH, file)
        if os.path.isdir(filename):
            for image in os.listdir(filename):
                print(image)
                if ".BMP" in image:
                    image_path = os.path.join(filename, image)
                    image_class = image[image.find("_"): -4]
                    destination_path = os.path.join(DESTINATION_PATH, image_class)
                    os.makedirs(destination_path, exist_ok=True)
                    shutil.copyfile(image_path, os.path.join(destination_path, file + "_" + image))

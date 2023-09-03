"""Convert input data to HDF5 file."""
import os
from tqdm import tqdm
import imageio.v3 as iio
import pandas as pd
import h5py


PATH = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]


def main(png_dimension=512):
    """Read the information in train.csv, convert all paths to local, store images to hdf5."""
    df = pd.read_csv(os.path.join(PATH, "data", "train.csv"))
    df['tmp'] = df['image_path'].str.extract(r'(\d+)\.dcm$')[0]
    def construct_local_image_path(row):
        return os.path.join(PATH, "data", "train_images", str(row["patient_id"]),
                            str(row["series_id"]), f"{row['tmp']}.png")
    df["local_image_path"] = df.apply(construct_local_image_path, axis=1)
    del df["tmp"]

    # Convert unique images to png, add row with image_idx, save df
    with h5py.File(os.path.join(PATH, "data", "train_images", "training.h5"), "w") as hf:
        images_dataset = hf.create_dataset('images', (len(df), png_dimension, png_dimension),
                                           dtype='float32')
        for i, image_path in tqdm(enumerate(df["local_image_path"]), total=len(df)):
            img = iio.imread(image_path)
            images_dataset[i, ...] = img
            df.at[i, "images_idx"] = i

    df.to_json(os.path.join(PATH, "data", "training_labels_df.json"), orient="records")


main()

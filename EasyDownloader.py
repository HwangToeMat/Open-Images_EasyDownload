import argparse
import os
import sys
import requests
import ratelim
import pandas as pd
from tqdm import tqdm
from checkpoints.checkpoints import checkpoints
checkpoints.enable()

# parser settings
parser = argparse.ArgumentParser(
    description="Helper library for downloading OpenImages(https://storage.googleapis.com/openimages/web/index.html) categorically.")
parser.add_argument('--category', action='append',
                    help="Enter the category you want. If you want multi-category, please tag each category.")
parser.add_argument('--type', default="sum", type=str,
                    help="Enter the type of data you want. If you want 'Union data' enter 'sum' else if you want 'intersection data' enter 'inter'.")
parser.add_argument("--ndata", default=-1, type=int,
                    help="Number of data you want")
parser.add_argument("--label", default="https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv", type=str,
                    help="Path of class descriptions file.")
parser.add_argument("--annotation", default="https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv", type=str,
                    help="Path of bbox annotation file.")
parser.add_argument("--imageURL", default="https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv", type=str,
                    help="Path of imageURL file.")
parser.add_argument("--savepath", default="train_data",
                    type=str, help="Path where downloaded data will be saved")


def main():
    opt = parser.parse_args()
    print(opt)
    print('===>> Download or load the csv')

    # Download or load the LabelName of Category
    kwargs = {'header': None, 'names': ['LabelName', 'Category']}
    label = pd.read_csv(opt.label, **kwargs)

    # Download or load the annotation of bbox
    annotation = pd.read_csv(opt.annotation)

    # Download or load the imageURL
    imageURL = pd.read_csv(opt.imageURL)

    # Preprocess the data
    if opt.type == "inter":
        if opt.category == None:
            print('===>> Please enter the categories to create an intersection.')
            return False
        print(f'===>> Category : {opt.category}, Type : {opt.type}')
        Empty_data = True
        for ct in opt.category:
            label_map = dict(label.set_index('Category').loc[[ct], 'LabelName'].to_frame(
            ).reset_index().set_index('LabelName')['Category'])
            label_values = set(label_map.keys())
            temp = annotation[annotation.LabelName.isin(label_values)]
            temp = temp.set_index('ImageID').join(imageURL.set_index(
                'ImageID'))
            if Empty_data:
                URL_data = temp
                Total_data = temp
                Empty_data = False
                continue
            Total_data = pd.concat([Total_data, temp])
            URL_data = pd.merge(URL_data, temp, on='ImageID', how='inner').rename(
                columns={"OriginalURL_x": "OriginalURL"})
        URL_data = URL_data.drop_duplicates(['OriginalURL']).dropna(
            subset=['OriginalURL']).iloc[:opt.ndata, :].loc[:, 'OriginalURL']
    else:
        if opt.category == None:
            print(f'===>> All data, Type : {opt.type}')
            label_map = dict(label.set_index('Category').loc[:, 'LabelName'].to_frame(
            ).reset_index().set_index('LabelName')['Category'])
        else:
            print(f'===>> Category : {opt.category}, Type : {opt.type}')
            label_map = dict(label.set_index('Category').loc[opt.category, 'LabelName'].to_frame(
            ).reset_index().set_index('LabelName')['Category'])
        label_values = set(label_map.keys())
        Total_data = annotation[annotation.LabelName.isin(label_values)]
        Total_data = Total_data.set_index(
            'ImageID').join(imageURL.set_index('ImageID'))
        URL_data = Total_data.drop_duplicates(['OriginalURL']).dropna(
            subset=['OriginalURL']).iloc[:opt.ndata, :].loc[:, 'OriginalURL']

    # Print remaining_todo
    remaining_todo = len(URL_data) if checkpoints.results is None else\
        len(URL_data) - len(checkpoints.results)
    print(f"Parsing {remaining_todo} images "
          f"({len(URL_data) - remaining_todo} have already been downloaded)")

    # Download the images
    with tqdm(total=remaining_todo) as progress_bar:
        Request_data = URL_data.safe_map(
            lambda url: _download_image(url, progress_bar))
        progress_bar.close()

    # Write the images to files, adding them to the package as we go along.
    if not os.path.isdir(f"{opt.savepath}/"):
        os.mkdir(f"{opt.savepath}/")
    if not os.path.isdir(f"{opt.savepath}/images/"):
        os.mkdir(f"{opt.savepath}/images/")
    for ((_, r), (_, url)) in zip(Request_data.iteritems(), URL_data.iteritems()):
        try:
            r.raise_for_status()
            image_name = url.split("/")[-1]
            _write_image(r, image_name, f"{opt.savepath}/images/")
        except:
            continue

    print("===>> Save the images to files")

    # Write the bbox data to csv file.
    if not os.path.isdir(f"{opt.savepath}/bbox/"):
        os.mkdir(f"{opt.savepath}/bbox/")
    label_data = label.set_index(
        'Category').loc[opt.category, 'LabelName'].to_frame().reset_index()
    label_data.to_csv(f"{opt.savepath}/bbox/label_data.csv")
    Total_data.to_csv(f"{opt.savepath}/bbox/bbox_data.csv")
    print("===>> Save the bbox data to csv file")


@ratelim.patient(5, 5)
def _download_image(url, progress_bar):
    """Download a single image from a URL, rate-limited to once per second"""
    r = requests.get(url)
    progress_bar.update(1)
    return r


def _write_image(r, image_name, savepath):
    """Write an image to a file"""
    filename = f"{savepath}/{image_name}"
    with open(filename, "wb") as f:
        f.write(r.content)


if __name__ == '__main__':
    main()

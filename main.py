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
parser.add_argument('--category', default=[], type=list, help="list type")
parser.add_argument('--set', default="sum", type=str,
                    help="If you want 'sum of sets' : 'sum' else if you want 'intersection' : 'inter'")
parser.add_argument("--ndata", default=-1, type=int,
                    help="number of data you want")
parser.add_argument("--label", default="class-descriptions-boxable.csv", type=str,
                    help="path of 'class-descriptions-boxable.csv'")
parser.add_argument("--annotation", default="oidv6-train-annotations-bbox.csv", type=str,
                    help="path of 'xxx-annotations-bbox.csv'")
parser.add_argument("--imageURL", default="train-images-boxable-with-rotation.csv", type=str,
                    help="path of imageURL file.(ex : 'xxx/train-images-boxable-with-rotation.csv')")
parser.add_argument("--datapath", default="train_data", type=str)


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
    print(f'===>> Category : {opt.category}, Type : {opt.set}')
    if opt.set == "inter":
        if opt.category == []:
            print('===>> Please enter a category')
            return False
        Empty_data = True
        for ct in opt.category:
            label_map = dict(label.set_index('Category').loc[[ct], 'LabelName'].to_frame(
            ).reset_index().set_index('LabelName')['Category'])
            label_values = set(label_map.keys())
            Total_data = annotation[annotation.LabelName.isin(label_values)]
            temp = Total_data.set_index('ImageID').join(imageURL.set_index(
                'ImageID')).drop_duplicates(['OriginalURL'])
            if Empty_data:
                URL_data = temp
                Empty_data = False
                continue
            URL_data = pd.merge(URL_data, temp, on='ImageID', how='inner').rename(
                columns={"OriginalURL_x": "OriginalURL"})
        URL_data.iloc[:opt.ndata, :].dropna(
            subset=['OriginalURL']).loc[:, 'OriginalURL']
    else:
        if opt.category == []:
            label_map = dict(label.set_index('Category').loc[:, 'LabelName'].to_frame(
            ).reset_index().set_index('LabelName')['Category'])
        else:
            label_map = dict(label.set_index('Category').loc[opt.category, 'LabelName'].to_frame(
            ).reset_index().set_index('LabelName')['Category'])
        label_values = set(label_map.keys())
        Total_data = annotation[annotation.LabelName.isin(label_values)]
        URL_data = Total_data.set_index('ImageID').join(imageURL.set_index(
            'ImageID')).drop_duplicates(['OriginalURL']).iloc[:opt.ndata, :].loc[:, 'OriginalURL']

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
    if not os.path.isdir(f"{opt.datapath}/"):
        os.mkdir(f"{opt.datapath}/")
    for ((_, r), (_, url)) in zip(Request_data.iteritems(), URL_data.iteritems()):
        image_name = url.split("/")[-1]
        _write_image(r, image_name, opt.datapath)
    print("===>> Save the images to files")

    # Write the bbox data to csv file.
    if not os.path.isdir(f"{opt.datapath}/bbox/"):
        os.mkdir(f"{opt.datapath}/bbox/")
    Total_data.to_csv(f"{opt.datapath}/bbox/bbox_data.csv")
    print("===>> Save the bbox data to csv file")


@ratelim.patient(5, 5)
def _download_image(url, progress_bar):
    """Download a single image from a URL, rate-limited to once per second"""
    try:
        r = requests.get(url)
        r.raise_for_status()
        progress_bar.update(1)
        return r
    except:
        r = requests.get(
            'https://live.staticflickr.com/1/91/227545714_828efc7148_z.jpg')
        r.raise_for_status()
        progress_bar.update(1)
        return r


def _write_image(r, image_name, datapath):
    """Write an image to a file"""
    filename = f"{datapath}/{image_name}"
    with open(filename, "wb") as f:
        f.write(r.content)


if __name__ == '__main__':
    main()

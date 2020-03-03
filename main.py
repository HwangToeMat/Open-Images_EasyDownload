import argparse
import os
import sys
import requests
import ratelim
import pandas as pd
from tqdm-master import tqdm
from checkpoints-master.checkpoints import checkpoints
checkpoints.enable()

# parser settings
parser = argparse.ArgumentParser(
    description="Helper library for downloading OpenImages(https://storage.googleapis.com/openimages/web/index.html) categorically.")
parser.add_argument('--category', default=[], type=list, help="list type")
parser.add_argument('--set', default="sum", type=str,
                    help="If you want 'sum of sets' : 'sum' else you wnat 'intersection' to 'inter'")
parser.add_argument("--ndata", default=-1, type=int,
                    help="number of data you want")
parser.add_argument("--label", default="", type=str,
                    help="path of 'class-descriptions-boxable.csv'")
parser.add_argument("--annotation", default="", type=str,
                    help="path of 'xxx-annotations-bbox.csv'")
parser.add_argument("--imageURL", default="", type=str,
                    help="path of imageURL file.(ex : 'xxx/train-images-boxable-with-rotation.csv')")
opt = parser.parse_args()


def download(category=opt.category, set=opt.set, ndata=opt.ndata, label=opt.label, annotation=opt.annotation, imageURL=opt.imageURL):
    """Download images in categories from flickr"""

    # Download or load the class names pandas DataFrame
    kwargs = {'header': None, 'names': ['LabelID', 'LabelName']}
    orig_url = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
    class_names = pd.read_csv(
        class_names_fp, **kwargs) if class_names_fp else pd.read_csv(orig_url, **kwargs)

    # TODO: setting index_col should not be necessary in this and the next section, update the save-to-disk code
    # Download or load the boxed image metadata pandas DataFrame
    orig_url = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
    train_boxed = pd.read_csv(
        train_boxed_fp, index_col=0) if train_boxed_fp else pd.read_csv(orig_url)

    # Download or load the image ids metadata pandas DataFrame
    orig_url = "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"
    image_ids = pd.read_csv(
        image_ids_fp, index_col=0) if image_ids_fp else pd.read_csv(orig_url)

    # Get category IDs for the given categories and sub-select train_boxed with them.
    # label_map = dict(class_names.set_index('LabelName').loc[categories, 'LabelID']
    #                  .to_frame().reset_index().set_index('LabelID')['LabelName'])
    # label_values = set(label_map.keys())
    # relevant_training_images = train_boxed[train_boxed.LabelName.isin(
    #     label_values)]
    label_map = dict(class_names.set_index('LabelName').loc[[
                     'Football', 'Person'], 'LabelID'].to_frame().reset_index().set_index('LabelID')['LabelName'])
    label_map_1 = dict(class_names.set_index('LabelName').loc[[
                       'Football'], 'LabelID'].to_frame().reset_index().set_index('LabelID')['LabelName'])
    label_map_2 = dict(class_names.set_index('LabelName').loc[[
                       'Person'], 'LabelID'].to_frame().reset_index().set_index('LabelID')['LabelName'])
    label_values_1 = set(label_map_1.keys())
    label_values_2 = set(label_map_2.keys())
    Football_Data = dict(
        train_boxed[train_boxed.LabelName.isin(label_values_1)]['ImageID'])
    Football_Data = set(Football_Data.values())
    Football_Data = train_boxed[train_boxed.ImageID.isin(Football_Data)]
    temp = dict(
        Football_Data[Football_Data.LabelName.isin(label_values_2)]['ImageID'])
    temp = set(temp.values())
    relevant_training_images = Football_Data[Football_Data.ImageID.isin(temp)]

    # Start from prior results if they exist and are specified, otherwise start from scratch.
    relevant_flickr_urls = relevant_training_images.set_index('ImageID').join(
        image_ids.set_index('ImageID')).drop_duplicates(['OriginalURL']).loc[:, 'OriginalURL']

    # relevant_flickr_img_metadata = (relevant_training_images.set_index('ImageID').loc[relevant_flickr_urls.index]
    #                                .pipe(lambda df: df.assign(LabelValue=df.LabelName.map(lambda v: label_map[v]))))

    remaining_todo = len(relevant_flickr_urls) if checkpoints.results is None else\
        len(relevant_flickr_urls) - len(checkpoints.results)

    print(f"Parsing {remaining_todo} images "
          f"({len(relevant_flickr_urls) - remaining_todo} have already been downloaded)")

    # Download the images
    with tqdm(total=remaining_todo) as progress_bar:
        relevant_image_requests = relevant_flickr_urls.safe_map(
            lambda url: _download_image(url, progress_bar))
        progress_bar.close()

    # Initialize a new data package or update an existing one
    # p = t4.Package.browse(packagename, registry) if packagename in t4.list_packages(registry) else t4.Package()

    # Write the images to files, adding them to the package as we go along.
    if not os.path.isdir("train_data/"):
        os.mkdir("train_data/")

    for ((_, r), (_, url)) in zip(relevant_image_requests.iteritems(), relevant_flickr_urls.iteritems()):
        image_name = url.split("/")[-1]
        _write_image_file(r, image_name)

    if not os.path.isdir("train_data/bbox/"):
        os.mkdir("train_data/bbox/")
    relevant_training_images.to_csv('train_data/bbox/bbox_data.csv')


@ratelim.patient(5, 5)
def _download_image(url, pbar):
    """Download a single image from a URL, rate-limited to once per second"""
    try:
        r = requests.get(url)
        r.raise_for_status()
        pbar.update(1)
        return r
    except:
        r = requests.get(
            'https://live.staticflickr.com/1/91/227545714_828efc7148_z.jpg')
        r.raise_for_status()
        pbar.update(1)
        return r


def _write_image_file(r, image_name):
    """Write an image to a file"""
    filename = f"train_data/{image_name}"
    with open(filename, "wb") as f:
        f.write(r.content)


if __name__ == '__main__':
    categories = sys.argv[1:]
    download(categories)

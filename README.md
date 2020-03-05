# Open-Images_EasyDownload
Helper library for downloading OpenImages(https://storage.googleapis.com/openimages/web/index.html) categorically.
## Settings
This code need 'ratelim', 'tqdm' and 'checkpoint'. 'tqdm' and 'checkpoint' are included in this repository. But ***you need to install 'ratelim' using the code below.***
```
pip install ratelim
```

## Usage

```
usage: main.py [-h] [--category CATEGORY] [--type TYPE] [--ndata NDATA]
               [--label LABEL] [--annotation ANNOTATION] [--imageURL IMAGEURL]
               [--savepath SAVEPATH]
               
optional arguments:
  -h, --help           Show this help message and exit
  --category           Enter the category you want. If you want multi-category, 
                       please tag each category. [type=str, default=None(All data)]
  --type               Enter the type of data you want. If you want 'Union data' enter "sum" 
                       else if you want 'intersection data' enter "inter". [type=str, default="sum"]
  --ndata              Number of data you want [type=int, default=-1(All data)]
  --label              Path of class descriptions file. [type=str,
                       default="https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"]
  --annotation         Path of bbox annotation file. [type=str,
                       default="https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"]
  --imageURL           Path of imageURL file. [type=str,
                       default="https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"]
  --savepath           Path where downloaded data will be saved [type=str, default="train_data"]
```
### An example of usage is shown as follows. (*If you use this code at colab, add ! at the beginning.*)
```
python main.py --category "Football" --category "Person" --type "inter" --savepath "Football_data"
```
This example can get images that have **'Football category' and 'Person category' on each image.**

If you enter **"sum"** instead of "inter", you can get images that have **'Football category' or 'Person category' on each image.**

## etc..
Image is saved at "{--savepath}/images/[imageURL].jpg".

Information of bbox is saved at "{--savepath}/bbox/bbox.csv".

***You can use name of imagefile and column('OriginalURL') of 'bbox.csv' to match annotation to image***

### If you want download more faster, change parameters of ratelim in line 117.
```
@ratelim.patient(5, 5) # 5 times in 5 seconds (Gets called at most every 1. seconds)
@ratelim.patient(10, 5) # 10 times in 5 seconds (Gets called at most every 0.5 seconds)
```

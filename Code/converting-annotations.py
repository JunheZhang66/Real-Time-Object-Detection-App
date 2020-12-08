import pandas as pd
import os

#get the path of csv_folder
full_path_to_csv = '/home/ubuntu/open-image-data/OIDv4_ToolKit/OID/csv_folder'

#get the path of all images that downloaded from the Open image dataset
full_path_to_images = \
    '/home/ubuntu/open-image-data/OIDv4_ToolKit/OID/Dataset/train/Car_Bicycle_wheel_Bus_Traffic_light_Jeans_Laptop_Person'

# Create the list to store all category of th image that downloader from the dataset
labels = ['Car', 'Bicycle wheel', 'Bus', 'Traffic light', 'Jeans', 'Laptop', 'Person']

#all needed variables
classes=None
e=[]
encrypted_strings = []
sub_classes=None
annotations=pd.DataFrame()
sub_ann=pd.DataFrame()
r=pd.DataFrame()

def read_class():
    global classes
    classes = pd.read_csv(full_path_to_csv + '/' + 'class-descriptions-boxable.csv',
                          usecols=[0, 1], header=None)

def get_encrypted_strings():
    global encrypted_strings
    global sub_classes
    global e
#     print(classes.head())
#     print(classes[1])
    for v in labels:
        sub_classes = classes.loc[classes[1] == v]
        e = sub_classes.iloc[0][0]
        encrypted_strings.append(e)

def get_annontation():
    global annotations
    global sub_ann
    global encrypted_strings
    annotations = pd.read_csv(full_path_to_csv + '/' + 'train-annotations-bbox.csv',
                              usecols=['ImageID',
                                       'LabelName',
                                       'XMin',
                                       'XMax',
                                       'YMin',
                                       'YMax'])
    sub_ann = annotations.loc[annotations['LabelName'].isin(encrypted_strings)].copy()
#     print(sub_ann.head())

def get_yolo_coordinate():
    global sub_ann
    global encrypted_strings
    global r
    # Adding new empty columns which satisfy yolo data format
    sub_ann['classNumber'] = ''
    sub_ann['center x'] = ''
    sub_ann['center y'] = ''
    sub_ann['width'] = ''
    sub_ann['height'] = ''

    #converting them to numbers
    for i in range(len(encrypted_strings)):
        sub_ann.loc[sub_ann['LabelName'] == encrypted_strings[i], 'classNumber'] = i


    sub_ann['center x'] = (sub_ann['XMax'] + sub_ann['XMin']) / 2
    sub_ann['center y'] = (sub_ann['YMax'] + sub_ann['YMin']) / 2

    sub_ann['width'] = sub_ann['XMax'] - sub_ann['XMin']
    sub_ann['height'] = sub_ann['YMax'] - sub_ann['YMin']

    #extract only needed column and put them into another dataframe
    r = sub_ann.loc[:, ['ImageID',
                        'classNumber',
                        'center x',
                        'center y',
                        'width',
                        'height']].copy()
    print(r.head())

def create_annotation():
    global r
    os.chdir(full_path_to_images)
    #going through all file in current workingdirectory
    for current_dir, dirs, files in os.walk('.'):
        for f in files:
            # locate image
            if f.endswith('.jpg'):
                # extract file name, leave out extension name
                image_name = f[:-4]
                sub_r = r.loc[r['ImageID'] == image_name]

                # add new columns to store the classnumber and coordinate
                resulted_frame = sub_r.loc[:, ['classNumber',
                                               'center x',
                                               'center y',
                                               'width',
                                               'height']].copy()

                #save txt file
                path_to_save = full_path_to_images + '/' + image_name + '.txt'
                resulted_frame.to_csv(path_to_save, header=False, index=False, sep=' ')


def main():
#     get_path(full_path_to_csv,full_path_to_images)
    read_class()
    get_encrypted_strings()
    get_annontation()
    get_yolo_coordinate()
    create_annotation()

main()
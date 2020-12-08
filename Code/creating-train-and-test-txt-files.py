import os


full_path_to_images = \
    '/home/ubuntu/open-image-data/OIDv4_ToolKit/OID/Dataset/train/Car_Bicycle_wheel_Bus_Traffic_light_Jeans_Laptop_Person'

p=[]
p_pest=[]


def data_split():
    # Getting the current directory
    print(os.getcwd())
    # Changing the current directory
    os.chdir(full_path_to_images)
    # Getting the current directory
    print(os.getcwd())
    global p
    global p_test
    for current_dir, dirs, files in os.walk('.'):
        # Going through all files
        for f in files:
            # Checking if filename ends with '.jpg'
            if f.endswith('.jpg'):
                path_to_save_into_txt_files = full_path_to_images + '/' + f

                p.append(path_to_save_into_txt_files + '\n')

    p_test = p[:int(len(p) * 0.15)]

    # Deleting from initial list first 15% of elements
    p = p[int(len(p) * 0.15):]



def write_file():
    # Creating file train.txt and writing 85% of lines in it
    with open('train.txt', 'w') as train_txt:
        # Going through all elements of the list
        for e in p:
            # Writing current path at the end of the file
            train_txt.write(e)

    # Creating file test.txt and writing 15% of lines in it
    with open('test.txt', 'w') as test_txt:
        # Going through all elements of the list
        for e in p_test:
            # Writing current path at the end of the file
            test_txt.write(e)

def main():
    data_split()
    write_file()
main()
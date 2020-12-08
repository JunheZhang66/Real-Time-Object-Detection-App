#getting work directory
full_path_to_images = \
    '/home/ubuntu/open-image-data/OIDv4_ToolKit/OID/Dataset/train/Car_Bicycle_wheel_Bus_Traffic_light_Jeans_Laptop_Person'
# Defining counter for classes
c=0


def creat_names():
    global c
    # Creating file classes.names from existing one classes.txt
    with open(full_path_to_images + '/' + 'classes.names', 'w') as names, \
            open(full_path_to_images + '/' + 'classes.txt', 'r') as txt:
        # Going through all lines
        for line in txt:
            names.write(line)
            c += 1

def creat_data():
    # Creating file custom_data.data
    with open(full_path_to_images + '/' + 'custom_data.data', 'w') as data:

        # Number of classes
        data.write('classes = ' + str(c) + '\n')

        # Location of the train.txt file
        data.write('train = ' + full_path_to_images + '/' + 'train.txt' + '\n')

        # Location of the test.txt file
        data.write('valid = ' + full_path_to_images + '/' + 'test.txt' + '\n')

        # Location of the classes.names file
        data.write('names = ' + full_path_to_images + '/' + 'classes.names' + '\n')

        # Location where to save weights
        data.write('backup = backup')

def main():
    creat_names()
    creat_data()
    print(c)
main()
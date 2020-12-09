
full_path_to_images = \
'/home/ubuntu/open-image-data/OIDv4_ToolKit/OID/Dataset/train/Car_Bicycle_wheel_Bus_Traffic_light_Jeans_Laptop_Person'
# counter
c=0


def creat_names():
    global c
    with open(full_path_to_images + '/' + 'classes.names', 'w') as names, \
            open(full_path_to_images + '/' + 'classes.txt', 'r') as txt:
        for line in txt:
            names.write(line)
            c += 1

def creat_data():
    with open(full_path_to_images + '/' + 'custom_data.data', 'w') as data:
        #write .data
        data.write('classes = ' + str(c) + '\n')
        data.write('train = ' + full_path_to_images + '/' + 'train.txt' + '\n')
        data.write('valid = ' + full_path_to_images + '/' + 'test.txt' + '\n')
        data.write('names = ' + full_path_to_images + '/' + 'classes.names' + '\n')
        data.write('backup = backup')

def main():
    creat_names()
    creat_data()
    print(c)
main()

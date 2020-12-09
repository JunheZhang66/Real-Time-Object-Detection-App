import os


full_path_to_images = \
'/home/ubuntu/open-image-data/OIDv4_ToolKit/OID/Dataset/train/Car_Bicycle_wheel_Bus_Traffic_light_Jeans_Laptop_Person'

p=[]
p_pest=[]

#data split
def data_split():
    os.chdir(full_path_to_images)
    print(os.getcwd())
    global p
    global p_test
    for current_dir, dirs, files in os.walk('.'):
        for f in files:
            if f.endswith('.jpg'):
                path_to_save_into_txt_files = full_path_to_images + '/' + f
                p.append(path_to_save_into_txt_files + '\n')
    p_test = p[:int(len(p) * 0.15)]
    p = p[int(len(p) * 0.15):]


# Creating file train.txt and test.txt
def write_file():
    with open('train.txt', 'w') as train_txt:
        for e in p:
            train_txt.write(e)

    with open('test.txt', 'w') as test_txt:
        for e in p_test:
            test_txt.write(e)

def main():
    data_split()
    write_file()
main()

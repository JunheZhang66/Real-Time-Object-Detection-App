
'''
Auther: Junhe Zhang
'''
# To run the code, scroll down to the bottom of the file

# import necessary libraries
import numpy as np
import cv2
import time
# read img
# return img blob, width and height
def read_image(img_dir, video = 0):
    # img should be absolute directory of image
    # cv2 image has format blue green red
    if not video:
        image_BGR = cv2.imread(img_dir)
    else:
        image_BGR = img_dir
    # getting height and width of image
    h, w = image_BGR.shape[:2] # first two value contains height and width
    blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416),
                             swapRB=True, crop=False)
    # reshape blob from (1, 3, 416, 416 ) to (416, 416, 3)
    blob_clean = blob[0, :, :, :].transpose(1, 2, 0)
    return (image_BGR, blob, w, h)

# read label dir
# return list of labels
def read_labels(label_dir):
    labels = []
    with open(label_dir) as f:
        labels = [line.strip() for line in f]
    return labels

# load and return trained YOLO model
def load_model(cfg_dir, weights_dir):
    return cv2.dnn.readNetFromDarknet(cfg_dir, weights_dir)

# return YOLO model's names
# (all layers names, output layers names)
def get_model_names(model):
    # get all layers names
    all_layers_names = model.getLayerNames()
    # get output layers names only
    # output layers are layer 82, 94 and 106
    out_layers_names = [all_layers_names[layer[0] - 1] for layer in model.getUnconnectedOutLayers()]
    return all_layers_names, out_layers_names

# process image on trained model
# return list of bounding box of detected objects
# img has to be converted to blob beforehead
def process_image(model, output_layers, img):
    print('Start process image')
    start = time.time()
    model.setInput(img)
    end = time.time()
    print('End process image, processing time {:.5f}'.format(end-start))
    return model.forward(output_layers)

# process output for three layers: 82, 94, 106
# return bounding boxes, confidence and class label for each identified object
def process_output(output_from_network, threshold, w, h):
    # Preparing lists for detected bounding boxes,
    # obtained confidences and class's number
    bounding_boxes = []
    confidences = []
    class_numbers = []
    for result in output_from_network:
        # Going through all detections from current output layer
        for detected_objects in result:
            # Getting 80 classes' probabilities for current detected object
            scores = detected_objects[5:]
            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)
            # Getting value of probability for defined class
            confidence_current = scores[class_current]

            # Eliminating weak predictions with minimum probability
            if confidence_current > threshold:

                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                # Now, from YOLO data format, we can get top left corner coordinates
                # that are x_min and y_min
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Adding results into prepared lists
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)
    return bounding_boxes, confidences, class_numbers

# apply Non-Maximun-Suppression technic on collected bounding boxex
# remove multiple bounding boxes on the same object with lower confidence
def non_maximun_suppression(bounding_boxes, confidences, class_numbers, probability_minimum, threshold):
    return cv2.dnn.NMSBoxes(bounding_boxes, confidences,\
                           probability_minimum, threshold)

def draw_detected_object(results, image_BGR, labels, bounding_boxes, confidences, class_numbers, colors):
    if len(results) > 0:
        # Going through indexes of results
        for i in results.flatten():

            # Getting current bounding box coordinates,
            # its width and height
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            # Preparing colour for current bounding box
            # and converting from numpy array to list
            color_box_current = colors[class_numbers[i]].tolist()

            # Drawing bounding box on the original image
            cv2.rectangle(image_BGR, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          color_box_current, 2)

            # Preparing text with label and confidence for current bounding box
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                   confidences[i])

            # Putting text with label and confidence on the original image
            cv2.putText(image_BGR, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, color_box_current, 2)

# run image_main to apply YOLOV3 on image
def image_main():
    # setting directories for processing
    img_dir = '/home/ubuntu/darknet/darknet/data/laptop-jean-test2.jpg'
    label_dir = '/home/ubuntu/open-image-data/OIDv4_ToolKit/OID/Dataset/train/Car_Bicycle_wheel_Bus_Traffic_light_Jeans_Laptop_Person/classes.names'
    cfg_dir = '/home/ubuntu/darknet/darknet/cfg/yolov3-custom_train.cfg'
    weights_dir = '/home/ubuntu/darknet/darknet/backup/yolov3-custom_train_last.weights'
    
    # getting img_bgr, img blob, width and height 
    img, blob, w, h = read_image(img_dir)
    # reading classes, example ['people','car']
    labels = read_labels(label_dir)
    # loading trained model
    model = load_model(cfg_dir, weights_dir)
    # getting output layers' names. example ['yolo82', 'yolo94', 'yolo106']
    all_layers_names, output_layers_names = get_model_names(model)
    # process blob image on trained model
    model_output = process_image(model, output_layers_names, blob)
    # setting accuracy threshold
    threshold = 0.25
    # process model output and collect bounding boxes, confidences and class label for each
    # detected object in blob image
    bounding_boxes, confidences, class_numbers = process_output(model_output, threshold, w, h)
    
    # Setting minimum probability to eliminate weak predictions
    probability_minimum = 0.5
    # Setting threshold for filtering weak bounding boxes
    # with non-maximum suppression
    threshold = 0.3
    # Remove multiple bounding boxes on same object with lower confidence
    results = non_maximun_suppression(bounding_boxes, confidences, class_numbers, probability_minimum, threshold)
    
    # Generating colours for representing every detected object
    # example: classes = ['people','car'], colors = [(0, 100, 200), (100, 200, 254)]
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
    # draw bounding boxes and corresponding label on the image
    draw_detected_object(results, img, labels, bounding_boxes, confidences, class_numbers, colors)
    # save modified image using openCV
    cv2.imwrite('/home/ubuntu/darknet/darknet/data/result-laptop-jean-test2.jpg', img)

# run video_main to apply YOLOV3 on video
def video_main():
    # setting directories for processing
    video_dir = '/home/ubuntu/darknet/darknet/data/'
    # setting video name
    video_name = 'buses-to-test'
    label_dir = '/home/ubuntu/open-image-data/OIDv4_ToolKit/OID/Dataset/train/Car_Bicycle_wheel_Bus_Traffic_light_Jeans_Laptop_Person/classes.names'
    cfg_dir = '/home/ubuntu/darknet/darknet/cfg/yolov3-custom_train.cfg'
    weights_dir = '/home/ubuntu/darknet/darknet/backup/yolov3-custom_train_last.weights'
    # create video object using openCV
    video = cv2.VideoCapture(video_dir+video_name+'.mp4')
    # getting class labels
    labels = read_labels(label_dir)
    # loading trained model
    model = load_model(cfg_dir, weights_dir)
    # getting output layers' names example ['yolo82', 'yolo94', 'yolo106']
    all_layers_names, output_layers_names = get_model_names(model)
    # Generating colours for representing every detected object
    # example: classes = ['people','car'], colors = [(0, 100, 200), (100, 200, 254)]
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
    # initalize writer for saving generated video
    writer = None
    # count number of frames of the video
    f = 0
    # while video is not end
    while True:
        # Capturing frame-by-frame
        print('frame: ', f)
        # ret - whether receiving any image
        ret, img = video.read()

        # If the frame was not retrieved break loop (end of the video)
        if not ret:
            break
        
        img, blob, w, h = read_image(img, 1)
            
        model_output = process_image(model, output_layers_names, blob)
        
        threshold = 0.25
        bounding_boxes, confidences, class_numbers = process_output(model_output, threshold, w, h)
        
        # Setting minimum probability to eliminate weak predictions
        probability_minimum = 0.25
        # Setting threshold for filtering weak bounding boxes
        # with non-maximum suppression
        threshold = 0.3
        results = non_maximun_suppression(bounding_boxes, confidences, class_numbers, probability_minimum, threshold)
        
        draw_detected_object(results, img, labels, bounding_boxes, confidences, class_numbers, colors)
        
        if writer is None:
            # Constructing code of the codec
            # to be used in the function VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # Writing current processed frame into the video file
            writer = cv2.VideoWriter(video_dir+'result-'+video_name+'.avi', fourcc, 30,
                                     (img.shape[1], img.shape[0]), True)
        writer.write(img)
        f += 1

# change directory before running on your machine
print('start processing image')
image_main()
print('end processing image, processed image saved')
print('\n')
print('start processing video')
video_main()
print('end processing video, processed video saved')
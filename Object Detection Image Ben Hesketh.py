import cv2
import numpy as np

print("For example: 'benGranTurismo1.jpg'")
imageFile = input("Enter the filename of the sample image (requires the extension): ")

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')  # variable used to read the weight and configuration files.
classes = []  # we store the object names into a list from the .txt. file here.
with open('coco.txt', 'r') as readOut:  # variable 'readOut' is used in next line, read() function is used to print
    # out the objects.
    classes = readOut.read().splitlines()

# print out all classes from the current COCO text file. 
print(classes)

test1 = cv2.imread(imageFile)  # 'imread' is a cv2 function!
height, width, _ = test1.shape  # extract the height and width dimensions from the image.

creatorFont = cv2.FONT_HERSHEY_COMPLEX_SMALL
cv2.putText(test1, 'Created by Ben Hesketh', (30, 30), creatorFont, 1, (225, 0, 255), 2, cv2.LINE_4)

    #  Next we prepare the test image for Yolo by resizing image.
    #  We also need to normalise the image by dividing the pixels by 255.
    #  Needs to be changed from BG out to RB out.

blob = cv2.dnn.blobFromImage(test1, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

net.setInput(blob)  # Binary Large Objects are set as the input for the model.
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)  # Output layer names are sent using the forward function.

boxes = []
confidences = []
class_ids = []

#  LOOPS (for each output of detection, 4 bounding box offsets, 1 box condifence calls, 80 class predictions).
for output in layerOutputs:
    for detection in output:
        points = detection[5:]
        class_id = np.argmax(points)  # We are using Numpy function to find biggest points.
        confidence = points[class_id]  # Extract the biggest points and assign into confidence variable.
        if confidence > 0.5:  # THRESHOLD!
            center_x = int(detection[0] * width)  # YOLO uses the centers of the bounding boxes.
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)  # This gives us the positions of the upper left corner.
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

print(len(boxes))  # TELLS US NUMBER OF DETECTIONS IN CONSOLE (8 for test 1 image).
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)  # (bboxes, scores, score_threshold, nms_threshold)
print(indexes.flatten())  # TELLS US THE REDUNDANCIES IN CONSOLE (only detections 2, 6, 7 were not redundant)

font = cv2.FONT_HERSHEY_PLAIN
font_scale = 2
colours = np.random.uniform(0, 255, size=(len(boxes), 3))  # Defining color using numpy 'random' (low, high, size)

#  FOR LOOP FOR EACH OBJECT DETECTED
if len(indexes) > 0 and len(indexes) <= 24:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]  # Locations and size of rectangle
        if class_ids[i] >= 24: continue # FIX! Essentially skips the error when the range of the array exceeds the limit
        label = str(classes[class_ids[i]])  # extracting string text from the name of the class extracted from
        # the .txt file.
        confidence = str(round(confidences[i], 2))  # extracting the threshold score, i is just a number id.
        colour = colours[i]  # each object deection box is given a color each.
        cv2.rectangle(test1, (x, y), (x + w, y + h), colour, 4)  # Creating the borders of the boxes (verticle and
        # horizontal + width).
        cv2.putText(test1, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)  # The text which
        # is name of class + threshold score + position + font + font size + colour, thickness

    #  3 versions of the same image appears in seperate windows, each with one channel (R,G,B).
#  Blob stands for Binary Large OBjects which is a number of bytes treated as a object.
for b in blob:

    for n, Image_blob in enumerate(b):
        cv2.imshow(str(n), Image_blob)

key = cv2.waitKey(1)
# this is the delay function (1000 seems to be a second?)
cv2.imshow('Image', test1)  # first test image is replaced by the second image in the same window.
cv2.waitKey(50000)
cv2.destroyAllWindows()  # After 3 seconds, windows are closed.

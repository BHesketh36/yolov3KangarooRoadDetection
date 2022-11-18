import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')  # variable used to read the weight and configuration files.

classes = []  # we store the object names into a list from the .txt. file here.
classFile = 'coco.txt'
with open(classFile, 'rt') as f: #rt as in read it!
    classes = f.read().rstrip('\n').split('\n') # strip it onto a new line using /n

print(classes)


liveCam = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
welcomeMsg = "Hello, this is my Object Detection Implementation developed with YOLOv3 and OpenCV"
creator = "Ben Hesketh"
numberId = "18008836"
closeMsg = "Press the Escape Key (ESC) to close down the application"
# Attempt to output the liveCam feed with 640 x 480 resolution

print(welcomeMsg)
print(creator)
print(numberId)
print(closeMsg)

# Debugging opening video errors
if not liveCam.isOpened():
    liveCam = cv2.VideoCapture(0)
if not liveCam.isOpened():
    raise IOError("ERROR: Video Camera could not be accessed")
if liveCam.isOpened():
    print("Video camera found!")

while True:
    _, feed = liveCam.read()
    height, width, _ = feed.shape  # extract the height and width dimensions from the image.

    #  Next we prepare the live feed for Yolo by resizing the window.
    #  We also need to normalise the image by dividing the pixels by 255.
    #  Needs to be changed from BG out to RB out.

    blob = cv2.dnn.blobFromImage(feed, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

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

    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 2
    colours = np.random.uniform(0, 255, size=(len(boxes), 3))  # Defining color using numpy 'random' (low, high, size)

    # cv2.putText(feed, "Created by Ben Hesketh", (x, y + 20), font, 2, (0, 0, 255), 2)

    #  FOR LOOP FOR EACH OBJECT DETECTED
    if len(indexes) > 0 and len(indexes) <= 26:
        for i in indexes.flatten():
                x, y, w, h = boxes[i]  # Locations and size of rectangle
                if class_ids[i] >= 26: continue
                label = str(classes[class_ids[i]])  # extracting string text from the name of the class extracted from
                # the .txt file.
                confidence = str(round(confidences[i], 2))  # extracting the threshold score, i is just a number id.
                colour = colours[i]  # each object deection box is given a color each.
                cv2.rectangle(feed, (x, y), (x + w, y + h), colour, 4)  # Creating the borders of the boxes (verticle and
                # horizontal + width).
                cv2.putText(feed, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)  # The text which
                # is name of class + threshold score + position + font + font size + colour, thickness

    #  3 versions of the same image appears in seperate windows, each with one channel (R,G,B).
    #  Blob stands for Binary Large OBjects which is a number of bytes treated as a object.
    for b in blob:

        for n, feed_blob in enumerate(b):
            cv2.imshow(str(n), feed_blob)

    cv2.imshow('Image', feed)
    newWaitKey = cv2.waitKey(1)
    if newWaitKey == 27: # required to break the while loop (27 is the ESC key in ASCII)
        break
liveCam.release()
cv2.destroyAllWindows()  # After 3 seconds, windows are closed.

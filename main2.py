import cv2
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=False)
parser.add_argument('--play_video', help="Tue/False", default=False)
parser.add_argument('--image', help="Tue/False", default=False)
parser.add_argument('--video_path', help="Path of video file", default="demo1.mp4")
parser.add_argument('--image_path', help="Path of image to detect objects", default="venv/images/armas (1).jpg")
parser.add_argument('--verbose', help="To print statements", default=True)
args = parser.parse_args()


# Load yolo
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


def load_image(img):
    # image loading
    img = cv2.imread("C:/Users/jonna/PycharmProjects/pythongun/images")
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels


def start_webcam():
    cap = cv2.VideoCapture(0)

    return cap


def display_blob(blob):
    '''
        Three images each for RED, GREEN, BLUE channel
    '''
    for b in blob:
        for n, img in enumerate(b):
            cv2.imshow(str(n), img)


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    img = cv2.resize(img, (800, 600))
    cv2.imshow("Image", img)


def image_detect(img):
    model, classes, colors, output_layers = load_yolo()
    image, height, width, channels = load_image(img)
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    draw_labels(boxes, confs, colors, class_ids, classes, image)
    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break


def webcam_detect():
    model, classes, colors, output_layers = load_yolo()
    cap = start_webcam()
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()


def start_video(video_play):
    model, classes, colors, output_layers = load_yolo()
    cap = cv2.VideoCapture("demo1.mp4")

    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame)

        key = cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


def detect_gun_firearms(boxes, confs, class_ids, classes):
    gun_positions = []
    firing_positions = []
    for i in range(len(boxes)):
        if classes[class_ids[i]] == 'gun':
            x, y, w, h = boxes[i]
            gun_positions.append((x, y, x + w, y + h))  # Appending gun bounding box coordinates

        if classes[class_ids[i]] == 'person':
            x, y, w, h = boxes[i]
            firing_positions.append((x, y, x + w, y + h))  # Appending person bounding box coordinates

    return gun_positions, firing_positions

def detect_from_input(input_path):
    model, classes, colors, output_layers = load_yolo()

    if input_path.endswith(('jpg', 'png', 'jpeg')):  # For image paths
        image = cv2.imread(input_path)
        image, height, width, channels = load_image(image)
        blob, outputs = detect_objects(image, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, image)
        gun_positions, firing_positions = detect_gun_firearms(boxes, confs, class_ids, classes)
        print("Gun positions:", gun_positions)
        print("Firing positions:", firing_positions)

    else:  # For video paths or webcam
        if input_path.isdigit():  # Check if input is a digit (for webcam)
            cap = cv2.VideoCapture(int(input_path))
        else:  # Assume input is a video file path
            cap = cv2.VideoCapture(input_path)

        while True:
            _, frame = cap.read()
            if frame is None:
                break
            height, width, channels = frame.shape
            blob, outputs = detect_objects(frame, model, output_layers)
            boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
            draw_labels(boxes, confs, colors, class_ids, classes, frame)
            gun_positions, firing_positions = detect_gun_firearms(boxes, confs, class_ids, classes)
            print("Gun positions:", gun_positions)
            print("Firing positions:", firing_positions)
            key = cv2.waitKey(1)
            if key == 27:
                break
        cap.release()

if __name__ == '__main__':
    if args.image:
        img_path = args.image_path
        if args.verbose:
            print("Opening " + img_path + " .... ")
        detect_from_input(img_path)

    if args.play_video:
        video_path = args.video_path
        if args.verbose:
            print('Opening ' + video_path + " .... ")
        detect_from_input(video_path)

    if args.webcam:
        if args.verbose:
            print('---- Starting Web Cam object detection ----')
        detect_from_input(args.webcam)
    cv2.destroyAllWindows()








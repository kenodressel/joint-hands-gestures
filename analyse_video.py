# hardcoded data for video anaylsis
src = "./RG_2009_11_10_2/"
name = "RG_2009_11_10_2"
new_name = "RG_2009_11_10_2"
video_ending = ".mp4"

# print the name of the file to identify each process
print(new_name)

# Required moduls
import cv2
import numpy as np

# get center of any contour
def get_center(c):
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return np.array([cX, cY])

# add a empty dataset for a new frame
def add_empty_row(data, frame):
    data = np.hstack((data, np.empty((4, 1, 2))))
    data[0, frame] = frame
    return data

# "predict" the movement (take last position)
def predict_movement(row):
    # temp
    new_center = row[-2, 0]
    expected_size = row[-2, 1]
    return [new_center, expected_size]

# add any center to the data
def append_center(data, contour, index):
    # append left
    cX, cY = get_center(contour)
    data[index, frame] = [np.array([cX, cY]), cv2.contourArea(contour)]

# add a predicted center (need to calculate the contours)
def append_predicted_center(data, index, predict=True):
    if (predict):
        [new_center, expected_size] = predict_movement(data[index])
    else:
        new_center = data[index, -2, 0]
        expected_size = data[index, -2, 1]

    cX, cY = new_center
    cX = int(cX)
    cY = int(cY)
    data[index, frame] = [np.array([cX, cY]), expected_size]

# in case of debug print debug info
def debug(*args):
    # print(args)
    pass

# Color identifier. Click anywhere anytime on the video to get color information
color_pos = [0, 0]
color_check = False


def click_and_crop(event, x, y, flags, param):
    if (event != cv2.EVENT_LBUTTONDOWN):
        return
    global color_pos, color_check
    color_pos = [x, y]
    color_check = True



# Check for Colorranges of skincolor. YCBCR and HSV
# min_YCrCb = np.array([0, 135, 77], np.uint8) 
# max_YCrCb = np.array([255, 170, 125], np.uint8) 
min_HSV = np.array([0,40,170],np.uint8)
max_HSV = np.array([20,140,255],np.uint8)

# color [185 152 105]
# Create a window to display the camera feed
cv2.namedWindow('Camera Output')

# Get pointer to video frames from primary device
cap = cv2.VideoCapture(src + name + video_ending)
cv2.setMouseCallback("Camera Output", click_and_crop)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

data = np.empty((4, 1, 2), dtype=object)  # frame, left hand, right hand, center

frame = 0
offset = 0
# cap.set(1,1300)

# define weights
weights = np.array([np.linspace(1, 1, 5), np.linspace(1, 1, 5)]).T

import time

start = time.time()
while (cap.isOpened()):
    # Grab video frame, decode it and return next video frame
    readSuccess, sourceImage = cap.read()

    if (not readSuccess):
        break

    # Convert image to YCrCb
    # imageYCrCb = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2YCR_CB)
    imageYCrCb = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2HSV)

    # Color check
    if (color_check):
        color_check = False
        [x_c, y_c] = color_pos
        print("color", imageYCrCb[y_c, x_c])

    # Find region (HSV or YCBCR)
    #skinRegion = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
    skinRegion = cv2.inRange(imageYCrCb, min_HSV, max_HSV)


    # contour detection
    _, contours, _ = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []

    # filter contours
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)

        # thresholds to avoide capturing too small or too big areas
        if area > 200 and area < 6500 \
                and c[0, 0, 1] > sourceImage.shape[0] // 2 \
                and c[0, 0, 0] > sourceImage.shape[1] // 4 \
                and c[0, 0, 0] < (sourceImage.shape[1] // 4) * 3:
            valid_contours.append(c)

	# draw contours    
    cv2.drawContours(sourceImage, valid_contours, -1, (50, 50, 50), 3)

    # no contours..
    if (len(valid_contours) == 0):

        if (frame < 5):
            offset += 1
            continue

        # predict left hand
        append_predicted_center(data, 1, False)

        # predict right hand
        append_predicted_center(data, 2, False)

        debug("NO VALID CONTOURS", frame)

        data = add_empty_row(data, frame)

    if (len(valid_contours) == 1):
        # both hands together

        if (frame < 5):
            offset += 1
            # data = add_empty_row(data, frame)
            continue

        area = cv2.contourArea(valid_contours[0])
        debug("AREA", area)

        if area < w * h * 0.005:
            # one hand is obscured (for now do nothing)
            debug("PREDICT ONE HAND START")

            [expected_direction1, expected_size1] = predict_movement(data[1])
            [expected_direction2, expected_size2] = predict_movement(data[2])

            center = get_center(valid_contours[0])

            # calculate offset to both centers
            offset1 = np.linalg.norm(center - expected_direction1)
            offset2 = np.linalg.norm(center - expected_direction2)

            debug("expected_change", expected_direction1, expected_direction2)
            debug("offset", offset1, offset2)
            debug("center", center)
            debug("recent", data[1, -2, 0], data[2, -2, 0])
            debug("PREDICT ONE HAND  END", frame)

            # if offset1 is the smaller,
            if (offset1 < offset2):
                append_center(data, valid_contours[0], 1)
                append_predicted_center(data, 2, False)
            else:

                append_predicted_center(data, 1, False)
                append_center(data, valid_contours[0], 2)

            data = add_empty_row(data, frame)


        else:
            debug("GIANT HAND", frame)
            # calculate currenct center
            cX, cY = get_center(valid_contours[0])

            # predict left hand
            append_predicted_center(data, 1, False)

            # predict right hand
            append_predicted_center(data, 2, False)

            # add new data column
            data = add_empty_row(data, frame)

            # save frame info
            append_center(data, valid_contours[0], 3)
            data[3, frame] = [(cX, cY), cv2.contourArea(valid_contours[0])]

    elif (len(valid_contours) == 2):
        # two seperate hand
        [c1, c2] = valid_contours

        # find out which is which
        center_1 = get_center(c1)
        center_2 = get_center(c2)

        if (frame < 2):
            debug("BELOW 2", center_1, center_2)
            # very basic identification
            left_contour = c1
            right_contour = c2
            if (c1[0, 0, 0] < c2[0, 0, 0]):
                # c1 is right, c2 is left
                right_contour = c1
                left_contour = c2

        elif (frame < 9):
            debug("BELOW 9", center_1, center_2)
            debug("RECENT", len(data), data[1, -3], data[2, -3])
            distances_l = [np.linalg.norm(c - data[1, frame]) for c in (center_1, center_2)]
            distances_r = [np.linalg.norm(c - data[2, frame]) for c in (center_1, center_2)]
            left_contour = (c1, c2)[np.argmin(distances_l)]
            right_contour = (c1, c2)[np.argmin(distances_r)]

        else:
            # more than eight entries already
            # predict next point for left hand

            row = data[1]

            [expected_direction, expected_size] = predict_movement(row)

            # calculate offset to both centers
            debug("expected change for left", expected_direction)
            offset1 = np.linalg.norm(center_1 - expected_direction)
            offset2 = np.linalg.norm(center_2 - expected_direction)

            debug("offset", offset1, offset2)
            debug("center", center_1, center_2)
            debug("recent", data[1, -2, 0], data[2, -2, 0])
            debug("END", frame)

            # if offset1 is the smaller,
            if (offset1 < offset2):
                left_contour = c1
                right_contour = c2
            else:
                left_contour = c2
                right_contour = c1

        # add empty row
        debug("ADDING ROW")
        debug(len(data[0]))
        data = add_empty_row(data, frame)

        append_center(data, left_contour, 1)
        append_center(data, right_contour, 2)
        debug(len(data[0]))

    # more than two hands...
    elif (len(valid_contours) > 2):

        if (frame < 5):
            offset += 1
            continue

        debug("MORE THAN TWO CONTOURS")

        [pos_left, size_left] = predict_movement(data[1])
        [pos_right, size_right] = predict_movement(data[2])

        distances_l = []
        distances_r = []

        # check distances to each predicted point
        for c in valid_contours:
            center = get_center(c)
            area = cv2.contourArea(c)
            distances_l.append(np.linalg.norm(center - pos_left) + (area - size_left))
            distances_r.append(np.linalg.norm(center - pos_right) + (area - size_right))

        # select min points as appropriate hand
        append_center(data, valid_contours[np.argmin(distances_l)], 1)
        append_center(data, valid_contours[np.argmin(distances_r)], 2)

        data = add_empty_row(data, frame)

    # start drawing from 15th frame on
    if (frame > 15):
        for fi in range(-15, -1):
            if (type(data[1, fi, 0]) != float):
                cX, cY = data[1, fi, 0]
                # commented for processing speed, uncomment for debug
                # cv2.circle(sourceImage, (cX, cY), 7, (255, 0, 0), 3)

                cX, cY = data[2, fi, 0]
                # commented for processing speed, uncomment for debug
                # cv2.circle(sourceImage, (cX, cY), 7, (0, 255, 0), 3)
            if (type(data[3, fi, 0]) != float):
                cX, cY = data[3, fi, 0]
                cv2.circle(sourceImage, (cX, cY), 7, (0, 0, 255), 3)

    # Display the source image
    cv2.imshow('Camera Output', sourceImage)

    # Check for user input to close program
    if frame % 100 == 0:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()
            break

    # increment round counter
    frame += 1

end = time.time()
print(end - start)

# Close window and camera after exiting the while loop
cap.release()
cv2.destroyAllWindows()

# save capture data (optionally skip annotation step and start here)
import pickle

with open(src + name + ".pickle", 'wb') as f:
    pickle.dump(data, f)


# create continous series
data_new = data


def make_continous(series):
    # threshold in frames:
    threshold = 26

    # split in single sequences
    last = -1

    smaller_series = []
    curr = []

    for i, e in enumerate(series):
        if (i == 0):
            last = e
            curr.append(e)
        else:
            if (last + 1 == e):
                last = e
                curr.append(e)
            elif (last + threshold > e):
                print("Threshold saved the day")
                last = e
                [curr.append(x) for x in range(last + 1, e + 1)]
            else:
                print("Series Ended, difference to next", e - last, "curr length", len(curr))
                smaller_series.append(curr)
                curr = []
                last = e

    smaller_series.append(curr)
    return smaller_series

# extract audio

from pydub import AudioSegment

sound = AudioSegment.from_mp3(src + name + ".mp3")

# len() and slicing are in milliseconds
ms_per_frame = 1000 / 25

data_new[3, 0] = 0

both_hands = [i for i, e in enumerate(data_new[3, :, 0]) if type(e) == tuple]
series = make_continous(both_hands)
for i, s in enumerate(series):
    if (len(s) < 13):
        continue
    start = (min(s) + offset) * ms_per_frame
    end = (max(s) + offset) * ms_per_frame
    hands_together = sound[start - 500: end + 500]
    hands_together.export(src + new_name + "_both_hands_" + str(i) + ".mp3", format="mp3")


# extract video
cap = cv2.VideoCapture(src + name + video_ending)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(w, h)

both_hands = [i for i, e in enumerate(data_new[3, :, 0]) if type(e) == tuple]
series = make_continous(both_hands)

for i, s in enumerate(series):
    if (len(s) < 13):
        continue
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_writer = cv2.VideoWriter(src + new_name + "_both_hands_" + str(i) + ".mp4", fourcc, 25, (w, h))
    start = min(s) - 13 + offset
    end = max(s) + 13 + offset
    cap.set(1, start)
    f = 0
    while (cap.isOpened() and f < (end - start)):
        # Grab video frame, decode it and return next video frame
        readSuccess, sourceImage = cap.read()
        # video recorder
        if (readSuccess):
            video_writer.write(sourceImage)
        else:
            print("READ FAIL")
        f += 1

    video_writer.release()

cap.release()
exit()

# define which video to cut
src = "./RG_2015_05_21/"
name = "2015-05-21-regierungserklaerung-merkel_LQ"
new_name = "RG_2015_05_21"
video_ending = ".mp4"

# print name to make multiprocessing easier
print(new_name)

# Required moduls
import cv2
import numpy as np
from pydub import AudioSegment

# prepare comparison frame chose
check = False

def click_and_crop(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    global check
    check = True


cv2.namedWindow('Cutting')

# Get pointer to video frames from primary device
cap = cv2.VideoCapture(src + name + video_ending)
cv2.setMouseCallback("Cutting", click_and_crop)

comparison_frame = None

while cap.isOpened():
    # Grab video frame, decode it and return next video frame
    readSuccess, sourceImage = cap.read()

    if not readSuccess:
        break

    # wait for user to click for referenceframe
    if (check):
        comparison_frame = sourceImage
        check = False
        break

    # Display the image
    cv2.imshow('Cutting', sourceImage)

    # Abort?
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()
        break

# clean up
cap.release()

# start video from the beginning
cap = cv2.VideoCapture(src + name + video_ending)
data = np.array([])  # frame

# frame nr
f = 0

while cap.isOpened():
    # Grab video frame, decode it and return next video frame
    readSuccess, sourceImage = cap.read()

    if not readSuccess:
        break

    # calculate differences
    differences = cv2.norm(sourceImage, comparison_frame, cv2.NORM_L1)

    # choose threshold... (depends on screen size and changes in light etc.)
    if differences < 20000000:
        data = np.append(data, [f])

    # Display the source image + difference
    cv2.putText(sourceImage, str(differences), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    cv2.imshow('Camera Output', sourceImage)

    # Check for user input to close program (only every 4 seconds since we want to have a quick processing)
    if f % 100 == 0:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()
            break

    f += 1
# clean up
cap.release()
cv2.destroyAllWindows()

# make continous
def make_continous(series):
    # threshold in frames:
    threshold = 26

    # split in single sequences
    last = -1

    smaller_series = []
    curr = []

    for i, e in enumerate(series):
        e = int(e)
        if (i == 0):
            last = int(e)
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
                print("Series Ended, difference to next", e - last)
                smaller_series.append(curr)
                curr = []
                last = e

    smaller_series.append(curr)
    return smaller_series

# read video one last time
cap = cv2.VideoCapture(src + name + video_ending)

# read width / height
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(w, h)

# cut video according to extracted series
series = make_continous(data)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video_writer = cv2.VideoWriter(src + new_name + ".mp4", fourcc, 25, (w, h))

for i, s in enumerate(series):
    if (len(s) == 0):
        continue
    start = min(s)
    end = max(s)
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

# also cut the audio so it stays in sync.

sound = AudioSegment.from_mp3(src + name + ".mp3")

# len() and slicing are in milliseconds
ms_per_frame = 1000 / 25

offset = 0
series = make_continous(data)

hands_together = []

for s in series:
    if (len(s) == 0):
        continue
    start = (min(s) + offset) * ms_per_frame
    end = (max(s) + offset) * ms_per_frame
    hands_together.append(sound[start - 500: end + 500])

starter = hands_together[0]
for x in hands_together:
    starter = starter + x

starter.export(src + new_name + ".mp3", format="mp3")

exit()
# imports
import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise


# global variables
# background
bg = None


# -------------------------------------------------------------------------------
# Function - To find the running average over the background
# running average helps to detect the background and later remove it
# -------------------------------------------------------------------------------
def run_avg(image, accumWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    # The function calculates the weighted sum of the input image src
    # and the accumulator dst so that dst becomes a running average of a frame sequence
    # cv2.accumulateWeighted(src, dst, alpha[, mask])
    # alpha - weight of the input image
    cv2.accumulateWeighted(image, bg, accumWeight)


# -------------------------------------------------------------------------------
# Function - To segment the region of hand in the image
# -------------------------------------------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    # Calculates the per-element absolute difference between two arrays or between an array and a scalar
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    # First argument is the source image
    # Second argument is the threshold value which is used to classify the pixel values.
    # Third argument is the maxVal which represents the value to be given if pixel value is more than
    # (sometimes less than) the threshold value
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    # first one is source image, second is contour retrieval mode, third is contour approximation method.
    # And it outputs the contours and hierarchy. contours is a Python list of all the contours in the image.
    # Each individual contour is array of (x,y) coordinates of boundary points of the object.
    (_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        # key for maximum function is contourArea
        segmented = max(cnts, key=cv2.contourArea)
        return thresholded, segmented


# -------------------------------------------------------------------------------
# Function - To count the number of fingers in the segmented hand region
# -------------------------------------------------------------------------------
def count(thresholded, segmented):
    # find the convex hull of the segmented hand region
    chull = cv2.convexHull(segmented)

    # find the most extreme points in the convex hull
    # top - min y coordinate
    # bottom - max y coordinate
    extreme_top = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    # left - min x coordinate
    # right - max x coordinate
    extreme_left = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right = tuple(chull[chull[:, :, 0].argmax()][0])

    # find the center of the palm as the average
    cX = (extreme_left[0] + extreme_right[0]) / 2
    cY = (extreme_top[1] + extreme_bottom[1]) / 2

    # find the maximum euclidean distance between the center of the palm
    # and the extreme points of the convex hull
    # Considering the rows of X (and Y=X) as vectors, compute the distance matrix between each pair of vectors
    # X - [(cX, cY)] - center points
    # Y - [el, er, et, eb] - points as input
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    # maximum distance from center of palm
    maximum_distance = distance[distance.argmax()]

    # We took 80% of maximum distance as the radius of circle of interest
    # and 60% of maximum distance as radius of inner circle of interest
    # calculate the radius of the circle with euclidean distance obtained
    radius = int(0.80 * maximum_distance)
    radius2 = int(0.60 * maximum_distance)

    # find the circumference of the circle
    circumference = (2 * np.pi * radius)
  
    # take out the circular region of interest which has
    # the palm and the fingers
    # initialize black images to be used in further drawing operations
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    circular_roi2 = np.zeros(thresholded.shape[:2], dtype="uint8")

    # draw the circular ROI
    # params -
    # img - image
    # center - center of circle
    # radius - radius of circle
    # color - color in pixel value
    # thickness - thickness of circle
    cv2.circle(circular_roi, (int(cX), int(cY)), radius, 255, 2)
    cv2.circle(circular_roi2, (int(cX), int(cY)), radius2, 255, 1)

    # cv2.circle(circular_roi, (int(cX),int(cY)), radius2, 255, 1)

    # take bit-wise AND between thresholded hand using the circular ROI as the mask
    # which gives the cuts obtained using mask on the thresholded hand image
    thresholded2 = thresholded

    # img1 - first image
    # img2 - second image
    # mask - area to be considered from the AND result
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
    circular_roi2 = cv2.bitwise_and(thresholded2, thresholded2, mask=circular_roi2)

    cv2.imshow("circular_roi", circular_roi)

    # retrieves the contours in the binary circular ROI
    # cnts, cnts2 - returned contours array
    # parameters - image, contour retrieval mode, contour approximation method
    # RETR_EXTERNAL - extreme outer contours
    # CHAIN_APPROX_NONE - stores all contour points instead of saving approx
    (_, cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (_, cnts2, _) = cv2.findContours(circular_roi2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # initialize the finger count
    count = 0
    count2 = 0

    cv2.circle(thresholded, (int(cX), int(cY)), int(radius), (50, 205, 50), 2)
    cv2.circle(thresholded, (int(cX), int(cY)), int(radius2), (50, 5, 50), 2)

    # cv2.imshow('thresholded',thresholded)

    # loop through the contours found
    # centers - saves all centroids of contours
    centers = []
    D = 0
    for c in cnts:
        # moments gives a dictionary of all moments calculated
        M = cv2.moments(c)
        # compute the bounding box of the contour
        # x, y - top left point coordinates
        # w, h - width, height of rectangle
        (x, y, w, h) = cv2.boundingRect(c)

        # increment the count of fingers only if -
        # 1. The contour region is not the wrist (bottom area)
        # 2. The number of points along the contour does not exceed
        # 25% of the circumference of the circular ROI
        # This creates the problem of the thumb getting out of the inner ROI circle but following in through to
        # the wrist and failing to get recognized
        if ((circumference * 0.16) > c.shape[0]) and ((cY + (cY * 0.25)) > (y + h)):
            count += 1

            if M['m00'] != 0:
                # centroid coordinates
                # centroid can be found from moments using m10/m00 and m01/m00
                cX1 = int(M['m10'] / M['m00'])
                cY1 = int(M['m01'] / M['m00'])
                centers.append([cX1, cY1])
            else:
                D = D + 1
        # Find the distance D between the two contours:
    # len(centers) - number of fingers
    if len(centers) == 2:
        dx = centers[0][0] - centers[1][0]
        dy = centers[0][1] - centers[1][1]
        D = np.sqrt(dx*dx+dy*dy)

    # finding number of fingers intersecting the inner ROI circle
    for c in cnts2:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # increment the count of fingers only if -
        # 1. The contour region is not the wrist (bottom area)
        # 2. The number of points along the contour does not exceed
        # 25% of the circumference of the circular ROI
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.16) > c.shape[0]):
            count2 += 1

    if count == 1 or count == 2 or count == 3:
        # detecting numbers greater than 5 in the circle
        # add 5 if inner circle is intersected and outer is not
        if count2 == (count+1):
            count = 5 + count

    # put text in image
    # img - image
    # text - text
    # org - bottom left corner of the text string in the image
    # fontFace - FONT_HERSHEY_SIMPLEX - normal size sans-serif font
    # thickness
    # color
    # lineType
    cv2.putText(thresholded, str(D), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # If distance between fingers is > 130, count it as 10
    if count == 2 and D > 130:
        count = 10
    # else if D > 80, count it as 8
    if count == 2 and D > 80:
        count = 11
    # if 3 fingers and at least one is not intersecting outer circle, count it as 9
    if count == 3 and count2 < 3:
        count = 9

    return count


# -------------------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------------------
if __name__ == "__main__":
    # initialize accumulated weight
    accumWeight = 0.5

    # get the reference to the web cam
    camera = cv2.VideoCapture(0)

    pix = 0
    flag = 0
    flag1 = 0
    paused = 0
    cntframe = 0
    char = ''
    stri = ''
    num = 0
    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0

    # calibration indicator
    calibrated = False

    # keep looping, until interrupted
    while True:
        # get the current frame
        (grabbed, frame) = camera.read()
        board = cv2.imread('images/blackboard.png')

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame vertically so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to gray scale and blur it
        # change pic from one color space to another
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # blur filter for image using Gaussian function
        # specify the width and height of the kernel which should be positive and odd.
        # We also should specify the standard deviation in the X and Y directions, sigmaX and sigmaY respectively.
        # If only sigmaX is specified, sigmaY is taken as equal to sigmaX.
        # If both are given as zeros, they are calculated from the kernel size.
        # Gaussian filtering is highly effective in removing Gaussian noise from the image.
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our weighted average model gets calibrated
        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
                print("[STATUS] please wait! calibrating...")
            elif num_frames == 29:
                print("[STATUS] please wait! successful...")
        else:
            # segment the hand region
            hand = segment(gray)
            cntframe = cntframe+1
            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                # clone - destination image
                # contours – All the input contours. Each contour is stored as a point vector.
                # contourIdx – Parameter indicating a contour to draw. If it is negative, all the contours are drawn.
                # color – Color of the contours.
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

                # count the number of fingers
                fingers = count(thresholded, segmented)
                char = ''
                if cntframe > 90:
                    cntframe = 0
                    # every 90 frames, record the frame in ROI and process it
                if cntframe == 90 and num % 2 == 0:
                    # put text in image
                    # img - image
                    # text - text
                    # org - bottom left corner of the text string in the image
                    # fontFace - FONT_HERSHEY_SIMPLEX - normal size sans-serif font
                    # thickness
                    # color
                    # lineType
                    cv2.putText(clone, "record", (70, 145), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cntframe = 0
                    if fingers == 10:
                        # if 10, change mode and set num to odd
                        num = num + 1
                    # if flag1 == 1, it means operator is chosen
                    # revert back to operand state
                    if fingers == 10 and flag1 == 1:
                        flag1 = 2
                    if fingers == 1:
                        char = '1'
                    elif fingers == 2:
                        char = '2'
                    elif fingers == 3:
                        char = '3'
                    elif fingers == 4:
                        char = '4'
                    elif fingers == 5:
                        char = '5'
                    elif fingers == 6:
                        char = '6'
                    elif fingers == 7:
                        char = '7'
                    elif fingers == 8:
                        char = '8'
                    elif fingers == 9:
                        char = '9'
                    elif fingers == 11:
                        char = '0'
                    stri = stri + char

                # record operator value
                if cntframe == 90 and num % 2 == 1:
                    cv2.putText(clone, "record", (70, 145), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cntframe = 0

                    if fingers == 1:
                        char = '+'
                    elif fingers == 2:
                        char = '-'
                    elif fingers == 3:
                        char = '*'
                    elif fingers == 4:
                        char = '/'
                    flag1 = 1
                    num = num + 1
                    stri = stri + char

                # if flag1 == 2, calculate value of expression
                if flag1 == 2:
                    flag = 0
                    for i, c in enumerate(stri):
                        if c == '+':
                            stri = str(int(stri[0:i]) + int(stri[i + 1:len(stri)]))
                            break
                        if c == '-':
                            stri = str(int(stri[0:i]) - int(stri[i + 1:len(stri)]))
                            break
                        if c == '*':
                            stri = str(int(stri[0:i]) * int(stri[i + 1:len(stri)]))
                            break
                        if c == '/':
                            stri = str(int(stri[0:i]) / int(stri[i + 1:len(stri)]))
                            break
                # display number of fingers being detected
                cv2.putText(clone, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # display expression on board
                cv2.putText(board, stri, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # show the thresholded image
                cv2.imshow("Thresholded", thresholded)
                cv2.imshow("board", board)

                if char == '=':
                    break

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

    while True:
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()

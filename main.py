import numpy as np
import cv2
import imutils
import os
from matplotlib import pyplot as plt


def vid_to_frames(filename):
    vidcap = cv2.VideoCapture(filename)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def findEdges(filename):
    image = cv2.imread(filename)
    image = imutils.resize(image, height=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    return screenCnt


def drawOutline(frameInput, coords):
    filename = frameInput.split(os.sep)[-1].split(".")[0]
    image = cv2.imread(frameInput)
    image = imutils.resize(image, height=500)
    cv2.drawContours(image, [coords], -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(framesOutput, filename + "_outline.jpg"), image)


framesInput = os.path.join("frames", "input")
framesOutput = os.path.join("frames", "output")
framesInput = [os.path.join(framesInput, frame) for frame in os.listdir(framesInput)]
template = r"template.png"
templateImage = cv2.imread(template)
firstFrame = cv2.imread(framesInput[0])


edges = findEdges(framesInput[0]) * firstFrame.shape[0] / 500
templateImageEdges = np.array([[0, 0], [1275, 0], [1275, 1650], [0, 1650]])

# calculate matrix H
h, status = cv2.findHomography(edges, templateImageEdges)

# provide a point you wish to map from image 1 to image 2
pointsToTransform = np.array([[940, 700], [640, 530]], dtype='float32')


# finally, get the mapping
# print("template image size:", templateImage.shape)
pointsOut = cv2.perspectiveTransform(np.array([pointsToTransform]), h)
pointsOut.resize(pointsToTransform.shape)
for p in pointsOut:
    x = int(p[0])
    y = int(p[1])
    templateImage[x-5:x+5, y-5:y+5] = [255, 0, 0]

cv2.imwrite('out.png', templateImage)

#print(templateImage[pointsOut.astype(int)[0,0,0], pointsOut.astype(int)[0,0,1]])
#cv2.imwrite(os.path.join(framesOutput, "test_outline.jpg"), templateImage)

#test = cv2.imread(os.path.join(framesOutput, "test_outline.jpg"))


# print(edges)
#print("STEP 2: Draw outlines")
# for frame in framesInput:
#    drawOutline(frame, edges)

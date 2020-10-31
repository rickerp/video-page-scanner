# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
from skimage.filters import threshold_local


def vid_to_frames(filename, write=False):
    vidcap = cv2.VideoCapture(filename)
    success, image = vidcap.read()
    count = 0
    images = []
    while success:
        if write:
            cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file
        else:
            images.append(image)
        success, image = vidcap.read()
        count += 1
    return images


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


"""
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
args = vars(ap.parse_args())
"""


def get_points(image):
    # Resize image for optimization purposes
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            return (approx * ratio).astype(int)


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file", default="video.mp4")
ap.add_argument("-t", "--template", help="template file", default="template.png")
ap.add_argument("-o", "--output", help="output video", default="output.mp4")
args = vars(ap.parse_args())

frames = vid_to_frames(args["video"])
template = cv2.imread(args["template"])
points = get_points(frames[0]).reshape(4, 2)
out_video = cv2.VideoWriter(args["output"], cv2.VideoWriter_fourcc(
    *'mp4v'), 30, (template.shape[1], template.shape[0]))

for f in range(len(frames)):
    # cv2.drawContours(frames[f], [paper_pts], -1, (0, 255, 0), 2)
    # cv2.imwrite(f"output/frame{f}-drawn.jpg", frames[f])
    new_points = get_points(frames[f])
    if (type(new_points) != type(None) and (abs(new_points - points).sum() < 10)):
        new_points.reshape(4, 2)
        points = new_points

    warped = four_point_transform(frames[f], points)
    # cv2.imwrite(f"output/frame{f}.jpg", cv2.resize(cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE), (1275, 1650)))
    out_video.write(cv2.resize(cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE),
                               (template.shape[1], template.shape[0])))

out_video.release()

#!/usr/bin/env python3
import cv2
import sys
import numpy as np
import imutils
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
    # order 4 points by top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def get_homography(points, dst):
    # Gets the homography transformation matrix from a set of 4 points to another 4 points

    rect = order_points(points)

    dst = np.array([[0, 0],
                    [dst[0] - 1, 0],
                    [dst[0] - 1, dst[1] - 1],
                    [0, dst[1] - 1]],
                   dtype="float32")

    return cv2.getPerspectiveTransform(rect, dst)


def get_points(image):
    # Resize image for optimization purposes
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    # convert the image to grayscale, blur it, and find edges in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # find the contours in the edged image, keeping only the largest ones
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            return (approx * ratio).astype(int)


if (len(sys.argv) < 3):
    print("Error: " + sys.argv[0] +
          " <video-path> <template-path> <optional-output-path>", file=sys.stderr)
    exit(1)

args = {"video": sys.argv[1], "template": sys.argv[2],
        "output": sys.argv[3] if len(sys.argv) > 3 else 'nasty.mp4'}

frames = vid_to_frames(args["video"])
template = cv2.imread(args["template"])
out_video = cv2.VideoWriter(args["output"], cv2.VideoWriter_fourcc(
    *'mp4v'), 30, (template.shape[1], template.shape[0]))

points = get_points(frames[0]).reshape(4, 2)
homography = get_homography(points, template.shape)

for f in range(len(frames)):
    # cv2.drawContours(frames[f], [paper_pts], -1, (0, 255, 0), 2)
    # cv2.imwrite(f"output/frame{f}-drawn.jpg", frames[f])

    warped = cv2.warpPerspective(frames[f], homography, template.shape[:2])
    # cv2.imwrite(f"output/frame{f}.jpg", cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE), (1275, 1650))
    out_video.write(cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE))

out_video.release()

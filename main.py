#!/usr/bin/python3

import cv2
import numpy as np
import imutils
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file", default="video.mp4")
ap.add_argument("-t", "--template", help="template png file with the wanted output dimensions")
ap.add_argument("-o", "--output", help="output video", default="output.mp4")
args = vars(ap.parse_args())


def findEdges(image):
    ratio = image.shape[0] / 500.0
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
            return (approx * ratio).astype(int)


def sortPoints(pts):
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


def part1(video, template, output):
    # split video input frame by frame
    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read()
    count = 0
    inputFrames = []
    while success:
        inputFrames.append(image)
        success, image = vidcap.read()
        count += 1

    # open the template image
    output_shape = cv2.imread(template).shape if template else (1650, 1275, 3)

    # start the video writer for the video output
    out_video = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(
        *'mp4v'), 30, (output_shape[1], output_shape[0]))

    # since the a4 page doesn't change position we can simply grab the
    # edges position from the first video frame
    a4Edges = sortPoints(findEdges(inputFrames[0]).reshape(4, 2))

    # add a small margin to compensate for the not so perfect edge detection
    # *this margin was checked manually*
    a4Edges = np.array([a4Edges[0] + [4, -8],
                        a4Edges[1] + [9, -2],
                        a4Edges[2] + [4, 6],
                        a4Edges[3] + [0, 5]],
                       dtype="float32")

    outputEdges = np.array([[0, 0],
                            [output_shape[0] - 1, 0],
                            [output_shape[0] - 1, output_shape[1] - 1],
                            [0, output_shape[1] - 1]],
                           dtype="float32")

    # get the homography matrix using both images edges
    homography = cv2.getPerspectiveTransform(a4Edges, outputEdges)

    # apply the homography transformation to every frame from the input vid
    for frame in inputFrames:
        transformed = cv2.warpPerspective(frame, homography, output_shape[:2])
        out_video.write(cv2.rotate(transformed, cv2.ROTATE_90_CLOCKWISE))

    out_video.release()


part1(args["video"], args["template"], args["output"])

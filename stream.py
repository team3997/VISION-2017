import argparse
import time
from networktables import NetworkTables
import imutils
import logging
import numpy as np
import sys
import cv2

#NETWORK TABLES
robot_ip = "10.39.97.10"
logging.basicConfig(level=logging.DEBUG)
NetworkTables.initialize(server=robot_ip)
dashboard = NetworkTables.getTable("SmartDashboard")

#FILTER and IMAGE SETTINGS
areaFilter = (0.005)
quality = 0.2

#HSV FILTER
lower_green = np.array([39,0,234]) #H,S,V
upper_green = np.array([180, 140, 255]) #H,S,V

#parse args
ap = argparse.ArgumentParser("Team 3997's vision program for 2017 FRC game. runs on rPi")
group = ap.add_mutually_exclusive_group()
group.add_argument("-i", "--image", nargs=1, required=False,
        help="path to the input image")
group.add_argument("-c", "--webcam", nargs=1, type=int, required=False,
        help="webcam number source to use")
args = ap.parse_args()

count = 0
forcount = 0
i = 0

def main():
    global cam
    global image

    if args.webcam is not None:
        cam = cv2.VideoCapture(0)
        cam.set(CV_CAP_PROP_EXPOSURE, 1)
	cam.read()
    elif args.image is not None:
        image = cv2.imread(args.image[0])
        show_webcam()
    else:
        print("expected image or webcam arguement. use --help for more info")
        exit(0)

    main_count = 0
    while(True):
        dashboard.putNumber('piCount:', time.clock())
        main_count += 1
        if is_processing():
            show_webcam()
        else:
            time.sleep(0.3)
        if cv2.waitKey(1) == ord('q'):
            break  # 'q' to quit


def is_processing():
    img_proc = False
    try:
        print('VISION_isProcessing:', dashboard.getBoolean('VISION_isProcessing', False))
        img_proc = dashboard.getBoolean('VISION_isProcessing', False)
    except:
        print('VISION_isProcessing: False')
    return img_proc

def show_webcam():
    global count
    global forcount
    global quality
    global i
    global cam
    global image

    if args.webcam is not None:
        ret_val, image = cam.read()

    #try:
    #    print('DEBUG_FPGATimestamp:', dashboard.getNumber('DEBUG_FPGATimestamp'))
    #except:
    #    print('DEBUG_FPGATimestamp: N/A')


    imgHeight, imgWidth, channels = image.shape

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)#convert image to hsv

    mask = cv2.inRange(hsv, lower_green, upper_green)#create mask for hsv filter
    res = cv2.bitwise_and(image, image, mask=mask)#apply hsv filter to the image

    backtocolor = cv2.cvtColor(res, cv2.COLOR_HSV2RGB); #convert to greyscale
    gray = cv2.cvtColor(backtocolor, cv2.COLOR_RGB2GRAY); #convert to greyscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) #gaussian blur to smooth edges
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1] #create binary image

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    cX = 0.0
    biggest_contour = 0
    next_biggest_contour = 0

    # loop over the contours
    for c in cnts:

        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            break

        # limit area
        if cv2.contourArea(c) / (imgHeight * imgWidth) > areaFilter:
            # draw the contour and center of the shape on the image
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(image, "center", (cX - 20, cY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            #UDPudp.sendto("cX: %s, cY, %s" % (str(cX), str(cY)), (UDP_IP, UDP_PORT))
        else:
            cX = 0
            cY = 0

        forcount = forcount + 1
        
        currentContourArea = cv2.contourArea(c)
        if currentContourArea > biggest_contour:
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
            if biggest_contour > next_biggest_contour:
                next_biggest_contour = biggest_contour
            	biggest_contour = currentContourArea
            	dashboard.putNumber('cX', cX)
            elif currentContourArea > next_biggest_contour:
            	next_biggest_contour = currentContourArea
	   	dashboard.putNumber("cX_2", cX)
        #if forcount < 10:
        #    cv2.imwrite( "./forimg" + str(forcount) + ".jpg", thresh);
        #    cv2.imwrite( "./forimg" + str(forcount) + "binary" + ".jpg", image);


    #show the image
    #cv2.imshow('Webcam',image)
    #cv2.imshow('Filtered',thresh)

    small = cv2.resize(image, (0,0), fx=quality, fy=quality)
    smallbinary = cv2.resize(thresh, (0,0), fx=quality, fy=quality)
    if count % 2 == 0:
        cv2.imwrite("/home/pi/mjpg/out.jpg", small);
    else:
        cv2.imwrite("/home/pi/mjpg/out.jpg", smallbinary);
    #cv2.imwrite("/home/pi/mjpg/out.jpg", smallbinary);

    count = count + 1
    #if count < 10:
        #cv2.imwrite( "./img" + str(count) + ".jpg", thresh);
        #cv2.imwrite( "./img" + str(count) + "binary" + ".jpg", image);


if __name__ == '__main__':
    main()


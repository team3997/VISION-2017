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
areaFilter = (0.1) #area of contour must be at least 'areaFilter' percent of the image
quality = 0.5 #quality of image sent to the smartdashboard

#HSV FILTER
lower_green = np.array([50,195,0]) #H,S,V
upper_green = np.array([79, 255, 112]) #H,S,V

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
    global lower_green
    global upper_green

    if args.webcam is not None:
        cam = cv2.VideoCapture(args.webcam[0])
	cam.read()
        #cam.set(cv2.cv.CV_CAP_PROP_EXPOSURE, -100)
    elif args.image is not None:
        image = cv2.imread(args.image[0])
        show_webcam()
    else:
        print("expected image or webcam arguement. use --help for more info")
        exit(0)

    main_count = 0
    while(True):
        H_LOW = dashboard.getNumber("H_LOW", 0);
        H_HIGH = dashboard.getNumber("H_HIGH", 0);
        S_LOW = dashboard.getNumber("S_LOW", 0);
        S_HIGH = dashboard.getNumber("S_HIGH", 0);
        V_LOW = dashboard.getNumber("V_LOW", 0);
        V_HIGH = dashboard.getNumber("V_HIGH", 0);

        dashboard.putNumber('piCount:', time.clock())
        main_count += 1
        #lower_green = np.array([H_LOW,S_LOW,V_LOW]) #H,S,V
        #upper_green = np.array([H_HIGH, S_HIGH, V_HIGH]) #H,S,V

        if is_processing():
            show_webcam()
        else:
            time.sleep(0.3)
        if args.image is not None:
            while(True):
                if cv2.waitKey(1) == ord('q'):
                    exit(0) # 'q' to quit
        elif cv2.waitKey(1) == ord('q'):
            exit(0) # 'q' to quit


def is_processing():
    img_proc = False
    try:
        #print('VISION_isProcessing:', dashboard.getBoolean('VISION_isProcessing', False))
        img_proc = dashboard.getBoolean('VISION_isProcessing', False)
    except:
        #print('VISION_isProcessing: False')
        print("except reached when getting dashboard");
    return img_proc
    #return True

def show_webcam():
    global count
    global forcount
    global quality
    global i
    global cam
    global image

    if args.webcam is not None:
        ret_val, image = cam.read()
    elif args.image is not None:
        image = cv2.imread(args.image[0])

    #try:
    #    print('DEBUG_FPGATimestamp:', dashboard.getNumber('DEBUG_FPGATimestamp'))
    #except:
    #    print('DEBUG_FPGATimestamp: N/A')
    #image = cv2.transpose(image)
    #image = cv2.flip(image, flipCode=0)

    imgHeight, imgWidth, channels = image.shape

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)#convert image to hsv

    mask = cv2.inRange(hsv, lower_green, upper_green)#create mask for hsv filter
    #res = cv2.bitwise_and(image, image, mask=mask)#apply hsv filter to the image
    #cv2.imshow("res", res)
    #backtocolor = cv2.cvtColor(res, cv2.COLOR_HSV2RGB); #convert to greyscale
    #gray = cv2.cvtColor(backtocolor, cv2.COLOR_RGB2GRAY); #convert to greyscale
    blurred = cv2.GaussianBlur(mask, (5, 5), 0) #gaussian blur to smooth edges
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1] #create binary image

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    cX = 0.0
    cY = 0.0
    biggest_contour = 0
    next_biggest_contour = 0
    center_biggest = 0;
    center_next_biggest = 0

    c_amnt = 0
    # loop over the contours
    for c in cnts:

        c_amnt += 1
        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            print "divide by zero break"
            continue

        #while True:
        #    cv2.imshow('Webcam',image)
        #    cv2.imshow('Filtered',thresh)
        #    if cv2.waitKey(1) == ord('f'):
        #        break # 'q' to quit

        currentContourArea = cv2.contourArea(c)
        print("currentContourArea: %s" % currentContourArea)

        # limit area
        #for c in cnts:
        if (cv2.contourArea(c) / (imgHeight * imgWidth)) > (areaFilter / 100.0):
        #if True:
            # draw the contouresr and center of the shape on the image
            print ("DRAWING")
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.drawContours(image, [c], -1, (0, 0, 255), 2)
            cv2.circle(image, (cX, cY), 1, (0, 255, 255), -1)
            cv2.putText(image, ("%s:" % c_amnt), (cX - 15, cY - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)
            cv2.putText(image, ("%s;" % cX), (cX - 15, cY - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            #cv2.putText(image, ("%s;" % cv2.contourArea(c)), (cX - 15, cY - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if currentContourArea > biggest_contour:
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                if biggest_contour > next_biggest_contour:
                    next_biggest_contour = biggest_contour
                biggest_contour = currentContourArea
                center_biggest = cX
                #print("center_biggest: %s" % center_biggest)
                #dashboard.putNumber('center_biggest', center_biggest)
                #print("biggest: %s" % biggest_contour)
            elif currentContourArea > next_biggest_contour:
                next_biggest_contour = currentContourArea
                center_next_biggest = cX
                #print("center_next_biggest: %s" % center_next_biggest)
                #dashboard.putNumber("center_next_biggest", center_next_biggest)
                #print("nextbiggest: %s" % next_biggest_contour)
        else:
            cX = 0
            cY = 0

        forcount = forcount + 1

        

        #if forcount < 10:
        #    cv2.imwrite( "./forimg" + str(forcount) + ".jpg", thresh);
        #    cv2.imwrite( "./forimg" + str(forcount) + "binary" + ".jpg", image);
        print("loop amount %d " % c_amnt)


    if center_biggest <= center_next_biggest: #determine left and right contours
        dashboard.putNumber("VISION_leftContour", center_biggest)
        dashboard.putNumber("VISION_rightContour", center_next_biggest)
    else:
        dashboard.putNumber("VISION_leftContour", center_next_biggest)
        dashboard.putNumber("VISION_rightContour", center_biggest)
        dashboard.putNumber("center_next_biggest", center_next_biggest)
        cv2.putText(image, ("right"), (center_biggest - 15, 300 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)
        cv2.putText(image, ("left"), (int(center_next_biggest) - 15, 300 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)
    print("NEW_center_biggest: %s" % center_biggest)
    print("NEW_center_next_biggest: %s" % center_next_biggest)
    print("NEW_biggest_contour: %s" % biggest_contour)
    print("NEW_next_biggest_contour: %s" % next_biggest_contour)

    small = cv2.resize(image, (0,0), fx=quality, fy=quality)
    smallbinary = cv2.resize(thresh, (0,0), fx=quality, fy=quality)
    if count % 2 == 0:
        cv2.imwrite("/home/pi/mjpg/out.jpg", small);
    else:
        cv2.imwrite("/home/pi/mjpg/out.jpg", smallbinary);
    #cv2.imwrite("/home/pi/mjpg/out.jpg", smallbinary);

    #show the image
    #cv2.imshow('Filtered',thresh)
    #cv2.imshow('Webcam',image)
    #show low quality image
    cv2.imshow('Filtered',smallbinary)
    cv2.imshow('Webcam',small)

    print("------------AMOUNT: %s " % c_amnt)

    count = count + 1
    #if count < 10:
    #    cv2.imwrite( "./img" + str(count) + ".jpg", thresh);
    #    cv2.imwrite( "./img" + str(count) + "binary" + ".jpg", image);


if __name__ == '__main__':
    main()


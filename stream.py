import argparse
import socket
import imutils
import numpy as np
import cv2

#HSV FILTER
lower_green = np.array([39,0,234]) #H,S,V
upper_green = np.array([180, 140, 255]) #H,S,V

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
MESSAGE = "Hello, World!"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP

areaFilter = (0.01)

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, image = cam.read()
        #if mirror: 
        #    img = cv2.flip(img, 1)
        cv2.imshow('my webcam', image)

        height, width, channels = image.shape

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)#convert image to hsv

        mask = cv2.inRange(hsv, lower_green, upper_green)#create mask for hsv filter
        res = cv2.bitwise_and(image, image, mask=mask)#apply hsv filter to the image

        backtocolor = cv2.cvtColor(res, cv2.COLOR_HSV2RGB); #convert to greyscale
        gray = cv2.cvtColor(backtocolor, cv2.COLOR_RGB2GRAY); #convert to greyscale
        blurred = cv2.GaussianBlur(gray, (5, 5), 0) #gaussian blur to smooth edges
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1] #create binary image

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

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
            if cv2.contourArea(c) / (height * width) > areaFilter:
                # draw the contour and center of the shape on the image
                cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
                cv2.putText(image, "center", (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # send values over UDP
                # send cX, cY
                sock.sendto("cX: %s, cY, %s" % (str(cX), str(cY)), (UDP_IP, UDP_PORT))

            # show the image
            cv2.imshow('image',image)
            cv2.imshow('thresh',thresh)
            if cv2.waitKey(1) == 27: 
                break  # esc to quit
    cv2.destroyAllWindows()

def main():
    show_webcam(mirror=True)

if __name__ == '__main__':
    main()

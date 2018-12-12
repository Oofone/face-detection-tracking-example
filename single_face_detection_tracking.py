import numpy as np
import cv2
import dlib

tracker = dlib.correlation_tracker()
capture = cv2.VideoCapture(0)
trackingFace = 0

faceCascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml')

rectangleColor = (255, 0, 0)

while True:
    ret, baseImage = capture.read()
    gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
    resultImage = baseImage

    key = cv2.waitKey(2)
    if key == ord("q"):
        cv2.destroyAllWindows()
        exit()

    if not trackingFace:

        #For the face detection, we need to make use of a gray
        #colored image so we will convert the baseImage to a
        #gray-based image
        gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
        #Now use the haar cascade detector to find all faces
        #in the image
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        #In the console we can show that only now we are
        #using the detector for a face
        print("Using the cascade detector to detect face")


        #For now, we are only interested in the 'largest'
        #face, and we determine this based on the largest
        #area of the found rectangle. First initialize the
        #required variables to 0
        maxArea = 0
        x = 0
        y = 0
        w = 0
        h = 0


        #Loop over all faces and check if the area for this
        #face is the largest so far
        #We need to convert it to int here because of the
        #requirement of the dlib tracker. If we omit the cast to
        #int here, you will get cast errors since the detector
        #returns numpy.int32 and the tracker requires an int
        for (_x,_y,_w,_h) in faces:
            if  _w*_h > maxArea:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)
                maxArea = w*h

        #If one or more faces are found, initialize the tracker
        #on the largest face in the picture
        if maxArea > 0 :

            #Initialize the tracker
            tracker.start_track(baseImage,
                                dlib.rectangle( x-10,
                                                y-20,
                                                x+w+10,
                                                y+h+20))

            #Set the indicator variable such that we know the
            #tracker is tracking a region in the image
            trackingFace = 1

    #Check if the tracker is actively tracking a region in the image
    if trackingFace:

        #Update the tracker and request information about the
        #quality of the tracking update
        trackingQuality = tracker.update( baseImage )



        #If the tracking quality is good enough, determine the
        #updated position of the tracked region and draw the
        #rectangle
        if trackingQuality >= 8.75:
            tracked_position =  tracker.get_position()

            t_x = int(tracked_position.left())
            t_y = int(tracked_position.top())
            t_w = int(tracked_position.width())
            t_h = int(tracked_position.height())
            cv2.rectangle(resultImage, (t_x, t_y),
                                        (t_x + t_w , t_y + t_h),
                                        rectangleColor ,2)

        else:
            #If the quality of the tracking update is not
            #sufficient (e.g. the tracked region moved out of the
            #screen) we stop the tracking of the face and in the
            #next loop we will find the largest face in the image
            #again
            trackingFace = 0

    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #
    # for (x,y,w,h) in faces:
    #     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #     roi_gray = gray[y:y+h, x:x+w]
    #     roi_color = img[y:y+h, x:x+w]
    #     eyes = eye_cascade.detectMultiScale(roi_gray)
    #     for (ex,ey,ew,eh) in eyes:
    #         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    #

    cv2.imshow('img',resultImage)

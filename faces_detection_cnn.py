import face_recognition
import cv2
import time
# This is a demo of blurring faces in video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture('/home/thongpb/works/face_recognition/data/video/output_thong.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('out_faces.avi',fourcc, 5.0, (1280,720))
# Initialize some variables
face_locations = []

while True:
    t1 = time.time()
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if(ret == False):
        break

    # Resize frame of video to 1/4 size for faster face detection processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(small_frame, model="cnn")

    # Display the results
    for top, right, bottom, left in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        cv2.rectangle(frame, (left, top), (right, bottom), (255, 120, 120), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)
    # out.write(frame)
    t2 = time.time()
    print ("FPS ", 1/(t2-t1))
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
# out.release()
video_capture.release()
cv2.destroyAllWindows()


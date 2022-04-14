import cv2
import face_recognition
import numpy as np
from pympler import asizeof

def loadVideoAndTarget(*, fpath, tpath, tname, stride, outpath):
    video_capture = cv2.VideoCapture(fpath)
    i = 0
    frames = []
    print('Start loading and resizing each frame')
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        i += 1
        # print(f'Current frame {i}')
        if (i-1) % stride != 0:
            continue
        # Resize for efficiency
        smaller_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        print(asizeof.asizeof(frame))
        print(asizeof.asizeof(smaller_frame))
        frames.append(smaller_frame[:,:,::-1])
    print(f'Total frams: {i}')

    face_image = face_recognition.load_image_file(tpath)
    given_face_encoding = face_recognition.face_encodings(face_image)[0]
    output = {'frame_count': i, 'frames': frames, 'target_encodings': given_face_encoding, 'fpath': fpath, 'target_name': tname, 'stride': stride, 'outpath': outpath}
    
    print('Sharding complete')
    return output

def getLocationsAndEncodings(*, frames):
    print("Calculating Locations and Encodings of frames")
    locationsPFrame = []
    encodingsPFrame = []
    fi = 0
    for frame in frames:
        fi += 1
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        locationsPFrame.append(face_locations)
        encodingsPFrame.append(face_encodings)
    return {'face_locationsPF': locationsPFrame, 'face_encodingsPF': encodingsPFrame}

def tagTargetFrames(*, face_encodingsPF, target_encodings, target_name):
    print("Tagging target faces on the frames")
    face_namesPFrame = []
    for face_encodings in face_encodingsPF:
        face_names = []
        for encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces([target_encodings], encoding)
            name = 'unknown'
            if True in matches:
                name = target_name
            face_names.append(name)
        face_namesPFrame.append(face_names)
    return {'face_namesPF': face_namesPFrame}

def boxTargetFace(*, face_locationsPF, face_namesPF, target_name, stride, fpath, outpath):
    video_capture = cv2.VideoCapture(fpath) 
    wdtih = int(video_capture.get(3))
    height = int(video_capture.get(4))
    frameSize = (wdtih, height)
    result = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'MJPG'), 30, frameSize)
    i = 0
    print("Adding to the video")
    while True:
        ret, frame = video_capture.read() 
        if not ret:
            break
        i += 1
        if (i-1) % stride == 0:
            fid = (i-1) // stride
            if fid < len(face_locationsPF):
                face_locations = face_locationsPF[fid]
                face_names = face_namesPF[fid]
                # print(face_locations, face_names)
                for (top, right, bottom, left), cur_name in zip(face_locations, face_names):
                    if cur_name == target_name:
                        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        face_region = frame[top:bottom, left:right]
                        # Blur the Face with Gaussian Blur of Kernel Size 51*51
                        blur = cv2.GaussianBlur(face_region, (51, 51), 0)
                        frame[top:bottom, left:right] = blur
                        # # Draw a box around the face
                        # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                        # # Draw a label with a name below the face
                        # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        # font = cv2.FONT_HERSHEY_DUPLEX
                        # cv2.putText(frame, target_name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit!
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break
        result.write(frame)
    video_capture.release()
    result.release()
    cv2.destroyAllWindows()
    return {'status': 'OK'}
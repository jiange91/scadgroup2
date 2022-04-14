#@ type: compute
#@ corunning:
#@  mem:
#@     trans: mem
#@     type: mem
#@ parents:
#@   - None
#@ dependents:
#@   - getLocationAndEncoding

from disagg import *
from obj_pool import *
import base64
import copy
import math
from typing import *


def unstackedOutDict(outDict: Dict, targets: Dict):
    if not targets:
        return input
    output = {}
    for name, offset in targets.items():
        l = outDict[name]
        for i in range(len(l)):
            output[f'{name}-idx{i + offset}'] = l[i]
        outDict[name] = 'List', len(l)
    return {**outDict, **output}


def unstackedUploads(outDict: Dict, uploads: Dict, targets: Dict):
    if not targets:
        return uploads
    newUploads = {}
    for nextNode, objList in uploads.items():
        newUploads[nextNode] = []
        for objName, toL, toR in objList:
            if objName in targets:
                offset = targets[objName]
                l = len(outDict[objName])
                newUploads[nextNode] += [(f'{objName}-idx{offset + i}', toL,
                    toR) for i in range(l)]
            newUploads[nextNode].append((objName, toL, toR))
    return newUploads


import cv2
import face_recognition
import numpy as np


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
        if (i - 1) % stride != 0:
            continue
        smaller_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        frames.append(smaller_frame[:, :, ::-1])
    print(f'Total frams: {i}')
    face_image = face_recognition.load_image_file(tpath)
    given_face_encoding = face_recognition.face_encodings(face_image)[0]
    output = {'frame_count': i, 'frames': frames, 'target_encodings':
        given_face_encoding, 'fpath': fpath, 'target_name': tname, 'stride':
        stride, 'outpath': outpath}
    print('Sharding complete')
    return output


def getLocationsAndEncodings(*, frames):
    print('Calculating Locations and Encodings of frames')
    locationsPFrame = []
    encodingsPFrame = []
    fi = 0
    for frame in frames:
        fi += 1
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        locationsPFrame.append(face_locations)
        encodingsPFrame.append(face_encodings)
    return {'face_locationsPF': locationsPFrame, 'face_encodingsPF':
        encodingsPFrame}


def tagTargetFrames(*, face_encodingsPF, target_encodings, target_name):
    print('Tagging target faces on the frames')
    face_namesPFrame = []
    for face_encodings in face_encodingsPF:
        face_names = []
        for encoding in face_encodings:
            matches = face_recognition.compare_faces([target_encodings],
                encoding)
            name = 'unknown'
            if True in matches:
                name = target_name
            face_names.append(name)
        face_namesPFrame.append(face_names)
    return {'face_namesPF': face_namesPFrame}


def boxTargetFace(*, face_locationsPF, face_namesPF, target_name, stride,
    fpath, outpath):
    video_capture = cv2.VideoCapture(fpath)
    wdtih = int(video_capture.get(3))
    height = int(video_capture.get(4))
    frameSize = wdtih, height
    result = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'MJPG'), 30,
        frameSize)
    i = 0
    print('Adding to the video')
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        i += 1
        if (i - 1) % stride == 0:
            fid = (i - 1) // stride
            if fid < len(face_locationsPF):
                face_locations = face_locationsPF[fid]
                face_names = face_namesPF[fid]
                for (top, right, bottom, left), cur_name in zip(face_locations,
                    face_names):
                    if cur_name == target_name:
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        face_region = frame[top:bottom, left:right]
                        blur = cv2.GaussianBlur(face_region, (51, 51), 0)
                        frame[top:bottom, left:right] = blur
        result.write(frame)
    video_capture.release()
    result.release()
    cv2.destroyAllWindows()
    return {'status': 'OK'}


def main(params, action):
    objPool = ObjPool(name='mem', memSize=1073741824, pageSize=16384)
    objPool.registerTrans(actionLib=action)
    localPool = {'fpath':
        '/users/Zijian/scad/scadPriv/runtime/scripts/faceTag/interview-short-short.mp4'
        , 'tpath':
        '/users/Zijian/scad/scadPriv/runtime/scripts/faceTag/Gordon-Ramsay.jpeg'
        , 'tname': 'Gordon Ramsay', 'stride': 1, 'outpath':
        'faceTag/result.avi'}
    uploads = {'getLocationAndEncoding': [('frames', False, True), (
        'target_encodings', False, True), ('fpath', False, True), (
        'target_name', False, True), ('stride', False, True), ('outpath', 
        False, True)]}
    loadVTPullR = {}
    loadVTBeC = set()
    for name in loadVTPullR:
        localPool[name] = objPool.materialize(name)
    loadVTInMap = {'fpath', 'tname', 'outpath', 'stride', 'tpath'}
    loadVTInDict = {}
    for inName in loadVTInMap:
        if inName in loadVTBeC:
            loadVTInDict[inName] = copy.deepcopy(localPool[inName])
        else:
            loadVTInDict[inName] = localPool[inName]
    loadVTOutDict = loadVideoAndTarget(**loadVTInDict)
    loadVTUSTKTargets = {'frames': 0}
    uploads = unstackedUploads(loadVTOutDict, uploads, loadVTUSTKTargets)
    loadVTOutDict = unstackedOutDict(loadVTOutDict, loadVTUSTKTargets)
    localPool.update(loadVTOutDict)
    selectedUploads = list(uploads.values())[0]
    nextLocal = set()
    for vname, isLocal, isRemote in selectedUploads:
        if isLocal:
            objPool.upload_local(localPool[vname], vname)
            nextLocal.add(vname)
        if isRemote:
            objPool.upload_remote(localPool[vname], vname)
    objPool.filterLocal(nextLocal)
    clearRemote = set()
    if clearRemote:
        for vname in clearRemote:
            objPool.deleteRemote(vname)
    context_dict = {}
    context_dict['objPool'] = objPool
    context_dict_in_byte = serial_context.dumps(context_dict)
    return {'meta': base64.b64encode(context_dict_in_byte).decode('ascii')}

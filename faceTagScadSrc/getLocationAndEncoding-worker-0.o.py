#@ type: compute
#@ corunning:
#@  getLocationAndEncoding-worker-0-mem:
#@     trans: getLocationAndEncoding-worker-0-mem
#@     type: mem
#@  mem:
#@     trans: mem
#@     type: mem
#@ parents:
#@   - getLocationAndEncoding
#@ dependents:
#@   - getLocationAndEncoding-exit

from disagg import *
from obj_pool import *
import base64
import copy
import math
from typing import *


def stackedPullMap(inMap: Set, targets: Dict, objPool):
    if not targets:
        return inMap, targets
    else:
        stkMap = set()
        for name, (start, end, step) in targets.items():
            start = start or 0
            step = step or 1
            if not end:
                empiricalEnd = start
                while True:
                    curUstk = f'{name}-idx{empiricalEnd}'
                    if objPool.contains(curUstk):
                        stkMap.add(curUstk)
                    else:
                        break
                    empiricalEnd += step
                targets[name] = start, empiricalEnd, step
            else:
                for i in range(start, end, step):
                    curUstk = f'{name}-idx{i}'
                    stkMap.add(curUstk)
    return inMap.union(stkMap), targets


def stackedInPool(inPool: Dict, targets: Dict):
    if not targets:
        return inPool, None
    targetBackUp = {k: inPool[k] for k in targets.keys()}
    output: Dict[str, List] = {}
    for name, (start, end, step) in targets.items():
        output[name] = [inPool[f'{name}-idx{i}'] for i in range(start, end,
            step)]
    return {**inPool, **output}, targetBackUp


def restoreFromSTK(localPool: Dict, backUp: Dict):
    if backUp:
        localPool.update(backUp)
    return localPool


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
    context_dict_in_b64 = params['getLocationAndEncoding'][0]['meta']
    context_dict_in_byte = base64.b64decode(context_dict_in_b64)
    context_dict = serial_context.loads(context_dict_in_byte)
    objPool: ObjPool = context_dict['objPool']
    objPool.registerTrans(actionLib=action)
    localPool = {}
    selfPool = ObjPool(name='getLocationAndEncoding-worker-0-mem', memSize=
        536870912, pageSize=16384)
    selfPool.registerTrans(actionLib=action)
    uploads = {'getLocationAndEncoding-exit': [('face_encodingsPF', False, 
        True), ('face_locationsPF', False, True)]}
    pullR = {'frames'}
    workloads = context_dict['workloads']
    stkTargets = context_dict['getLocationAndEncoding-worker-0']
    pullR, stkTargets = stackedPullMap(pullR, stkTargets, objPool)
    for name in pullR:
        localPool[name] = objPool.materialize(name)
    localPool, restore = stackedInPool(localPool, stkTargets)
    inMap = {'frames'}
    inDict = {}
    for inName in inMap:
        inDict[inName] = localPool[inName]
    outDict = getLocationsAndEncodings(**inDict)
    localPool.update(outDict)
    for vname, obj in outDict.items():
        selfPool.upload_remote(obj, vname)
    context_dict = {}
    context_dict['getLocationAndEncoding-worker-0-pool'] = selfPool
    context_dict_in_byte = serial_context.dumps(context_dict)
    return {'meta': base64.b64encode(context_dict_in_byte).decode('ascii')}

from faceTag.lambdas import *
from StateMachine.primitives import *
from StateMachine.console import *
import inspect

tagApp = Application('faceTag')

loadVT = Lambda('loadVT')
loadVT.materialize_worker(loadVideoAndTarget, {'fpath', 'tpath', 'tname', 'stride', 'outpath'}, {'frame_count', 'frames', 'target_encodings', 'target_name', 'fpath', 'stride', 'outpath'})
frame_unstack = Unstacker(name='frameUnstack', targets={'frames': 0})
loadVT.registUnstacker(frame_unstack)

paralGetLE = Paral('getLocationAndEncoding')
paralGetLE.registWorker({'frames'}, {'face_locationsPF', 'face_encodingsPF'}, getLocationsAndEncodings, {'frames'}, 10, {'face_locationsPF': True, 'face_encodingsPF': True})

paralTagName = Paral('tagNameOnFrames')
paralTagName.registWorker({'face_encodingsPF', 'target_encodings', 'target_name'}, {'face_namesPF'}, tagTargetFrames, {'face_encodingsPF'}, 2, {'face_namesPF': True})

boxer = Lambda('BoxTargetName')
boxer.materialize_worker(boxTargetFace, {'face_locationsPF', 'face_namesPF', 'target_name', 'stride', 'fpath', 'outpath'}, {'status'})
boxStacker = Stacker('boxStacker', {'face_locationsPF': [0,None,None], 'face_namesPF': [0,None,None]})
boxer.registStacker(boxStacker)

tagApp.setNext(loadVT)
loadVT.setNext(paralGetLE)
paralGetLE.setNext(paralTagName)
paralTagName.setNext(boxer)

inputs = Inputs()
inputs.addInput({'fpath': 'faceTag/interview-short-short.mp4', 'tpath': 'faceTag/Gordon-Ramsay.jpeg', 'tname': 'Gordon Ramsay', 'stride':1, 'outpath': 'faceTag/result.avi'})

sm = StateMachine(tagApp, inputs)
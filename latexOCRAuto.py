from latexOCR import workSetup, resizer, predictor, persist, auth, subscription
from StateMachine.primitives import *
from StateMachine.console import *

app = Application('latexOCR')
app.setInputMap({'userID', 'imgPath'})

bastAuth = SwitchFlow('Dummy_authentication')

subs = Lambda('Need_subscription')
subs.materialize_worker(subscription.subscription, {'userID'}, {'userID'})

initWork = Lambda('Work_setup')
initWork.materialize_worker(workSetup.workSetup, {'imgPath'}, {'img', 'imgName', 'args'})

resizeImg = Lambda('Resize_image')
resizeImg.materialize_worker(resizer.resizer, {'img', 'args'}, {'img'})

ml = Lambda('ML_inference')
ml.materialize_worker(predictor.predictor, {'img', 'args'}, {'pred'})

store = Lambda('Cache_render')
store.materialize_worker(persist.persist, {'userID', 'imgName', 'pred'}, {'status'})

app.setNext(bastAuth)
bastAuth.addRule({'userID', 'imgPath'}, {'userID', 'imgPath'}, auth.auth, [subs, initWork], 0)
initWork.setNext(resizeImg)
resizeImg.setNext(ml)
ml.setNext(store)

inputs = Inputs()
inputs.addInput({'userID': 100, 'imgPath': '/Users/zijian/Desktop/ucsd/291_Vir/project/playaround/latexOCR/imgs/eq1.png'}, 'valid auth') 
# inputs.addInput({'userID': 0, 'imgPath': '/Users/zijian/Desktop/ucsd/291_Vir/project/playaround/latexOCR/imgs/eq1.png'}, 'inadquate privilege')

sm = StateMachine(app, inputs)
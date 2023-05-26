import numpy as np
import time
import matplotlib.pyplot as plt

np.random.seed(1)
from ANNmodel import AdvAutoencoder
from functions import *

Option = Options()
Option.fitHorizon = 6
Option.dynamicalSystemSelector = systemSelectorEnum.wienerHammersteindataset
Option.closedLoopSim = False
Option.stringDynamicalSystemSelector = str( \
    Option.dynamicalSystemSelector).replace('<function systemSelectorEnum.',
    '').split(' at ')[0]
Option.nonLinearInputChar = False
Option.stateSize = 5
Option.n_a = 10
Option.affineStruct = True
Option.useGroupLasso = False;   
Option.regularizerWeight = 0.0001

import warnings
warnings.filterwarnings("ignore")

#%% DS generation and model learning
simulatedSystem, U_n, Y_n, U_Vn, Y_Vn = Option.dynamicalSystemSelector()

model = AdvAutoencoder(affineStruct = Option.affineStruct,
        useGroupLasso = Option.useGroupLasso,
        stateReduction = Option.stateReduction,
        fitHorizon = Option.fitHorizon,
        strideLen = Option.n_a,#n_a=n_b
        outputWindowLen = 2,#+1 wrt the paper
        n_layer = 3,
        n_neurons = 30,                     
        regularizerWeight = Option.regularizerWeight,
        stateSize = Option.stateSize)
model.setDataset(U_n.copy(), Y_n.copy(), U_Vn.copy(), Y_Vn.copy())

inputU, inputY = model.prepareDataset()
model.trainModel()
predictedLeft, stateLeft, oneStepAheadPredictionError, \
    forwardedPredictedError, forwardError = \
    model.model.predict([inputY, inputU])

#%% Model Validation Validation
voM = False
r = -1
start = time.time()
fit, NRMSE, logY, logYR = openLoopValidation(model,
        validationOnMultiHarmonic = voM,
        reset = r,
        YTrue = Y_Vn.copy(),
        U_Vn = U_Vn.copy(),
        openLoopStartingPoint = Option.openLoopStartingPoint)
end = time.time()

with open("res/wh_output.txt", "w+") as f:
    for i in range(0, logY.size):
        f.write(str(logY[i]) + "\n")

with open("res/wh_real_output.txt", "w+") as f:
    for i in range(0, logY.size):
        f.write(str(logYR[i]) + "\n")

plt.figure()
plt.plot(logYR, label = 'y')
plt.plot(logY - logYR, label = 'y-hy')
plt.legend()
plt.show()

#on training set
voM = False
r = -1
start = time.time()
fit_tr, NRMSE_tr, logY_tr, logYR_tr = openLoopValidation(model,
        validationOnMultiHarmonic = voM,
        reset = r,
        YTrue = Y_n.copy(),
        U_Vn = U_n.copy(),
        openLoopStartingPoint = Option.openLoopStartingPoint)
end = time.time()

plt.figure()
plt.plot(logYR_tr, label = 'y')
plt.plot(logY_tr - logYR_tr, label = 'y-hy')
plt.legend()
plt.show()

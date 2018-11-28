import os
from os import listdir
import numpy as np
from shutil import copyfile

def splitTrainAndTestData():
    file_names = listdir("../701_StillsRaw_full")
    file_names.remove("Seq05VD_f02610.png") # The mask of this image is invalid
    np.random.shuffle(file_names)
    file_names = list(map(lambda x: x.split(".")[0], file_names))
    os.mkdir("trainDataRaw")
    os.mkdir("trainDataMask")
    os.mkdir("testDataRaw")
    os.mkdir("testDataMask")
    for i in range(175):
        copyfile("../701_StillsRaw_full/"+file_names[i]+".png", "testDataRaw/"+file_names[i]+".png")
        copyfile("../LabeledApproved_full/"+file_names[i]+"_L.png", "testDataMask/"+file_names[i]+"_L.png")
    for i in range(175,700):
        copyfile("../701_StillsRaw_full/"+file_names[i]+".png", "trainDataRaw/"+file_names[i]+".png")
        copyfile("../LabeledApproved_full/"+file_names[i]+"_L.png", "trainDataMask/"+file_names[i]+"_L.png")


splitTrainAndTestData()

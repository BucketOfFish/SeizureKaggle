############################################################################
# Author: Matt Zhang                                                       #
# Email: set.stun@gmail.com                                                #
#                                                                          #
# This code calls feature extraction code on all training samples, then    #
# saves the features for each patient in a separate file.                  #
############################################################################

import os.path
from scipy.io import loadmat,savemat
import ExtractFeatures

def writeToFile(outputFile, features, overwrite=False):

    # CALLED BY: processPatient()

    # append new dictionary keys to .mat file if it already exists. else, create it.

    if os.path.exists(outputFile) and not overwrite:
        matFile = loadmat(outputFile)
        existingFeatures = [key for key in matFile.keys() if not key.startswith('_')]
        for feature in existingFeatures:
            features[feature] = matFile[feature] # add existing features to the dictionary

    savemat(outputFile, features)

def processPatient(folder, patient, nSamples, overwrite=False):

    # CALLS: ExtractFeatures.extractFeatures(), writeToFile()
    # CALLED BY: __main__()

    # there is training and test data for three patients, stored in folders like 'training_1', 'test_3', etc.
    # training data: I_J_K.mat - the Jth training data segment corresponding to the Kth class for the Ith patient
    # test data: I_J.mat - the Jth testing data segment for the Ith patient
    # K=0 for interictal, K=1 for preictal
    # this function looks at the training samples for a single patient
    # for each sample, call extractFeatures() and write new features to a single output file for the patient
    # uses writeToFile() to write new features to the patient's file
    # interictal and preictal data are combined in the file, but labelled

    print "Processing patient ", patient
    subFolder = "training_%d" % patient
    inputFolder = os.path.join(folder, subFolder) # folder full of samples for the patient

    outputFileName = "patient_%d_training.mat" % patient
    outputFile = os.path.join(folder, outputFileName) # the file that features will be written to

    # find out which features have already been written, so we do not calculate them again
    if os.path.exists(outputFile) and not overwrite:
        matFile = loadmat(outputFile)
        existingFeatures = [key for key in matFile.keys() if not key.startswith('_')]
        # print matFile['nSamplesSegment']
        del matFile
    else:
        existingFeatures = []

    if not overwrite:
        print "Existing features: ", existingFeatures
    else:
        print "Overwriting existing features."

    # keep track of extracted features and collect all of them
    newFeatures = {}

    for EEG_class in range(2): # 0=interictal, 1=preictal
        for sampleNumber in range(1,nSamples[EEG_class]+1):
            fileName = "%d_%d_%d.mat" % (patient, sampleNumber, EEG_class)
            inputSample = os.path.join(inputFolder, fileName) # single input sample
            print 'Extracting features from ', fileName
            # add extracted features to newFeatures dictionary
            matFile = loadmat(inputSample)
            sampleFeatures = ExtractFeatures.extractFeatures(matFile, existingFeatures)
            for key in sampleFeatures.keys():
                newFeatures.setdefault(key, []).append(sampleFeatures[key])
            newFeatures.setdefault('EEG_class', []).append(EEG_class)

    if not overwrite:
        print "Existing features: ", existingFeatures
    else:
        print "Overwriting existing features."
    print "New features: ", newFeatures.keys(), "\n"
    writeToFile(outputFile, newFeatures, overwrite) # add new features to output file

if __name__ == "__main__":

    # CALLS: processPatient()

    # calls processPatient() for each patient
    # contol panel here

    folder = "/Users/mattzhang/Dropbox/Research/Notebooks/Data/SeizureKaggle/"
    nSamples = [[3, 3], [0, 0], [0, 0]] # nSamples[nPatient][EEG_type] - hard coded - K=0 for interictal, K=1 for preictal
    # nSamples = [[1152, 150], [0, 0], [0, 0]] # nSamples[nPatient][EEG_type] - hard coded - K=0 for interictal, K=1 for preictal
    overwrite = True

    for patient in range(1,4): # three patients
        processPatient(folder, patient, nSamples[patient-1], overwrite)

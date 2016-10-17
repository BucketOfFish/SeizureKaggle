############################################################################
# Author: Matt Zhang                                                       #
# Email: set.stun@gmail.com                                                #
#                                                                          #
# This code uses features extracted from patient EEG data to train a       #
# seizure predictor for each patient.                                      #
############################################################################

from scipy.io import loadmat
from sklearn.cross_validation import train_test_split, KFold

def trainPatient (X, y): # y=0 for interictal, y=1 for preictal

    # CALLED BY: __main__()

    # given the data for a patient, train a classifier

    # do K-fold splitting on training data, where K is 5
    k_fold = KFold(n_splits=5)
    dataSplitIndices = k_fold.split(X) # use example: [svc.fit(X[train], y[train]) for train, test in k_fold.split(X)]

    # return classifier

if __name__ == "__main__":

    # import data

    folder = "/Users/mattzhang/Dropbox/Research/Notebooks/Data/SeizureKaggle/"

    for patient in range(1,4): # three patients

        inputFileName = "patient_%d_training.mat" % patient
        inputFile = os.path.join(folder, inputFileName)

        if os.path.exists(inputFile):

            matFile = loadmat(inputFile)
            y = matFile["EEG_class"][0] # a 1D array
            X = matFile # X is a dictionary
            del matFile["EEG_class"]
            trainPatient (patient, X, y)

            outputFileName = "patient_%d_classifier.mat" % patient
            outputFile = os.path.join(folder, inputFileName)

            # use Pickle to save classifier as outputFile

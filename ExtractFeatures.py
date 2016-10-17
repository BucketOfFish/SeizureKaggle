############################################################################
# Author: Matt Zhang                                                       #
# Email: set.stun@gmail.com                                                #
#                                                                          #
# Given an input sample, this code extracts a list of features.            #
############################################################################

from __future__ import print_function, division

import sys
import numpy as np
import pandas as pd
from math import *
from scipy.stats import skew, kurtosis
import pyeeg
import matplotlib.pyplot as plt

def corr(inputData):

	# CALLED BY: extractFeatures()

	# returns correlation matrix eigenvalues

	data = pd.DataFrame(data=inputData)
	C = np.array(data.corr()) # default correlation method is Pearson
	C[np.isnan(C)] = 0
	C[np.isinf(C)] = 0
	w,_ = np.linalg.eig(C)
	x = np.sort(w)
	x = np.real(x)
	return x # return eigenvalues of correlation matrix

def extractFeatures(matFile, existingFeatures):

	# CALLS: corr()
	# CALLED BY: ProcessPatients.processPatient()

	# given a single sample (.mat file), extract and return the features (divided by minutes) in a dictionary.

	dataStruct = matFile['dataStruct']
	# I'm not counting __version__, __header__, and __global__ as relevant keys
	# in the .mat file, the only relevant key should be dataStruct, with (only) the following fields:
	data = dataStruct['data'][0,0] # 2D array - for some reason, dataStruct[0] is an array inside an object inside an object 
	# data is a matrix of iEEG sample values arranged row x column as time sample x electrode.
	iEEGsamplingRate = dataStruct['iEEGsamplingRate'][0,0][0][0] # single value - more weird shit
	# iEEGsamplingRate: data sampling rate, i.e. the number of data samples representing 1 second of EEG data.
	nSamplesSegment = int(dataStruct['nSamplesSegment'][0,0][0][0]) # single value
	# nSamplesSegment: total number of time samples (number of rows in the data field).
	channelIndices = dataStruct['channelIndices'][0,0][0] # 1D array - just counts 1 through 16
	# channelIndices: an array of the electrode indexes corresponding to the columns in the data field.
	sequence = int(dataStruct['sequence'][0,0][0][0]) # single value
	# sequence: the index of the data segment within the one hour series of clips. For example, 1_12_1.mat has a sequence number of 6, and represents the iEEG data from 50 to 60 minutes into the preictal data. This field only appears in training data.

	[nTimesteps, nChannels] = data.shape
	nSamplesPerMinute = int(floor(iEEGsamplingRate * 60))
	nMinutes = int(floor(nTimesteps/nSamplesPerMinute))
	minuteIndices = range(0,(nMinutes+1)*nSamplesPerMinute,nSamplesPerMinute//2) # divide data up into overlapping minute-long slices

	# extract features per minute
	print('Processing minute slice ', end="")
	sys.stdout.flush()

	features = {} # a dictionary
	# call .ravel() on all data that comes back

	for minuteSlice in range(len(minuteIndices)-3): # yeah I checked, 3 is correct

		print('{}... '.format(minuteSlice+1), end="")
		sys.stdout.flush()

		minuteData = data[minuteIndices[minuteSlice]:minuteIndices[minuteSlice+2], :]

		# # basic statistics

		# if 'Mean' not in existingFeatures:
			# mean = np.mean(minuteData) # array of 16 numbers with mean for each channel
			# features.setdefault('Mean', []).append(mean)

		# if 'Variance' not in existingFeatures:
			# variance = np.var(minuteData) # array of 16 numbers with std for each channel
			# features.setdefault('Variance', []).append(variance)

		# if 'Skewness' not in existingFeatures:
			# skewness = skew(minuteData) # array of 16 numbers with skewness for each channel
			# features.setdefault('Skewness', []).append(skewness)

		# if 'Kurtosis' not in existingFeatures:
			# kurtosisFeature = kurtosis(minuteData) # array of 16 numbers with kurtosis for each channel
			# features.setdefault('Kurtosis', []).append(kurtosisFeature)

		# # Hjorth parameters

		# if 'Mobility' not in existingFeatures:
			# mobility = np.divide(np.std(np.diff(minuteData)), np.std(minuteData))
			# features.setdefault('Mobility', []).append(mobility)

		# if 'Complexity' not in existingFeatures:
			# complexity = np.divide(np.divide(np.std(np.diff(np.diff(minuteData))), np.std(np.diff(minuteData))), mobility)
			# features.setdefault('Complexity', []).append(complexity)

		# # fractal dimensions

		# if 'PetrosianFractalDimension' not in existingFeatures:
			# PetrosianFractalDimension = np.zeros(nChannels)
			# for channel in range(nChannels):
				# PetrosianFractalDimension[channel] = pyeeg.pfd(minuteData[:, channel]) # Petrosian fractal dimension
			# features.setdefault('FractalDimensions', []).append(PetrosianFractalDimension)
			# print(PetrosianFractalDimension)

		# if 'HjorthFractalDimension' not in existingFeatures:
			# HjorthFractalDimension = np.zeros(nChannels)
			# for channel in range(nChannels):
				# HjorthFractalDimension[channel] = pyeeg.hfd(minuteData[:, channel], 3) # Hjorth fractal dimension
			# features.setdefault('FractalDimensions', []).append(HjorthFractalDimension)
			# print(HjorthFractalDimension)

		# if 'HurstExponent' not in existingFeatures:
			# HurstExponent = np.zeros(nChannels)
			# for channel in range(nChannels):
				# HurstExponent[channel] = pyeeg.hurst(minuteData[:, channel]) # Hurst exponent - 0.5 is a random walk
			# HurstExponent[np.isnan(HurstExponent)] = 0
			# features.setdefault('FractalDimensions', []).append(HurstExponent)
			# print(HurstExponent)

		# frequency-based measures

		freqBandEdges = np.array([0.1, 4, 8, 12, 30, 70, 180]) # frequency band edges in Hz
		# These are frequency bands defined in "Crowdsourcing reproducible seizure forecastin in human and canine epilepsy".
		# Six bands of interest: delta (0.1-4 Hz), theta (4-8), alpha (8-12), beta (12-30), low-gamma (30-70), high-gamma(70-180).
		freqIndices = np.round(nTimesteps/iEEGsamplingRate*freqBandEdges).astype('int') # which indices in FFT array go to each frequency
		hammingWindow = np.hamming(len(minuteData)) # Hamming window
		windowedMinuteData = np.multiply(np.transpose(np.matrix(hammingWindow)), np.matrix(minuteData))
		FFTamplitude = np.absolute(np.fft.fft(windowedMinuteData, n=freqIndices[-1], axis=0)) # 108000 x 16 array
		FFTamplitude[0,:] = 0 # set the DC component to zero
		FFTamplitude /= FFTamplitude.sum() # normalize each channel
		plt.plot(minuteData)
		plt.show()
		plt.plot(windowedMinuteData)
		plt.show()
		plt.plot(FFTamplitude)
		plt.show()

		bandPowerDensity = np.zeros((len(freqBandEdges)-1, nChannels))
		for j in range(len(bandPowerDensity)):
			bandPowerDensity[j,:] = 2*np.sum(FFTamplitude[freqIndices[j]:freqIndices[j+1],:], axis=0) # scaled power in each band

		if 'BandPowerDensity' not in existingFeatures:
			features.setdefault('BandPowerDensity', []).append(bandPowerDensity)

		if 'ShannonEntropy' not in existingFeatures: # -k_B*sum[p_i*log(p_i)]
			ShannonEntropy = -1*np.sum(np.multiply(bandPowerDensity,np.log(bandPowerDensity)), axis=0) # power entropy in each channel
			features.setdefault('ShannonEntropy', []).append(ShannonEntropy)

		if 'SpectralEdgeFrequency' not in existingFeatures: # SEF x = frequency below which x% of total signal power is
			topFreq = 180 # to speed up calculations, we look at the range below 180 Hz, because most power is below this frequency
			powerCutoff = 0.5 # x = 90%
			topFreqIndex = int(round(nTimesteps/iEEGsamplingRate*topFreq)) # similar to freqIndices above
			cumPowerChannel = np.cumsum(FFTamplitude[:topFreqIndex,:], axis=0) # cumulative power at different freqs in each channel
			shiftedCumPowerChannel = cumPowerChannel - (cumPowerChannel.max(axis=0)*powerCutoff) # shift the baseline (see next line)
			spectralEdgeIndex = np.argmin(np.abs(shiftedCumPowerChannel), axis=0) # index of lowest value by channel (corresponds to SEF)
			spectralEdgeFrequency = spectralEdgeIndex/topFreqIndex*topFreq
			features.setdefault('SpectralEdgeFrequency', []).append(spectralEdgeFrequency)

		# correlations

		if 'TimeCorrelation' not in existingFeatures: # eigenvalues of correlation matrix (time corr b/w channels)
			features.setdefault('TimeCorrelation', []).append(corr(minuteData))
			
		if 'FrequencyCorrelation' not in existingFeatures: # eigenvalues of correlation matrix (freq corr b/w freq)
			features.setdefault('FrequencyCorrelation', []).append(corr(bandPowerDensity))

	print("")

	# delete everything to free up memory. not explicitly calling the garbage collector.
	del matFile
	del dataStruct
	del data
	del iEEGsamplingRate
	del nSamplesSegment
	del channelIndices
	del sequence

	return features # give back the features to be written to output

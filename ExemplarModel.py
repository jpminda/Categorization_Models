import numpy as np #this is needed for any arrays and other stuff
import matplotlib.pyplot as plt

#The values below should change for different catgory sets. 
# e.g. NumDim is 6 for 6dNLSStim.txt

NumDim  = 4  #the number of dimenions or features
NumAStim = 5 #the number of category A exemplars
NumBStim = 4 #the number of category B exemplars
NumTrainStim = 9 #The total number of training exemaplrs
NumTransferStim = 7 #the number of transfer stimuli (if used)
NumTotalStim = 16 #the total number of stimuli in the problem set

#To make the easy, create one array for all the stimuli to be categorized
Stim = np.zeros((NumTotalStim, NumDim))
Stim = np.loadtxt("54taskStm.txt",  delimiter=" ")

#other examples
#Stim = np.loadtxt("6dNLSStim.txt",  delimiter=" ")
#Stim = np.loadtxt("The75Task.txt",  delimiter=" ")


#an array of the weights in the task, corresponds to numDim
weights = np.array ([.25, .25, .25, .25])
sens = 6
SharedTraits = np.zeros(NumDim)
SimA = np.zeros(NumTotalStim)
SimB = np.zeros(NumTotalStim)
probs = np.zeros(NumTotalStim)

thismult = 0
cummulta = 0
cummultb = 0

for ThisStim in range(NumTotalStim):
	for ThisTrainStim in range(NumTrainStim):
		for ThisDim in range(NumDim):
			if Stim[ThisStim, ThisDim] == Stim[ThisTrainStim, ThisDim]: 
				SharedTraits[ThisDim] = 0
			else:
				SharedTraits[ThisDim] = weights[ThisDim]
		for ThisDim in range(NumDim):
			thismult = thismult + SharedTraits[ThisDim]
		if ThisTrainStim <= NumAStim:
			cummulta = cummulta + np.exp(-sens * thismult)
		else:
			cummultb = cummultb + np.exp(-sens * thismult)
		thismult = 0.0
		for ThisDim in range(NumDim):
			SharedTraits[ThisDim] = 0.0
	SimA[ThisStim] = cummulta
	SimB[ThisStim] = cummultb
	cummulta = 0
	cummultb = 0

for ThisStim in range(NumTotalStim):
	probs[ThisStim] = SimA[ThisStim] / (SimA[ThisStim] + SimB[ThisStim])

probs = np.vstack(probs)
print(np.round(probs,3))

plt.show(plt.plot(probs))



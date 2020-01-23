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
sens = 3
SharedTraits = np.zeros(NumDim)
SimA = np.zeros(NumTotalStim)
SimB = np.zeros(NumTotalStim)
probs = np.zeros(NumTotalStim)


for ThisStim in range(NumTotalStim): #loop through all of the stimuli
	for ThisTrainStim in range(NumAStim): #compare the stimulus to be classified with the As
		DistA = np.zeros(NumDim)
		for ThisDim in range(NumDim):
			DistA[ThisDim] = DistA[ThisDim] + (weights[ThisDim]) * abs(Stim[ThisStim,ThisDim] - Stim[ThisTrainStim,ThisDim])
		SimA[ThisStim] = SimA[ThisStim] + np.exp(-sens * sum(DistA))
	
	for ThisTrainStim in range((NumAStim), NumTrainStim): #compare the stimulus to be classified with the Bs
		DistB = np.zeros(NumDim)
		for ThisDim in range(NumDim):
			DistB[ThisDim] = DistB[ThisDim] + (weights[ThisDim]) * abs(Stim[ThisStim,ThisDim] - Stim[ThisTrainStim,ThisDim])
		SimB[ThisStim] = SimB[ThisStim] + np.exp(-sens * sum(DistB))

for ThisStim in range(NumTotalStim):
	probs[ThisStim] = SimA[ThisStim] / (SimA[ThisStim] + SimB[ThisStim])

probs = np.vstack(probs) #This command just make the array print up at as a column of numbers
print(np.round(probs,3))

plt.show(plt.plot(probs))



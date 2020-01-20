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

#declaring the prototypes here
A_Prot=np.array([0,0,0,0])
B_Prot=np.array([1,1,1,1])
#A_Prot=np.array([0,0,0,0,0,0])
#B_Prot=np.array([1,1,1,1,1,1])


#an array of the weights in the task, corresponds to numDim
#There are 6, so 0-5. #weights sum to 1.0 and have a range of 0-1
weights = np.array ([.25, .25, .25, .25]) #for the four dim set
#weights = np.array ([.17, .17, .17, .16, .16, .17]) #for the 6 dim set

sens = 2 #can vary for simulations, usually between 0.1 and 10.0
DistA = np.zeros(NumDim)
DistB = np.zeros(NumDim)
SimA = np.zeros(NumTotalStim)
SimB = np.zeros(NumTotalStim)
probs = np.zeros(NumTotalStim)



for ThisStim in range(NumTotalStim):
	DistA = np.zeros(NumDim)
	DistB = np.zeros(NumDim)

	for ThisDim in range(NumDim):
		#The comparison process is across all the dimenions in the task
		#each dimension has an attentional weight (which is given or estimated) that is a multiplier for each dimension

		DistA[ThisDim] = DistA[ThisDim] + (weights[ThisDim]) * abs((Stim[ThisStim,ThisDim] - A_Prot[ThisDim]))
		DistB[ThisDim] = DistB[ThisDim] + (weights[ThisDim]) * abs((Stim[ThisStim,ThisDim] - B_Prot[ThisDim]))
	SimA[ThisStim] = np.exp(-sens * sum(DistA))
	SimB[ThisStim] = np.exp(-sens * sum(DistB))



for ThisStim in range(NumTotalStim):
	probs[ThisStim] = SimA[ThisStim] / (SimA[ThisStim] + SimB[ThisStim])

probs = np.vstack(probs)
print(np.round(probs,3))

plt.show(plt.plot(probs))



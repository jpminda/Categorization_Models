###########################################################################
#																		  #
# Prototype model from and Minda and Smith 2001   						  #
# Coded by JP Minda, Feb 2020											  #
#																		  #
###########################################################################



import numpy as np #this is needed for any arrays and other stuff
import matplotlib.pyplot as plt


#The values below should change for different catgory sets. 
# e.g. NumDim is 6 for 6dNLSStim.txt

NumDim  = 6  #the number of dimenions or features
NumAStim = 7 #the number of category A exemplars
NumBStim = 7 #the number of category B exemplars
NumTrainStim = 14 #The total number of training exemaplrs
NumTransferStim = 0 #the number of transfer stimuli (if used)
NumTotalStim = 14 #the total number of stimuli in the problem set

#To make the easy, create one array for all the stimuli to be categorized
Stim = np.zeros((NumTotalStim, NumDim))

#other examples
Stim = np.loadtxt("6dNLSStim.txt",  delimiter=" ")
#Stim = np.loadtxt("The75Task.txt",  delimiter=" ")
#Stim = np.loadtxt("54taskStm.txt",  delimiter=" ")


#declaring the prototypes 
#A_Prot=np.array([0,0,0,0])
#B_Prot=np.array([1,1,1,1])
A_Prot=np.array([0,0,0,0,0,0])
B_Prot=np.array([1,1,1,1,1,1])


#decalre a single array to hold the average predictions for each stimulis
BigProbs = np.zeros(NumTotalStim)

#Begin Monte Carlo Loop
#SimLoop is the number of simulated subjects.
#if you just want a single prediction, go ahead and run a loop of 1
#you can override the random seed

for SimLoop in range(1000):
	#start with one random number and then choose each of the next weights from what is left 
	#for the other weights. This makes sure that they sum to 100 (or 1.0)
	w1 = np.random.random_integers(0,100)
	w2 = np.random.random_integers(0,(100-w1))
	w3 = np.random.random_integers(0,(100-(w1+w2)))
	w4 = np.random.random_integers(0,(100-(w1+w2+w3)))
	w5 = np.random.random_integers(0,(100-(w1+w2+w3+w4)))
	w6 = 100-(w1+w2+w3+w4+w5) 

	#some of the code here is just for checking things, usually commented out
	w = np.array([w1,w2,w3,w4,w5,w6]) #all the weights in an array
	#print(w) #uncomment if you want to see the weights
	np.random.shuffle(w) #shuffle the weights to make it random
	#print(w) #uncomment if you want to print the suffled weights
	#print(sum(w)) #uncomment if you want to print the sum (should always be 100)
	
	weights = w/100.0 #takes the random itger weights and converd to 0-1.0 numbers

	#the examples below can be used to override random weights to demonstrate specific attentional configurations
	
	#weights = np.array ([.25, .25, .25, .25]) #for the four dim set
	#weights = np.array ([.17, .17, .17, .16, .16, .17]) #for the 6 dim set
	#weights = np.array ([1.0, .0, .0, .0, .0, .0]) #for the 6 dim set, attention to one dim
	#print(weights)  #uncomment if you want to see the weights
	#print(sum(weights)) #uncomment if you want to print the sum (should always be 1.0)

	sens = np.random.random_integers(0,100) #can vary for simulations, usually between 0.1 and 10.0, divide by 10
	sens = sens/10 #this is the similarity scaling paramter 
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

	print(np.round(probs,3)) #prints out the probabilities

	#d = plt.plot(probs) #plots each one, uncomment if you want a plot with each run

	BigProbs=BigProbs+probs #ads the current probs to the array that will hold the average

#plt.show(d) #plots all of the lines 

BigProbs=BigProbs/SimLoop #find the average perforance

SaveData = np.asarray(np.round(BigProbs,3)) 
print(SaveData)
np.savetxt('prot_out.csv', SaveData, fmt='%s',delimiter=',') #save to a file

plt.show(plt.plot(BigProbs, 'o-')) #plots the average 





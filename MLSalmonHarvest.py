import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

#part 1: salmon machine learning
for filename in ['dip-har-eff.csv','drift-har-eff.csv','set-har-eff.csv']:
    #read in the data
    data = pd.read_csv(filename)

    #getting the number of rows and columns
    rows = data.shape[0]
    cols = data.shape[1]
    #batch_size = 19

    #convert df into arr
    data = data.values
    data = data[np.arange(0,rows),:]
    
    #the lambda val/step value
    mu = 0.1
    iters = 50
    # : means all, splits into vecctor of just x col and y col
    X = data[:,2]
    Y = data[:,1]
    #print(X)
    #print(Y)

    #normalize the data
    X = np.true_divide(X,np.max(X))
    Y = np.true_divide(Y,np.max(Y))

    #print(X,Y)

    #label the plot
    plt.figure(1)
    plt.xlabel("Days Fished")
    plt.ylabel("Number of Salmon Caught")


    #init starter b0 and b1 vals
    b0,b1 = 0,0
    #used to keep colors consistent
    color = ['red','blue','green','purple','orange']
    m = 0

    for batch_size in [1,5,10,15,19]:
        currColor = color[m]
        plt.figure(1)
        plt.scatter(X,Y,color='blue')
        plt.plot(X, X*b1+b0,color='black')
        plt.title("Effort vs Harvest of Salmon: " + filename)
        E = np.empty(iters)
        for epoch in np.arange(iters):
                # for each training sample, compute the gradient
                grad0 = 1.0/batch_size * sum([(b0 + b1*X[i] - Y[i]) for i in range(batch_size)]) 
                grad1 = 1.0/batch_size * sum([(b0 + b1*X[i] - Y[i])*X[i] for i in range(batch_size)])
                # update the theta_temp
                temp0 = b0 - mu * grad0
                temp1 = b1 - mu * grad1
                # update theta
                b0 = temp0
                b1 = temp1
                # mean squared error
                e = sum( [ (b0 + b1*X[i] - Y[i])**2 for i in range(batch_size)] ) 
                #adding all errors to array for plotting
                E[epoch] = e 
        #plotting the three charts for each dataset
        plt.plot(X,X*b1 + b0, label=str(batch_size) + ' Y = ' + str(b1) + 'x + ' + str(b0),color=currColor) 
        if m < 4: m+=1
        plt.legend()
        plt.figure(3) 
        plt.plot(X,X*b1 + b0, label= 'batch: ' + str(batch_size),color=currColor)      
        plt.legend()
        plt.figure(2)
        plt.plot(E, label= 'batch: ' + str(batch_size),color=currColor)
        plt.legend()
        plt.title("Convergence of Error vs Epoch of Batch Sizes ")
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        r2 = r2_score(Y,b1*X + b0)
        print(str(batch_size) + " R Squared: " + str(r2))
        validation_nums = np.random.randint(0,40000,5)
        validation_nums = np.true_divide(validation_nums,max(validation_nums))
        plt.figure(3)
        plt.scatter(X,Y,color='black')
        plt.scatter(validation_nums,b1*validation_nums + b0,label = str(batch_size),color=currColor)
        plt.title("Predicted Values using Parameter Equations")
        plt.xlabel("Number of Days")
        plt.ylabel("Harvest")
        print("learned parameters: b1 = " + str(b1) + " b0 = " + str(b0))
        #reset the parameters for each batch
        b0,b1 = 0,0      
    plt.show()


# Part Two: Correlation between heavy metal levels in fish
albacoreData = pd.read_csv('albacore_metal.dat')
rows = albacoreData.shape[0]
albacoreData = albacoreData.values
albacoreData = albacoreData[np.arange(0,rows),:]

cad = albacoreData[:,1]
merc = albacoreData[:,2]
lead = albacoreData[:,3]
plt.figure()
plt.scatter(cad,albacoreData[:,0],color='red')
plt.scatter(merc,albacoreData[:,0],color='blue')
plt.scatter(lead,albacoreData[:,0],color='green')
plt.show()

print("Correlation between cadmium and mercury: " + str(np.corrcoef(cad,merc)))
print("Correlation between mercury and lead: " + str(np.corrcoef(merc,lead)))
print("Correltation between cadmium and lead: " + str(np.corrcoef(lead,cad)))

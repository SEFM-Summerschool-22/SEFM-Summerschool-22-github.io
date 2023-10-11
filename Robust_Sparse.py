### Robust/Sparse Code in Python 

###Libraries needed:
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import svm
from itertools import chain

def createData():
    ### Create two random data sets of 5 points, in the range of 0-20 
    x1 = random.sample(range(0, 20), 5)
    x2 = random.sample(range(0, 20), 5)
    y1 = random.sample(range(0, 20), 5)
    y2 = random.sample(range(0, 20), 5)
    data1 = zip(x1,y1)
    data2 = zip(x2,y2)
    return list(data1), list(data2)
    
def ModelPlot(data1,data2):
    ### Use an SVM to model the data and find the best separating line between the two classes and plot it
    X = np.array(data1 + data2)
    Y = np.array([0] * len(data1) + [1] * len(data2))
    clf = svm.SVC(kernel='linear', C=10000) # C is important here
    clf.fit(X, Y)
    plt.figure(figsize=(4, 4))
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(0, 100)
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.plot(xx, yy, "k-") #********* This is the separator line ************
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,edgecolors="k")
    plt.xlim((0, 50))  
    plt.ylim((0, 50)) 
    return xx, yy

def Model(data1,data2):
    ### Use an SVM to model the data and find the best separating line between the two classes
    X = np.array(data1 + data2)
    Y = np.array([0] * len(data1) + [1] * len(data2))
    clf = svm.SVC(kernel='linear', C=10000)
    clf.fit(X, Y)
    w = clf.coef_[0]
    #error checking if no separating line can be made
    if w[1] == 0:
        return [],[]
    else:
        a = -w[0] / w[1]
        xx = np.linspace(0, 100)
        yy = a * xx - (clf.intercept_[0]) / w[1]
        return xx, yy

def LeftRightLine(xx,yy):
    ### Function to check if each point is left, right or on the separating line
    LeftRight = [[0 for x in range(50)] for y in range(50)]
    x1= xx[0]
    x2= xx[-1]
    y1= yy[0]
    y2= yy[-1]
    for i in range(0,50):
        for j in range(0,50):
            d=(i-x1)*(y2-y1)-(j-y1)*(x2-x1)
            if d < 0:
                LeftRight[i][j] = -1 #right of line
            elif d > 0:
                LeftRight[i][j] = 1 #left of line
            else:
                LeftRight[i][j] = 0 #on the line - would be bottom 
    return LeftRight
    
def Robust(data1,data2):
    #Create an array to store if a point is non-robust points 1, start with all points being robust points 0
    Robust = [[0 for x in range(50)] for y in range(50)]
    #Model orignal data with separating line
    xx,yy = Model(data1,data2)
    #Create an array for if each point is left, right or on the separating line
    LeftRight1 = LeftRightLine(xx,yy)
    for i in range(0,50):
        for j in range(0,50):
            #Add one new data point
            data1.append([i,j])
            #Model new data
            xx,yy = Model(data1,data2)
            if len(xx)!=0: #Removal of errors 
                LeftRight2 = LeftRightLine(xx,yy)
                for x in range(0,50):
                    for y in range(0,50):
                        #Check if the point has switched sides in the new data, and mark it in robust array
                        if LeftRight1[x][y] == 1 and LeftRight2[x][y] == -1:
                            Robust[x][y] = 1
                        elif LeftRight1[x][y] == -1 and LeftRight2[x][y] == 1:
                            Robust[x][y] = 1
                data1.pop(5)
    return Robust
    
def Sparse(data1,data2):
    #Create an array to store if a point is Sparse 1, start with all points being dense 0
    Sparse = [[0 for x in range(50)] for y in range(50)]
    #Model orignal data with separating line
    xx,yy = Model(data1,data2)
    #Create an array for if each point is left, right or on the separating line
    LeftRight1 = LeftRightLine(xx,yy)
    for i in range(0,50):
        for j in range(0,50):
            #Add one new data point
            data1.append([i,j])
            #Model new data
            xx,yy = Model(data1,data2)
            if len(xx)!=0: #Removal of errors 
                LeftRight2 = LeftRightLine(xx,yy)
                for x in chain(range(0,i-5), range(i + 5, 50)): #Excluding a square around the new data
                    for y in chain(range(0,j-5), range(j + 5, 50)):
                        #Check if the point has switched sides in the new data, and mark it in Sparse array
                        if LeftRight1[x][y] == 1 and LeftRight2[x][y] == -1:
                            Sparse[x][y] = 1
                        elif LeftRight1[x][y] == -1 and LeftRight2[x][y] == 1:
                            Sparse[x][y] = 1
                data1.pop(5)  
    return Sparse
    
def ColourGrid(data):
    color_map = {0: np.array([255, 0, 0]), # red
                 1: np.array([0, 0, 255])} # blue 
    # Transpose data so the grid reads teh axes in correctly 
    data = np.array(data)
    data = data.T
    # make a 3d numpy array that has a color channel dimension to be sure the values are always the same colour.
    data_3d = np.ndarray(shape=(50, 50, 3), dtype=int)
    for i in range(0, 50):
        for j in range(0, 50):
            data_3d[i][j] = color_map[data[i][j]]
    #plot and invert for comparisons sake     
    plt.imshow(data_3d)
    ax = plt.gca()
    ax.invert_yaxis()
    
### Example
data1,data2 = createData()
xx,yy = ModelPlot(data1, data2) #Plot original data
robust = Robust(data1,data2)
ColourGrid(robust) 
sparse = Sparse(data1,data2)
ColourGrid(sparse)

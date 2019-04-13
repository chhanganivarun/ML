import random
import matplotlib.pyplot as plt

filename = "../yacht_hydrodynamics.data"
n = 0.01
iterations = 20000
trasize = 200

def dotproduct(W,X):
    return sum( [ w*x for w,x in zip(W,X) ] )

def cost(W,X,t,traset,n):
    Error = 0
    for j in traset:
        diff = ( t[j] - sum( [ w*x for w,x in zip(W,X[j]) ] ) )
        Error += 0.5*(diff**2)/n
    return Error

fi = open(filename,'r')
lines = fi.readlines()
fi.close()



lines = list(map(float,i.split() )for i in lines if len(i))
lines = [ list(x) for x in lines ]
lines = [ x for x in lines if len(x) ]
X = [ [1] + x[:-1] for x in lines ]
t = [ x[-1] for x in lines ]

size = len(X)

if not len(lines):
    exit()

fvlen = len(X[0])

traset = []

for i in range(trasize):
    rv = random.randint(0,len(X)-1)
    while rv in traset:
        rv = random.randint(0,len(X)-1)
    else:
        traset.append(rv)

W = [ round(random.uniform(0,10),2) for i in range(fvlen) ]
#W = [ 0  for i in range(fvlen) ]

print(W)
#print(t)
#print(len(t))


change = [100]

Error = 0
print('Initial Error: '+str(cost(W,X,t,range(0,size),size)))

allErrors=[]

while max(change)>0.1:
    change = [ 0 for x in X[0]]
    for j in traset:
        diff = ( t[j] - sum( [ w*x for w,x in zip(W,X[j]) ] ) )
        change = [ diff*x+ch for x,ch in zip(X[j],change) ]
    W = [ w + n*ch/trasize for w,ch in zip(W,change) ]
    allErrors.append(cost(W,X,t,traset,trasize))
print (W)

plt.plot(allErrors)
plt.show()

#print(allErrors)

Error = 0
for j in traset:
    diff = ( t[j] - sum( [ w*x for w,x in zip(W,X[j]) ] ) )
    Error += 0.5*(diff**2)/(trasize)
print('Training Error: '+str(Error))

Error = 0
for j in [ x for x in range(0,len(X)) if x not in traset ]:
    diff = ( t[j] - sum( [ w*x for w,x in zip(W,X[j]) ] ) )
    Error += 0.5*(diff**2)/(size-trasize)
print('Testing Error: '+ str(Error))

ss= sum( [ (dotproduct(X[i],W)- t[i])**2 for i in traset] )
mean = sum([ t[i] for i in traset ])/trasize
MSE = sum( [ (mean - t[i])**2 for i in traset])
testset = [ i for i in range(size) if i not in traset ]
print("r squared error training",1-(ss/MSE))

testset = [ i for i in range(size) if i not in traset ] 
ss= sum( [ (dotproduct(X[i],W)- t[i])**2 for i in testset] )
mean = sum([ t[i] for i in testset ])/len(testset)
MSE = sum( [ (mean - t[i])**2 for i in testset])

print('r squared error testing',1-(ss/MSE))


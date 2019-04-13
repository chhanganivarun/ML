import random

filename = "../iris.data"
n = 0.01
iterations = 20000
trasize = 70

def dotproduct(W,X):
    return sum( [ w*x for w,x in zip(W,X) ] )

def cost(W,X,t,traset,n):
    misclassified=0
    for j in traset:
        if (dotproduct(W,X[j]) > 0 and t[j] < 0) or (dotproduct(W,X[j]) < 0 and t[j] > 0) :
            misclassified += 1
    return misclassified/n



fi = open(filename,'r')
lines = fi.readlines()
fi.close()


lines = list(map(str,i.replace(',',' ').split() )  for i in lines if len(i) )
lines = [ list(x) for x in lines ]
lines = [ x[:-1]+[-1 if 'virginica' in x[-1] else 1] for x in lines if len(x) and ( 'virginica' in x[-1] or 'versicolor' in x[-1]) ]
X = [ [1] + x[:-1] for x in lines ]
X = [ list(map(float,str(x).replace(']','').replace('[','').replace('\'','').split(','))) for x in X ]
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

W = [ round(random.uniform(0,10),0) for i in range(fvlen) ]

print (W)

change = [100]
Error = 0
print('Initial Error: '+str(cost(W,X,t,range(0,size),size)))

allErrors=[]

for i in range(iterations):
    change = [ 0 for x in X[0]]
    for j in traset:
        diff = ( t[j] - (-1 if dotproduct(X[j],W)<0 else 1) )
        change = [ diff*x+ch for x,ch in zip(X[j],change) ]
    W = [ w + n*ch/2 for w,ch in zip(W,change) ]
    allErrors.append(cost(W,X,t,traset,trasize))

print (W)

misclassified=0
for j in traset:
    if (dotproduct(W,X[j]) > 0 and t[j] < 0) or (dotproduct(W,X[j]) < 0 and t[j] > 0) :
        misclassified += 1
print('Training Classification Error:'+str(misclassified/trasize))
print('cost: '+str(misclassified*2))
misclassified = 0
for j in [ x for x in range(0,len(X)) if x not in traset ]:
    if (dotproduct(W,X[j]) > 0 and t[j] < 0) or (dotproduct(W,X[j]) < 0 and t[j] > 0) :
        misclassified += 1
print('Testing Classification Error:'+str(misclassified/(size-trasize)))
print('cost: '+str(misclassified*2))



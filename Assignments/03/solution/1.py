#!/usr/bin/python3
import random, math, ast, sys, datetime


n = 0.1
tratio = 0.8
attrs = 325

def sigmoid(x):
    return (1/(1+math.e**-x))

def diffsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def maxMat(A):
    if len(A)==0:
        return 0
    return max([ max(a) for a in A ])

def readfile(filename = '/home/varun/UG-Course/Sem4/ML/Assignments/03/Colon_Cancer_CNN_Features.csv'):
    fi = open(filename,'r')
    dataset = fi.readlines()
    fi.close()
    for i in range(len(dataset)):
        dataset[i] = list(ast.literal_eval(dataset[i]))
    random.shuffle(dataset)
    return dataset


def dotprod(A,B):
    #print('dotproduct A:{} B:{}'.format(len(A),len(B)))
    if len(A)!=len(B):
        raise ValueError('Vectors not compatible for dot product')
    return sum([ a*b for a,b in zip(A,B) ])

def diff(A,B):
    #print('diff A:{} B:{}'.format(A,B))
    if len(A)!=len(B):
        raise ValueError('Vectors not compatible for difference')
    return sum([ a-b for a,b in zip(A,B) ])

def vec(t,classes):
    tvec = [ [0,1][t == k] for k in classes ]
    return tvec

def sumMat(A,B):
    '''
        Sum two matrices
    '''
    if len(A)!=len(B):
        raise ValueError('Matrices not compatible for Addition')
    elif len(A) == 0 :
        return A
    
    initlen = len(A[0])
    for i,j in zip(A,B):
        if len(i) != initlen or len(j) != initlen:
            raise ValueError(['Input not matrix','Matrices not compatible for addition'][len(j)!=initlen])

    addition = [ [ 0 for j in a] for a in A ]

    for i in range(len(addition)):
        for j in range(len(addition[i])):
            addition[i][j] = A[i][j] + B[i][j]
    return addition



def SSDloss(classes,t,z):
    loss = 0
    tvec = vec(t,classes)
    for j in z:
        for k in range(len(classes)):
            loss+=(1/2)*(tvec[k]-j[k])**2
    return loss

def CEloss(classes,t,z):
    loss = 0
    for i,j in zip(t,z):
        tvec = [ i == k for k in classes ]
        for k in range(len(classes)):
            loss += -tvec[k] * math.log(j[k]) - (1-tvec[k]) * math.log(1-j[k])

def net_k(w_kj,y,r):
    '''
        given value of r in the range 0 to c-1, and y value for a particular input vector
        return value of z_r
    '''
    return dotprod(w_kj[r],y)

def net_j(w_ji,x,j):
    '''
        given value of j in the range 0 to n_h-1, and y value for a particular input vector
        return value of y_j
    '''
    return dotprod(w_ji[j],x)


def getYZ(dataset,w_kj,wji,n_h,classes):
    zset = [ [ 0 for k in classes ] for data in dataset ]
    yset = [ [ 0 for j in range(n_h) ] for data in dataset ]
    for data_index in range(len(dataset)) :
        for j in range(n_h):
            yset[data_index][j] = sigmoid(net_j(w_ji,dataset[data_index][:-1],j))
        for k in range(len(classes)):
            zset[data_index][k] = sigmoid(net_k(w_kj,yset[data_index],k))
    return yset,zset

def SSDgradient_kj(data,yz,w_kj,w_ji,n_h,classes):
    '''
        for given vector of data, its y and z, we are performing online update
    '''
    y = yz[0]
    z = yz[1]
    dw_kj = [ [ 0 for j in range(n_h)] for k in classes]
    for j in range(n_h):
        for k in range(len(classes)):
            dw_kj[k][j] = n*y[j]*(vec(data[-1],classes)[k]-z[k])*diffsigmoid(net_k(w_kj,y,k))
    return dw_kj

def SSDgradient_ji(data,yz,w_kj,w_ji,n_h,classes,dw_kj):
    #print('test')
    y = yz[0]
    z = yz[1]
    x = data[:-1]
    dw_ji = [ [ 0 for i in range(attrs)] for j in range(n_h)]
    for i in range(attrs):
        for j in range(n_h):
            dw_ji[j][i] = n*sum([ (dw_kj[r][j]/(y[j]*n)) * w_kj[r][j] for r in range(len(classes))]) * diffsigmoid(net_j(w_ji,x,j)) * x[i]
    return dw_ji

def CEgradient_kj(data,yz,w_kj,w_ji,n_h,classes):
    y = yz[0]
    z = yz[1]
    tvec = vec(data[-1],classes)
    dw_kj = [ [ 0 for j in range(n_h)] for k in classes]
    for j in range(n_h):
        for k in range(len(classes)):
            dw_kj[k][j] = n*y[j]*((-tvec[k]/z[k])+((1-tvec[k])/(1-z[k])))*diffsigmoid(net_k(w_kj,y,k))
    return dw_kj

def CEgradient_ji(data,yz,w_kj,w_ji,n_h,classes,dw_kj):
    #print('test')
    y = yz[0]
    z = yz[1]
    x = data[:-1]
    dw_ji = [ [ 0 for i in range(attrs)] for j in range(n_h)]
    for i in range(attrs):
        for j in range(n_h):
            dw_ji[j][i] = n*sum([ (dw_kj[r][j]/(y[j]*n)) * w_kj[r][j] for r in range(len(classes))]) * diffsigmoid(net_j(w_ji,x,j)) * x[i]
    return dw_ji

if __name__ == '__main__':
    if len(sys.argv)>1:
        dataset = readfile(sys.argv[1])
    else:
        dataset = readfile()

    trainset = dataset[0:int(len(dataset)*tratio)]
    testset = dataset[int(len(dataset)*tratio):]

    #trainset = dataset[0:2] # temporary for testing

    classes = list(set([ i[-1] for i in dataset]))
    print(classes)

    print(len(trainset))
    print(len(testset))

    for n_h in range(5,16):
        # w_ji will have dimension n_h x attrs
        w_ji = [ [ (random.random()+1)*5 for i in range(attrs)] for j in range(n_h) ]
        zero_ji = [ [ 0 for i in range(attrs)] for j in range(n_h)]
        # w_kj will have dimension 
        w_kj = [ [ (random.random()+1)*5 for j in range(n_h)] for j in classes ]
        zero_kj = [ [ 0 for i in range(n_h)] for j in range(len(classes))]
        
        CEw_kj = sumMat(w_kj,zero_kj)
        SSDw_kj = sumMat(w_kj,zero_kj)

        CEw_ji = sumMat(w_ji,zero_ji)
        SSDw_ji = sumMat(w_ji,zero_ji)


        print("{} :For n_h={} ".format(datetime.datetime.now(),n_h))

        y_train,z_train = getYZ(trainset,SSDw_kj,SSDw_ji,n_h,classes)
        t_train = [ data[-1] for data in trainset]
        

        ''' Debugging commands
        t_train_vec = [ vec(data[-1],classes) for data in trainset]
        
        print("w_kj:",w_kj)
        print("w_ji:",w_ji)

        print(trainset)
        print(t_train)
        print(y_train)
        print(z_train)
        print(t_train_vec)
        '''

        print('Initial normalized loss on SSD trainset:{}'.format(SSDloss(classes,t_train,z_train)/len(trainset)))
        
        while True:
            for data_index in range(len(trainset)):
                y_train,z_train = getYZ(trainset,SSDw_kj,SSDw_ji,n_h,classes)
                t_train = [ data[-1] for data in trainset]

                SSDdw_kj = SSDgradient_kj(trainset[data_index],(y_train[data_index],z_train[data_index]),SSDw_kj,SSDw_ji,n_h,classes)
                SSDdw_ji = SSDgradient_ji(trainset[data_index],(y_train[data_index],z_train[data_index]),SSDw_kj,SSDw_ji,n_h,classes,SSDdw_kj)
                
                SSDw_kj = sumMat(SSDw_kj,SSDdw_kj)
                SSDw_ji = sumMat(SSDw_ji,SSDdw_ji)
                
                diff = max(maxMat(SSDw_ji),maxMat(SSDw_kj))

                if diff < 0.1:
                    break

            if diff < 0.1:
                break
            
        print('Training finished for SSD with n_h = {}'.format(n_h))

        y_train,z_train = getYZ(trainset,SSDw_kj,SSDw_ji,n_h,classes)
        t_train = [ data[-1] for data in trainset]
        
        print('Final normalized loss on SSD trainset:{}'.format(SSDloss(classes,t_train,z_train)/len(trainset)))



        y_train,z_train = getYZ(trainset,CEw_kj,CEw_ji,n_h,classes)
        t_train = [ data[-1] for data in trainset]

        print('Initial normalized loss on CE trainset:{}'.format(CEloss(classes,t_train,z_train)/len(trainset)))
        
        while True:
            for data_index in range(len(trainset)):
                y_train,z_train = getYZ(trainset,CEw_kj,CEw_ji,n_h,classes)
                t_train = [ data[-1] for data in trainset]

                CEdw_kj = CEgradient_kj(trainset[data_index],(y_train[data_index],z_train[data_index]),CEw_kj,CEw_ji,n_h,classes)
                CEdw_ji = CEgradient_ji(trainset[data_index],(y_train[data_index],z_train[data_index]),CEw_kj,CEw_ji,n_h,classes,CEdw_kj)
                
                CEw_kj = sumMat(CEw_kj,CEdw_kj)
                CEw_ji = sumMat(CEw_ji,CEdw_ji)
                
                diff = max(maxMat(CEw_ji),maxMat(CEw_kj))

                if diff < 0.1:
                    break

            if diff < 0.1:
                break
            
        print('Training finished for CE with n_h = {}'.format(n_h))

        y_train,z_train = getYZ(trainset,CEw_kj,CEw_ji,n_h,classes)
        t_train = [ data[-1] for data in trainset]
        
        print('Final normalized loss on CE trainset:{}'.format(CEloss(classes,t_train,z_train)/len(trainset)))
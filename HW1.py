import numpy as np
import matplotlib.pyplot as plt

def load_data(num):
    #generate data
    f = lambda x:x**2 - 3*x + 1 #y=x^2 - 3x + 1
    F = np.vectorize(f)
    #print(X)
    #print(Y)

    #random data by F(X) + random residual(upper bound=2)
    random_sign = np.vectorize(lambda x: x if np.random.sample() > 1 else -x)
    data_x = 10* np.random.random_sample((1,num)).squeeze()
    data_x = np.sort(data_x)
    data_y = random_sign(np.random.sample(num) * 0.001) + F(data_x)
    
    #save ,load data
    #q = np.concatenate((data_x,data_y),axis=0)
    #np.savetxt(path, q)
    
    return data_x, data_y

def load_data_from_file(path):
    q = np.loadtxt(path, delimiter=',')    
    data_x = q[:, 0]
    data_y = q[:, 1]
    num = len(data_x)
    
    return data_x, data_y, num

#transpose function
def transpose(a):
    s = np.zeros((len(a[0]), len(a)), dtype=float)
    for row in range(len(a)):
        for col in range(len(a[0])):
            s[col][row] = a[row][col]
    return s
#m = [[1, 2], [3, 4], [5, 6]]  
#print (transpose(m))

#((A^T) * A)^-1 * (A^t)* b
#3*10 , 10*3 ,3*3

def LU(result):
    #print("result",result)
    L = np.zeros((len(result), len(result[0])), dtype=float)
    U = np.zeros((len(result), len(result[0])), dtype=float)
      
    for i in range(len(result)):        
        L[i][i] = 1
        for j in range(i+1):
            s1 = sum(U[k][i] * L[j][k] for k in range(j))
            U[j][i] = result[j][i] - s1
        for j in range(i, len(result)):
            s2 = sum(U[k][i] * L[j][k] for k in range(i))
            L[j][i] = (result[j][i] - s2) / U[i][i]
    
    #for r in L:        
        #print("L",L)
    
    #for r in U:        
        #print("U",U)
        
    return (L, U)  

#((lu)^-1) * (A^T) * b
#把(A^T) * b當成 t
#所求x = ((lu)^-1 *t), t = (lu) * x
#假設 ux = y, t = l*y, 先解y
#求的y之後，利用ux = y ,解x

#linverse L function  解y
def linverse(result1,L):
    inv = np.zeros((len(result1), len(result1[0])), dtype=float)
    for i in range(len(result1)): 
        inv[i] = result1[i]        
        for j in range(i):
            inv[i] = inv[i] - inv[j]*L[i][j]
        
    #print(L)
    #print(inv)
    #print(result1)
    return inv

#uinverse U function 解x
def uinverse(inv,U):
    
    
    inv1 = np.zeros((len(inv), len(inv[0])), dtype=float)
    
    for i in range(len(inv)): 
        inv1[len(inv)-i-1] = inv[len(inv)-i-1]        
        for j in range(i):
            inv1[len(inv)-i-1] = inv1[len(inv)-i-1] - inv1[ -j + (len(inv)-1) ]*U[len(inv)-i-1][ -j + (len(inv)-1)]
        inv1[len(inv)-i-1] = inv1[len(inv)-i-1]/U[len(inv)-i-1][len(inv)-i-1]
    #print(U)
    #print(inv1)
    #print(inv)
    return inv1

# identity matrix
def identity(size):    
    #print("size",size)      
    matrix = np.zeros((size, size), dtype=float)
    for i in range(size):
        matrix[i][i] = 1 
    #print(len(matrix))
    return matrix

#lambda*I
def multiplyfactor(factor,matrix):    
    #print("factor",factor)  
    #print(len(matrix))    
    matrixlambda = np.zeros((len(matrix), len(matrix[0])), dtype=float)
    for i in range(len(matrix)):
        matrixlambda[i][i] = matrix[i][i]*factor
    return matrixlambda



#newton_method
def newton_method(X,Y, data_num,base_num):
    print("\n-----newton_method-----\n")
    #base_num = 3
    
    s1 = (base_num,base_num)
    result = np.zeros(s1)# (A^t)*A
    
    s2 = (base_num,1)
    result1 = np.zeros(s2)# ((A^t)* b)   
    result2 = np.zeros(s2)# (A^T) * A * x    
    result3 = np.zeros(s2)# result2 - result1
    
    X = X.reshape(data_num,1)
    A = pow(X, 0)
    for i in range(1, base_num):
        X_power = pow(X, i)
        A = np.concatenate((A, X_power),axis=1)
    #A = np.concatenate((pow(X,0),X,pow(X,2)),axis=1)
    b = Y.reshape(data_num,1)    
    
    AT = transpose(A)
    
    dx_sum = 1
    x = np.zeros((base_num, 1), dtype=float)
    itera = 0
    while dx_sum > 0.00001:
        itera += 1
        # ((A^T) * A) 3*3
        for i in range(len(AT)):     
            for j in range(len(A[0])):         
                for k in range(len(A)): 
                    result[i][j] += AT[i][k] * A[k][j]
                    
        # result2 = (A^T) * A * x
        for i in range (len(result)):
            for j in range(len(x[0])):         
                for k in range(len(x)): 
                    result2[i][j] += result[i][k] * x[k][j]
    
        # result1 = ((A^t)* b) 3*1    
        for i in range(len(AT)):     
            for j in range(len(b[0])):
                for k in range(len(b)): 
                     result1[i][j] += AT[i][k] * b[k][j]
                        
        # result3 = result2 - result1 =  ((A^T) * A * x) - ((A^t)* b)  
        for i in range(len(result1)):     
            for j in range(len(result1[0])):             
                result3[i][j] = result2[i][j] - result1[i][j] 
                
                
    #for r in result1:       
        #print("result1",r)       
            
        #解x , x1 = x0 -  ((A^T) * A))^-1 * ((A^T) * A * x - ((A^t)* b))
        L, U = LU(result)    
        inv = linverse(result3,L)    
        dx = uinverse(inv,U)
        x -= dx
        
        dx_sum = 0
        for i in range(base_num):
            dx_sum += dx[i][0]   
    print("itera:",itera)    
    return x

# linear regression

def linear_regression(X,factor,Y, data_num, base_num):
    print("\n-----linear_regression-----\n")
    #base_num = 3
    
    s1 = (base_num,base_num)
    result = np.zeros(s1)
    
    s2 = (base_num,1)
    result1 = np.zeros(s2)
    
    
    X = X.reshape(data_num,1)
    A = pow(X, 0)
    for i in range(1, base_num):
        X_power = pow(X, i)
        A = np.concatenate((A, X_power),axis=1)
    #A = np.concatenate((pow(X,0),X,pow(X,2)),axis=1)
    b = Y.reshape(data_num,1)    
    
    AT = transpose(A)    
    size = (len(AT))    
    iden = identity(size) 
    matrixlambda = multiplyfactor(factor,iden)
    
    #((A^T) * A) 3*3
    for i in range(len(AT)):     
        for j in range(len(A[0])):         
            for k in range(len(A)): 
                result[i][j] += AT[i][k] * A[k][j]
    #print("ata:",result)
    #print(matrixlambda)
              
    for i in range(len(AT)):
        #print(len(matrixlambda))
        #print(len(result))
        result[i][i] += matrixlambda[i][i]
      
   
    #for r in result:        
        #print("(A^T) * A lambda",r)
    
    
    # ((A^t)* b) 3*1    
    for i in range(len(AT)):     
        for j in range(len(b[0])):            
            for k in range(len(b)): 
                result1[i][j] += AT[i][k] * b[k][j] 
                
                
    #for r in result1:       
        #print("result1",r)       
            
   
    L, U = LU(result)    
    inv = linverse(result1,L)    
    x = uinverse(inv,U)
    print("(A^T) * A:\n",result)
    print("\n-----利用LU分解-----\n")
    print("L:\n",L)
    print("\nU:\n",U)
    print("\n方程式係數:\n",x)
    
    s = "\nfitting line = \n"
    for i in range(base_num-1):
        s += ("%.6lf" % x[i][0]) + "x^" + ("%d" % i) + " + "
    s += ("%.6lf" % x[base_num-1][0]) + "x^" + ("%d" % (base_num-1))
    print(s)
    
    return x
    
def calculate_error(x,data_x, data_y, num, base_num):
    square_error = 0
    total = 0   
        
    #print("num:",num)
    for i in range(num):
        predict_y = 0
        for j in range(base_num):
            predict_y += x[j]*(data_x[i]**j)
        square_error +=(data_y[i] - predict_y) **2
    print('\nLSE:\n',str(square_error)) 
    square_error /= num
    square_error = np.sqrt(square_error)
    print('\nRMS:\n',str(square_error))   
     
    
    
if __name__=="__main__": 
    data_num = 20
    data_x, data_y = load_data(data_num)
    
    base_num = 10
    #data_x, data_y, data_num = load_data_from_file("C:/Users/hsps9/Desktop/cos.txt")
    factor = 0    
    x = linear_regression(data_x,factor,data_y, data_num, base_num)
    calculate_error(x,data_x, data_y, data_num,base_num)
    newton_x =newton_method(data_x,data_y, data_num, base_num)
    
    #LSE
    LR_X = data_x
    LR_Y = []
    for i in range(data_num):
        predict_y = 0
        for j in range(base_num):
            predict_y += x[j]*(data_x[i]**j)
        LR_Y = np.concatenate((LR_Y, predict_y), axis=0)
        
    #plot   
    plt.figure()
    plt.subplot(1,2,1)
    #plt.show()   
    plt.title('LSE')
    plt.plot(LR_X, LR_Y, color='g', label='fitting line')
    plt.scatter(data_x, data_y, color='r', marker = '^', label='data_point', linewidth=2)
    plt.legend(loc='lower right')
    
    #newton_method
    LR_X = data_x
    LR_Y = []
    for i in range(data_num):
        predict_y = 0
        for j in range(base_num):
            predict_y += x[j]*(data_x[i]**j)
        LR_Y = np.concatenate((LR_Y, predict_y), axis=0)
       
    
    #plot   
    plt.subplot(1,2,2)    
    print("\n方程式係數:\n",newton_x)
    s = "\nfitting line = \n"
    for i in range(base_num-1):
        s += ("%.6lf" % x[i][0]) + "x^" + ("%d" % i) + " + "
    s += ("%.6lf" % x[base_num-1][0]) + "x^" + ("%d" % (base_num-1))
    print(s)       
    calculate_error(newton_x,data_x, data_y, data_num, base_num)
    plt.title('newton_method')
    plt.plot(LR_X, LR_Y, color='g', label='fitting line')
    plt.scatter(data_x, data_y, color='r', marker = '^', label='data_point', linewidth=2)
    plt.legend(loc='lower right')
    plt.show()
    



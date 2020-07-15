
##################################################
## CEC functions implementation for Python
##################################################
## Usage: cec_functions(X, fnum) , X: 1*30 Array, fnum: Function Number
##################################################
## Author: Naeemeh Yadollahpour
## Email:  naeemeh@yorku.ca
## Please refer to our paper in your work:
## ...
##################################################

import numpy as np
import math
import csv

class CEC_functions:
    def __init__(self , dim):
        csv_file = open('extdata/M_D' + str(dim) + '.txt')
        csv_data = csv.reader(csv_file, delimiter=' ')
        csv_data_not_null = [[float(data) for data in row if len(data) > 0] for row in csv_data]
        self.rotate_data = np.array(csv_data_not_null)
        csv_file = open('extdata/shift_data.txt')
        csv_data = csv.reader(csv_file, delimiter=' ')
        self.sd = []
        for row in csv_data:
            self.sd += [float(data) for data in row if len(data) > 0]
        self.M1 = self.read_M(dim , 0)
        self.M2 = self.read_M(dim , 1)
        self.O  = self.shift_data(dim , 0)
        self.aux9_1= np.array([0.5**j for j in range(0,21)])
        self.aux9_2= np.array([3**j for j in range(0,21)])
        self.aux16 = np.array([2**j for j in range(1,33)])

    def read_M(self , dim , m):
        return self.rotate_data[m * dim : (m + 1) * dim]

    def shift_data(self , dim , m):
        return np.array(self.sd[m * dim : (m + 1) * dim])

    def carat(self , dim , alpha):    # I don't know this is correct or not!!!
        return alpha ** (np.arange(dim)/(2*(dim-1)))

    def T_asy(self , X , Y, beta):
        D = len(X)
        for i in range(D):
            if X[i] > 0:
                Y[i] = X[i] ** (1 + beta * (i/(D-1)) * np.sqrt(X[i])  ) 
        pass

    def T_osz (self ,X):
        for i in [0,-1]:
            c1 = 10 if X[i] > 0 else 5.5
            c2 = 7.9 if X[i] > 0 else 3.1
            x_hat = 0 if X[i] == 0 else np.log(abs(X[i]))
            X[i] = np.sign(X[i]) * np.exp (x_hat + 0.049 * (np.sin(c1 * x_hat) + np.sin(c2 * x_hat)))
        pass

    def cf_cal (self, X, delta, bias, fit):
        d = len(X)
        W_star = []
        cf_num = len(fit)

        for i in range(cf_num):
            X_shift = X - self.shift_data(d , i)
            W = 1 / np.sqrt(np.sum(X_shift**2)) * np.exp(-1 * np.sum(X_shift**2) / (2 * d * delta[i]**2))
            W_star.append(W)

        if (np.max(W_star) == 0):
            W_star = [1] * cf_num

        omega = W_star/np.sum(W_star) * (fit + bias)

        return np.sum(omega)
        
    def Y(self , X , fun_num, rflag = None):
        if rflag is None:
            rf = [0,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,0,1,1,1,1,0,1,1,1,1,1,1][fun_num - 1]
        else:
            rf = rflag

        # Unimodal Functions
        # Sphere Function
        # 1    
        if fun_num == 1:
            Z = X - self.O
            if rf == 1:
                Z = self.M1 @ Z
            Y = np.sum(Z**2) - 1400


        # Rotated High Conditioned Elliptic Function
        # 2       
        elif fun_num == 2:
            d = len(X)
            X_shift = X - self.O
            X_rotate = self.M1 @ X_shift
            self.T_osz(X_rotate)
            Y = np.sum((1e6 ** (np.arange(d)/(d-1))) * X_rotate**2) - 1300
        

        # Rotated Bent Cigar Function
        # 3
        elif fun_num == 3:
            X_shift = X - self.O
            X_rotate = self.M1 @ X_shift
            self.T_asy(X_rotate ,X_shift, 0.5)
            Z = self.M2 @ X_shift

            Y = Z[0]**2 + 1e6 * np.sum(Z[1:]**2) - 1200


        # Rotated Discus Function
        # 4
        elif fun_num == 4:
            d = len(X)
            X_shift = X - self.O
            X_rotate = self.M1 @ X_shift
            self.T_osz(X_rotate)
            Y = (1e6) * (X_rotate[0]**2) + np.sum(X_rotate[1:d]**2) - 1100


        # Different Powers Function
        # 5
        elif fun_num == 5:
            d = len(X)
            Z = X - self.O
            if rf == 1:
                Z = self.M1 @ Z
            Y = np.sqrt(np.sum(abs(Z)**(2 + (4 * np.arange(d)/(d-1)).astype(int)))) - 1000

        # Basic Multimodal Functions
        # Rotated Rosenbrock’s Function
        # 6
        elif fun_num == 6:
            d = len(X)
            X_shift = X - self.O
            X_rotate = self.M1 @ ((2.048 * X_shift)/100)
            Z = X_rotate + 1
            Y = np.sum (100 * (Z[:d-1]**2 - Z[1:d])**2 + (Z[:d-1] - 1)**2) - 900


        # Rotated Schaffers F7 Function
        # 7
        elif fun_num == 7:
            d = len(X)
            X_shift = X - self.O
            X_rotate = self.M1 @ X_shift
            self.T_asy(X_rotate , X_shift, 0.5)
            Z = self.M2 @ (self.carat(d , 10) * X_shift)

            Z = np.sqrt(Z[:-1]**2 + Z[1:]**2)
            Y = (np.sum(np.sqrt(Z) + np.sqrt(Z) * np.sin(50 * Z**0.2)**2 ) / (d-1))**2 - 800

        # Rotated Ackley’s Function
        # 8
        elif fun_num == 8:
            d = len(X)
            X_shift = X - self.O
            X_rotate = self.M1 @ X_shift
            self.T_asy(X_rotate , X_shift, 0.5)
            Z = self.M2 @ (self.carat(d , 10) * X_shift)

            Y = -20 * np.exp(-0.2 * np.sqrt(np.sum(Z**2)/d)) \
                - np.exp(np.sum(np.cos(2 * np.pi * Z))/d) \
                + 20 + np.exp(1) - 700


        # Rotated Weierstrass Function
        # 9
        elif fun_num == 9:
            d = len(X)
            X_shift = 0.005 * (X - self.O)
            X_rotate_1 = self.M1 @ ( X_shift)
            self.T_asy(X_rotate_1 ,X_shift, 0.5)
            Z = self.M2 @ (self.carat(d , 10) * X_shift)    

            # kmax = 20

            def _w(v):
                return np.sum(self.aux9_1 * np.cos(2.0 * np.pi * self.aux9_2 * v))

            Y = np.sum([_w(Z[i] + 0.5) for i in range(d)]) - (_w(0.5) * d) - 600


        # Rotated Griewank’s Function
        # 10
        elif fun_num == 10:
            d = len(X)
            X_shift = (600.0 * (X - self.O))/100.0
            X_rotate = self.M1 @  X_shift
            Z   = self.carat(d , 100) * X_rotate  # used carat as matrix not sure though!!!
 
            Y = 1.0 + (np.sum(Z**2 )/4000.0) - np.multiply.reduce(np.cos(Z / np.sqrt(np.arange(1, d + 1)))) - 500


        # Rastrigin’s Function
        # 11
        elif fun_num == 11:
            d = len(X)
            X_shift = 0.0512 * (X - self.O)
            if rf == 1:
                X_shift = self.M1 @ X_shift
            X_osz   = X_shift.copy()
            self.T_osz(X_osz)
            self.T_asy(X_osz ,X_shift , 0.2)
            if rf == 1:
                X_shift = self.M2 @ X_shift
            Z = self.carat(d, 10) * X_shift
            
            Y = np.sum(10 + Z**2 - 10 * np.cos(2 * np.pi * Z)) - 400

        # Rotated Rastrigin’s Function
        # 12
        elif fun_num == 12:
            d = len(X)
            X_shift = 0.0512 * (X - self.O)
            X_rotate =  self.M1 @ X_shift
            X_hat = X_rotate.copy()
            self.T_osz(X_hat)
            self.T_asy(X_hat , X_rotate , 0.2)
            X_rotate = self.M2 @ X_rotate
            X_rotate = self.carat(d , 10) * X_rotate
            Z = self.M1 @ X_rotate

            Y = np.sum(10 + Z **2 - 10 * np.cos(2 * np.pi * Z)) - 300


        # Non-continuous Rotated Rastrigin’s Function
        # 13
        elif fun_num == 13:
            d = len(X)
            X_shift = 0.0512 * (X - self.O)
            X_hat = self.M1 @ X_shift

            X_hat[abs(X_hat) > 0.5] = np.round(X_hat[abs(X_hat) > 0.5] * 2) / 2
            Y_hat = X_hat.copy()
            self.T_osz(X_hat)
            self.T_asy(X_hat , Y_hat ,  0.2)
            X_rotate = self.M2 @ Y_hat
            X_carat  = self.carat(d,10) * X_rotate
            Z = self.M1 @ X_carat

            Y = np.sum(10 + Z**2 - 10 * np.cos(2 * np.pi * Z)) - 200

        # Schwefel’s Function
        # 14 15    
        elif fun_num == 14 or fun_num == 15:
            d = len(X)
            X_shift = 10 * (X - self.O)
            if rf:
                X_shift = self.M1 @ X_shift
            Z = self.carat(d , 10) * X_shift + 420.9687462275036
            Z[abs(Z) <=500] = \
                Z[abs(Z) <=500] * np.sin(np.sqrt(abs(Z[abs(Z) <=500])))
            Z[Z > 500]      = \
                (500 - Z[Z > 500] % 500) * np.sin(np.sqrt(500 - Z[Z > 500] % 500)) \
                - (Z[Z > 500] - 500)**2 / (10000*d)
            Z[Z < -500]     = \
                (abs(Z[Z < -500]) % 500 - 500) * np.sin(np.sqrt(500 - abs(Z[Z < -500]) % 500)) \
                - (Z[Z < -500] + 500)**2 / (10000*d)
            Y = 418.9828872724338 * d - np.sum(Z) + (-100 if fun_num == 14 else 100)

        # Rotated Katsuura Function
        # 16
        elif fun_num == 16:
            d = len(X)
            X_shift = 0.05 * (X - self.O)
            X_rotate = self.M1 @ X_shift
            X_carat = self.carat(d , 100) * X_rotate
            Z = self.M2 @ X_carat

            def _kat(c):
                return np.sum(np.abs(self.aux16*c - np.round(self.aux16*c)) / self.aux16)

            for i in range(d):
                Z[i] = (1 + (i+1) * _kat(Z[i]))
            
            Z = np.multiply.reduce(Z**(10/d**1.2))
            Y = (10/d**2) * Z - (10/d**2) + 200


        # bi-Rastrigin Function
        # 17 18
        elif fun_num == 17 or fun_num == 18:
            d    = len(X)
            mu_0 = 2.5
            S    = 1 - 1/((2 * np.sqrt(d + 20)) - 8.2)
            mu_1 = -1 * np.sqrt((mu_0**2 - 1) / S)
            X_star  = self.O
            X_shift = 0.1 * (X - self.O)
            X_hat   = []
            for i in range(d):
                X_hat.append( 2 * np.sign(X_star[i]) * X_shift[i] + mu_0 )

            MU_0 = np.ones(30) * mu_0
            Z    = X_hat - MU_0
            if rf:
                Z = self.M1 @ Z
            Z    = self.carat(d , 100) * Z
            if rf:
                Z = self.M2 @ Z
            
            Y_1 = []
            Y_2 = []
            for i in range(d):
                Y_1.append((X_hat[i] - mu_0)**2)
                Y_2.append((X_hat[i] - mu_1)**2)

            Y_3 = np.minimum(np.sum(Y_1) , d + S * np.sum(Y_2))
            Y   = Y_3 + 10 * (d - np.sum(np.cos(2 * np.pi * Z))) + (300 if fun_num == 17 else 400)

        # Rotated Expanded Griewank’s plus Rosenbrock’s Function
        # 19    
        elif fun_num == 19:
            d = len(X)
            X_shift = 0.05 * (X - self.O) + 1

            tmp  = X_shift**2 - np.roll(X_shift,-1)
            tmp  = 100 * tmp**2 + (X_shift - 1)**2
            Z    = np.sum(tmp**2/4000 - np.cos(tmp) + 1)

            Y = Z + 500
           
        # Rotated Expanded Scaffer’s F6 Function
        # 20
        elif fun_num == 20:
            d = len(X)
            X_shift  = X - self.O
            X_rotate = self.M1 @ X_shift
            self.T_asy( X_rotate , X_shift , 0.5)
            Z = self.M2 @ X_shift

            tmp1 = Z**2 + (np.roll(Z,-1))**2

            Y = np.sum(0.5 + (np.sin(np.sqrt(tmp1))**2 - 0.5)/(1 + 0.001 * tmp1)**2) + 600

        # New Composition Functions
        # Composition Function 1
        # 21
        elif fun_num == 21:
            d = len(X)
            delta = np.array([10 , 20 , 30 , 40 , 50])
            bias  = np.array([0 , 100, 200, 300, 400])
            fit = []
            fit.append( (self.Y(X , 6 , rf) + 900)/1     )
            self.O  = self.shift_data(d, 1)
            self.M1 = self.read_M(d, 1) 
            self.M2 = self.read_M(d, 2) 
            fit.append( (self.Y(X , 5 , rf) + 1000)/1e6  )
            self.O = self.shift_data(d, 2)
            self.M1 = self.read_M(d, 2) 
            self.M2 = self.read_M(d, 3) 
            fit.append( (self.Y(X , 3 , rf) + 1200)/1e26 )
            self.O = self.shift_data(d, 3)
            self.M1 = self.read_M(d, 3) 
            self.M2 = self.read_M(d, 4) 
            fit.append( (self.Y(X , 4 , rf) + 1100)/1e6  )
            self.O = self.shift_data(d, 4)
            self.M1 = self.read_M(d, 4) 
            self.M2 = self.read_M(d, 5) 
            fit.append( (self.Y(X , 1 , rf) + 1400)/1e1  )

            Y= self.cf_cal(X, delta, bias, np.array(fit)) + 700

        elif fun_num == 22:
            d = len(X)
            delta = np.array([20 , 20 , 20])
            bias  = np.array([0 , 100, 200])
            fit   = []
            fit.append( (self.Y(X , 14 , rf) + 100)/1     )
            self.O  = self.shift_data(d, 1)
            self.M1 = self.read_M(d, 1) 
            self.M2 = self.read_M(d, 2) 
            fit.append( (self.Y(X , 14 , rf) + 100)/1     )
            self.O  = self.shift_data(d, 2)
            self.M1 = self.read_M(d, 2) 
            self.M2 = self.read_M(d, 3) 
            fit.append( (self.Y(X , 14 , rf) + 100)/1     )

            Y = Y= self.cf_cal(X, delta, bias, np.array(fit)) + 800

        elif fun_num == 23:
            d = len(X)
            delta = np.array([20 , 20 , 20])
            bias  = np.array([0 , 100, 200])
            fit   = []
            fit.append( (self.Y(X , 15 , rf) - 100)/1     )
            self.O  = self.shift_data(d, 1)
            self.M1 = self.read_M(d, 1) 
            self.M2 = self.read_M(d, 2) 
            fit.append( (self.Y(X , 15 , rf) - 100)/1     )
            self.O  = self.shift_data(d, 2)
            self.M1 = self.read_M(d, 2) 
            self.M2 = self.read_M(d, 3) 
            fit.append( (self.Y(X , 15 , rf) - 100)/1     )

            Y =  self.cf_cal(X, delta, bias, np.array(fit)) + 900
     
        elif fun_num == 24:
            d = len(X)
            delta = np.array([20 , 20 , 20])
            bias  = np.array([0 , 100, 200])
            fit   = []
            fit.append( (self.Y(X , 15 , rf) - 100) * 0.25   )
            self.O  = self.shift_data(d, 1)
            self.M1 = self.read_M(d, 1) 
            self.M2 = self.read_M(d, 2)
            fit.append( (self.Y(X , 12 , rf)  + 300) * 1   )
            self.O  = self.shift_data(d, 2)
            self.M1 = self.read_M(d, 2) 
            self.M2 = self.read_M(d, 3) 
            fit.append( (self.Y(X , 9 , rf)  + 600) * 2.5   )

            Y = self.cf_cal(X, delta, bias, np.array(fit)) + 1000

        elif fun_num == 25:
            d = len(X)
            delta = np.array([10 , 30 , 50])
            bias  = np.array([0 , 100, 200])
            fit   = []
            fit.append( (self.Y(X , 15 , rf) - 100) * 0.25   )
            self.O  = self.shift_data(d, 1)
            self.M1 = self.read_M(d, 1) 
            self.M2 = self.read_M(d, 2)
            fit.append( (self.Y(X , 12 , rf)  + 300) * 1   )
            self.O  = self.shift_data(d, 2)
            self.M1 = self.read_M(d, 2) 
            self.M2 = self.read_M(d, 3) 
            fit.append( (self.Y(X , 9 , rf)  + 600) * 2.5   )

            Y = self.cf_cal(X, delta, bias, np.array(fit)) + 1100

        elif fun_num == 26:
            d = len(X)
            delta = np.array([10 , 10 , 10 , 10, 10])
            bias  = np.array([0 , 100, 200 , 300 , 400])
            fit   = []
            fit.append( (self.Y(X , 15 , rf) - 100) * 0.25   )
            self.O  = self.shift_data(d, 1)
            self.M1 = self.read_M(d, 1) 
            self.M2 = self.read_M(d, 2)
            fit.append( (self.Y(X , 12 , rf)  + 300) * 1   )
            self.O  = self.shift_data(d, 2)
            self.M1 = self.read_M(d, 2) 
            self.M2 = self.read_M(d, 3)
            fit.append( (self.Y(X , 2 , rf)  + 1300) / 1e7   )
            self.O  = self.shift_data(d, 3)
            self.M1 = self.read_M(d, 3) 
            self.M2 = self.read_M(d, 4)
            fit.append( (self.Y(X , 9 , rf)  + 600) * 2.5   )
            self.O  = self.shift_data(d, 4)
            self.M1 = self.read_M(d, 4) 
            self.M2 = self.read_M(d, 5)
            fit.append( (self.Y(X , 10 , rf)  + 500) * 10   )

            Y = self.cf_cal(X, delta, bias, np.array(fit)) + 1200
        elif fun_num == 27:
            d = len(X)
            delta = np.array([10 , 10 , 10 , 20, 20])
            bias  = np.array([0 , 100, 200 , 300 , 400])
            fit   = []
            fit.append( (self.Y(X , 10 , rf)  + 500) * 100   )
            self.O  = self.shift_data(d, 1)
            self.M1 = self.read_M(d, 1) 
            self.M2 = self.read_M(d, 2)
            fit.append( (self.Y(X , 12 , rf)  + 300) * 10  )
            self.O  = self.shift_data(d, 2)
            self.M1 = self.read_M(d, 2) 
            self.M2 = self.read_M(d, 3)
            fit.append( (self.Y(X , 15 , rf) - 100) * 2.5   )
            self.O  = self.shift_data(d, 3)
            self.M1 = self.read_M(d, 3) 
            self.M2 = self.read_M(d, 4)
            fit.append( (self.Y(X , 9 , rf) + 600) * 25   )
            self.O  = self.shift_data(d, 4)
            self.M1 = self.read_M(d, 4) 
            self.M2 = self.read_M(d, 5)
            fit.append( (self.Y(X , 1 , rf) + 1400) / 1e1  )

            Y = self.cf_cal(X, delta, bias, np.array(fit)) + 1300

        elif fun_num == 28:
            d = len(X)
            delta = np.array([10 , 20 , 30 , 40, 50])
            bias  = np.array([0 , 100, 200 , 300 , 400])
            fit   = []
            fit.append( (self.Y(X , 19 , rf) - 500) * 2.5   )
            self.O  = self.shift_data(d, 1)
            self.M1 = self.read_M(d, 1) 
            self.M2 = self.read_M(d, 2)
            fit.append( ((self.Y(X , 7 , rf) + 800) * 2.5) / 1e3  )
            self.O  = self.shift_data(d, 2)
            self.M1 = self.read_M(d, 2) 
            self.M2 = self.read_M(d, 3)
            fit.append( (self.Y(X , 15 , rf) - 100) * 2.5   )
            self.O  = self.shift_data(d, 3)
            self.M1 = self.read_M(d, 3) 
            self.M2 = self.read_M(d, 4)
            fit.append( ((self.Y(X , 20 , rf) - 600) * 5) / 1e4   )
            self.O  = self.shift_data(d, 4)
            self.M1 = self.read_M(d, 4) 
            self.M2 = self.read_M(d, 5)
            fit.append( (self.Y(X , 1 , rf) + 1400) / 1e1  )

            Y = self.cf_cal(X, delta, bias, np.array(fit)) + 1400

        return Y

if __name__ == "__main__":
    f_num = 9
    cec_functions = CEC_functions(30)

    X = np.ones(30)

    #C Calculations
    # import cic13functions
    # C_Y = np.longdouble(cic13functions.run(str(f_num) + ',' + str(list(X))[1:-1]))

    #Python Calculations
    P_Y = cec_functions.Y(X , f_num)

    # print('c response:', C_Y )
    print('python response:' , P_Y)
    pass
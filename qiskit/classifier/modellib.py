# -*- coding: utf-8 -*-
"""
Welcome to Noah's Ark
"""
import numpy as np
import random
import math
import pickle

import matplotlib.pyplot as plt
from IPython import display

from scipy.optimize import minimize
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute
from qiskit.circuit.library import PauliFeatureMap, EfficientSU2

class mymodel:
    def __init__(self):
        '''
        Function:
            get_parameters()
            fit(x, y, z, x0=[], method='trust-constr', tol=1e-6)
            predict(x, y)
            calculate_cost_function(parameters, mode=False,x=[0],y=[0],z=[0])
        '''
        self.__parameters = []
        self.__x = []
        self.__y = []
        self.__z = []
        self.__itertimes = [] #迭代次数
        self.__losscurve = [] #损失曲线
    def get_parameters(self):
        return self.__parameters
    def fit(self, x, y, z, x0=[], method='trust-constr', tol=1e-6):
        '''
        Args:
            x (numpy.array[float])
            y (numpy.array[float])
            z (numpy.array[float])
            x0 (list[float]): 变量参数列表的初始猜测值 [a, b, c, d, e]，默认随机
            method (String): scipy.optimize.minimize优化器求极值的方法，默认为"trust-constr"
                常用："BFGS"、"COBYLA"、"Nelder-Mead"、"CG"
            tol (float): Tolerance for termination. 
        Returns:
            True if the model can be trained and False otherwise.
        '''
        if len(z) == 0:
            raise ValueError('Dataset is not standardized')
        if len({len(x),len(y),len(z)}) != 1:
            raise ValueError('Lengths of dataset are not equal')
        self.__x, self.__y, self.__z = x, y, z
        if x0 == []:
            x0 = [random.uniform(-1,1) for p in range(5)]
        try:
            self.__itertimes = []
            self.__losscurve = []
            out = minimize(self.calculate_cost_function, x0=x0, method=method, options={'maxiter':200}, tol=tol)
            self.__parameters = out['x']
            plt.show()
        except:
            return False
        return True
    def predict(self, x, y):
        '''
        Args:
            x (numpy.array[float])
            y (numpy.array[float])
        Returns:
            z (list[float])
        '''
        if len(x) != len(y):
            raise ValueError('Lengths of dataset are not equal')
        ansatz = np.array(self.__parameters[:-1])
        ansatzNorm = np.linalg.norm(ansatz)
        ansatz = ansatz/ansatzNorm
        res = []
        for i in range(len(x)):
            encoder = np.array([x[i]**2, y[i]**2, x[i], y[i]])
            encoderNorm = np.linalg.norm(encoder)
            encoder = encoder/encoderNorm
            z_ansatz = ansatzNorm * encoderNorm * self.__inner_prod(encoder,ansatz)
            res.append(z_ansatz+self.__parameters[-1])
        res = np.round(res)
        res /= np.linalg.norm(res)
        return np.round((res+1)/2)
    def __draw_loss(self, loss): #绘制损失曲线
        self.__itertimes.append(len(self.__itertimes))
        self.__losscurve.append(loss)
        plt.cla()
        plt.plot(self.__itertimes, self.__losscurve, color='black')  # 绘制曲线
        display.clear_output(wait=True)
        plt.pause(0.001)
    def calculate_cost_function(self, parameters, mode=False,x=[0],y=[0],z=[0]): #损失函数
        '''
        Args:
            parameters (list[float]): 参数列表[a, b, c, d, e]
            mode (bool): 工作模式，True时使用训练完成的参数计算损失函数
            x (numpy.array[float])
            y (numpy.array[float])
            z (numpy.array[float])
        Returns:
            loss (float): 损失值
        '''
        if mode:
            parameters = self.__parameters
            self.__x, self.__y, self.__z = x, y, z
        #[x**2,y**2,x,y][a,b,c,d]
        ansatz = np.array(parameters[:-1]) #拟设
        ansatzNorm = np.linalg.norm(ansatz)
        ansatz = ansatz/ansatzNorm
        loss = 0
        for i in range(len(self.__x)):
            encoder = np.array([self.__x[i]**2, self.__y[i]**2, self.__x[i], self.__y[i]]) #输入信息编码
            encoderNorm = np.linalg.norm(encoder)
            encoder = encoder/encoderNorm
            z_ansatz = ansatzNorm * encoderNorm * self.__inner_prod(encoder,ansatz)
            loss += (z_ansatz+parameters[-1]-self.__z[i])**2
        loss /= len(self.__z)
        #print(loss)
        if not mode:
            self.__draw_loss(loss)
        return loss
    def __inner_prod(self, vec1, vec2): #量子线路计算向量内积
        '''
        Args:
            vec1 (numpy.array[float]): 向量1，length=4
            vec2 (numpy.array[float]): 向量2，length=4
        Returns:
            vec1*vec2 (float): 向量内积
        '''
        if len(vec1) != len(vec2):
            raise ValueError('Lengths of states are not equal')
        vec = np.concatenate((vec1,vec2))/np.sqrt(2)
        
        circ = QuantumCircuit(3)
        circ.initialize(vec, range(3))
        circ.h(2)

        backend = Aer.get_backend('statevector_simulator')
        job = execute(circ, backend, backend_options = {"zero_threshold": 1e-20})
        result = job.result()
        o = np.real(result.get_statevector(circ))
        m_sum = 0
        for l in range(2**2):
            m_sum += o[l]**2
        return (2*m_sum-1)
        
        
class mymodel2:
    def __init__(self, parameters=None):
        '''
        Function:
            get_parameters()
            fit(x, y, z, x0=[], method='trust-constr', tol=1e-6)
            predict(x, y)
            score(x, y, z)
            draw(bd=2, prec=0.1)
            save(path='', name='mymodel2')
            calculate_cost_function(parameters, mode=False,x=[0],y=[0],z=[0])
        '''
        if parameters is not None:
            self.__parameters = parameters
        else:
            self.__parameters = []
        self.__x = []
        self.__y = []
        self.__z = []
        self.__itertimes = [] #迭代次数
        self.__losscurve = [] #损失曲线
    def get_parameters(self):
        return self.__parameters
    def fit(self, x, y, z, x0=[], method='trust-constr', tol=1e-6):
        '''
        Args:
            x (numpy.array[float])
            y (numpy.array[float])
            z (numpy.array[float])
            x0 (list[float]): 变量参数列表的初始猜测值 [a, b, c, d, e]，默认随机
            method (String): scipy.optimize.minimize优化器求极值的方法，默认为"trust-constr"
                常用："BFGS"、"COBYLA"、"Nelder-Mead"、"CG"
            tol (float): Tolerance for termination. 
        Returns:
            True if the model can be trained and False otherwise.
        '''
        if len(z) == 0:
            raise ValueError('Dataset is not standardized')
        if len({len(x),len(y),len(z)}) != 1:
            raise ValueError('Lengths of dataset are not equal')
        self.__x, self.__y, self.__z = x, y, z
        if x0 == []:
            x0 = [random.uniform(-np.pi,np.pi) for p in range(5)]
            x0 += [random.uniform(-1,1)]
        try:
            self.__itertimes = []
            self.__losscurve = []
            out = minimize(self.calculate_cost_function, x0=x0, method=method, options={'maxiter':200}, tol=tol)
            self.__parameters = out['x']
            plt.show()
        except:
            return False
        return True
    def predict(self, x, y):
        '''
        Args:
            x (numpy.array[float])
            y (numpy.array[float])
        Returns:
            z (numpy.array[float])
        '''
        return np.round(self.__predict_probability(x, y))
    def score(self, x, y, z):
        '''
        Args:
            x (numpy.array[float])
            y (numpy.array[float])
            z (numpy.array[float])
        Returns:
            score (float): 准确率
        '''
        return (self.predict(x, y)==z).sum()/len(z)
    def calculate_cost_function(self, parameters, mode=False,x=[0],y=[0],z=[0]): #损失函数
        '''
        Args:
            parameters (list[float]): 参数列表[a, b, c, d, e, f]
            mode (bool): 工作模式，True时使用训练完成的参数计算损失函数
            x (numpy.array[float])
            y (numpy.array[float])
            z (numpy.array[float])
        Returns:
            loss (float): 损失值
        '''
        if mode:
            parameters = self.__parameters
            self.__x, self.__y, self.__z = x, y, z
        loss = 0
        for i in range(len(self.__x)):
            #输入信息编码
            encoder = np.array([np.linalg.norm(np.array([self.__x[i], self.__y[i]])), parameters[-1]])
            encoder /= np.linalg.norm(encoder)
            z_ansatz = self.__qvc(encoder, parameters[:-1])
            loss += (z_ansatz - self.__z[i])**2
        loss /= len(self.__z)
        if not mode:
            #print(loss)
            self.__draw_loss(loss)
        return loss
    def draw(self, bd=2, prec=0.1): #绘制灰度图
        '''
        Args:
            bd (float): boundary
            prec (float): precision
        '''
        X0 = np.arange(-bd, bd, prec)
        Y0 = np.arange(-bd, bd, prec)
        X0, Y0 = np.meshgrid(X0, Y0)
        Z1 = []
        for i in range(len(X0)):
            Z1.append(self.__predict_probability(X0[i],Y0[i]))
        plt.contourf(X0, Y0, Z1, cmap=plt.cm.gray)
    def save(self, path='', name='mymodel2'): #保存模型
        with open (path+name+'.noe', 'wb') as f:
            pickle.dump(self, f)
    def __predict_probability(self, x, y): #预测概率幅
        if len(x) != len(y):
            raise ValueError('Lengths of dataset are not equal')
        res = []
        for i in range(len(x)):
            #输入信息编码
            encoder = np.array([np.linalg.norm(np.array([x[i], y[i]])), self.__parameters[-1]])
            encoder /= np.linalg.norm(encoder)
            z_ansatz = self.__qvc(encoder, self.__parameters[:-1])
            res.append(z_ansatz)
        return res
    def __qvc(self, encoder, ansatz, shots=1000): #Quantum Variational Circuit
        qreg_q = QuantumRegister(1, 'q')
        creg_c = ClassicalRegister(1, 'c')
        circuit = QuantumCircuit(qreg_q, creg_c)
        
        circuit.initialize(encoder, range(1))
        circuit.h(qreg_q[0])
        circuit.rx(ansatz[0], qreg_q[0])
        circuit.ry(ansatz[1], qreg_q[0])
        circuit.rz(ansatz[2], qreg_q[0])
        circuit.rx(ansatz[3], qreg_q[0])
        circuit.ry(ansatz[4], qreg_q[0])
        
        simulator = Aer.get_backend('qasm_simulator')
        circuit.measure([0],[0])
        job = execute(circuit, simulator, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        return counts.get('1',0)/shots
    def __draw_loss(self, loss): #绘制损失曲线
        self.__itertimes.append(len(self.__itertimes))
        self.__losscurve.append(loss)
        plt.cla()
        plt.plot(self.__itertimes, self.__losscurve, color='black')  # 绘制曲线
        display.clear_output(wait=True)
        plt.pause(0.001)
        
        
class mymodel3:
    def __init__(self, parameters=None, circuit=None):
        '''
        Function:
            get_parameters()
            fit(x, y, z, x0, method='trust-constr', tol=1e-6)
            predict(x, y)
            score(x, y, z)
            draw(bd=2, prec=0.1)
            save(path='', name='mymodel3')
            calculate_cost_function(parameters, mode=False,x=[0],y=[0],z=[0])
        '''
        self.__parameters = parameters or [] #参数列表
        self.__classifier = circuit or PauliFeatureMap(2, reps=1, paulis=['ZZ']).compose(EfficientSU2(2)) #量子线路
        self.__simulator = Aer.get_backend('qasm_simulator') #模拟器
        self.__x = []
        self.__y = []
        self.__z = []
        self.__itertimes = [] #迭代次数
        self.__losscurve = [] #损失曲线
    def get_parameters(self):
        return self.__parameters
    def fit(self, x, y, z, x0, method='trust-constr', tol=1e-6):
        '''
        Args:
            x (numpy.array[float])
            y (numpy.array[float])
            z (numpy.array[float])
            x0 (list[float]): 变量参数列表的初始猜测值，默认随机
            method (String): scipy.optimize.minimize优化器求极值的方法，默认为"trust-constr"
                常用："BFGS"、"COBYLA"、"Nelder-Mead"、"CG"
            tol (float): Tolerance for termination. 
        Returns:
            True if the model can be trained and False otherwise.
        '''
        if len(z) == 0:
            raise ValueError('Dataset is not standardized')
        if len({len(x),len(y),len(z)}) != 1:
            raise ValueError('Lengths of dataset are not equal')
        self.__x, self.__y, self.__z = x, y, z
        try:
            self.__itertimes = []
            self.__losscurve = []
            out = minimize(self.calculate_cost_function, x0=x0, method=method, options={'maxiter':200}, tol=tol)
            self.__parameters = out['x']
            plt.show()
        except:
            return False
        return True
    def predict(self, x, y):
        '''
        Args:
            x (numpy.array[float])
            y (numpy.array[float])
        Returns:
            z (numpy.array[float])
        '''
        return np.round(self.__predict_probability(x, y, self.__parameters))
    def score(self, x, y, z):
        '''
        Args:
            x (numpy.array[float])
            y (numpy.array[float])
            z (numpy.array[float])
        Returns:
            score (float): 准确率
        '''
        return (self.predict(x, y)==z).sum()/len(z)
    def calculate_cost_function(self, parameters, mode=False,x=[0],y=[0],z=[0]): #损失函数
        '''
        Args:
            parameters (list[float]): 参数列表[a, b, c, d, e, f]
            mode (bool): 工作模式，True时使用训练完成的参数计算损失函数
            x (numpy.array[float])
            y (numpy.array[float])
            z (numpy.array[float])
        Returns:
            loss (float): 损失值
        '''
        if mode:
            parameters = self.__parameters
            self.__x, self.__y, self.__z = x, y, z
        z_ansatz = self.__predict_probability(self.__x, self.__y, parameters)
        loss = ((z_ansatz - self.__z)**2).sum()
        loss /= len(self.__z)
        if not mode:
            #print(loss)
            self.__draw_loss(loss)
        return loss
    def draw(self, bd=2, prec=0.1): #绘制灰度图
        '''
        Args:
            bd (float): boundary
            prec (float): precision
        '''
        X0 = np.arange(-bd, bd, prec)
        Y0 = np.arange(-bd, bd, prec)
        X0, Y0 = np.meshgrid(X0, Y0)
        Z1 = []
        for i in range(len(X0)):
            Z1.append(self.__predict_probability(X0[i], Y0[i], list(self.__parameters)))
        plt.contourf(X0, Y0, Z1, cmap=plt.cm.gray)
    def save(self, path='', name='mymodel3'): #保存模型
        with open (path+name+'.noe', 'wb') as f:
            pickle.dump(self, f)
    def __predict_probability(self, x, y, parm): #预测概率幅
        if len(x) != len(y):
            raise ValueError('Lengths of dataset are not equal')
        res = []
        for i in range(len(x)):
            res.append(self.__qnn(x[i], y[i], list(parm)))
        return res
    def __qnn(self, x, y, parameters, shots=1000): #Quantum Neural Network
        ccl = self.__classifier.bind_parameters(np.array([x,y]+parameters))
        circuit = QuantumCircuit(2)
        circuit.append(ccl, range(2))
        circuit.measure_all()
        job = execute(circuit, self.__simulator, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        return counts.get('11',0)/shots
    def __draw_loss(self, loss): #绘制损失曲线
        self.__itertimes.append(len(self.__itertimes))
        self.__losscurve.append(loss)
        plt.cla()
        plt.plot(self.__itertimes, self.__losscurve, color='black')  # 绘制曲线
        display.clear_output(wait=True)
        plt.pause(0.001)



class Model_CCBC: #Concentric Circle Binary Classification Model based on Gaussian Function
    def __init__(self, omega=0.1):
        self.cof = [0, 0, 0, 0, 0] #二元二次函数系数
        self.omega = omega # o = min(lim l->0)  会影响到训练结果，本题暂不作分析
    def fit(self, xData, yData, label): #训练
        '''
        Args:
            xData (list[float]): X coordinate
            yData (list[float]): X coordinate
            label (list[0|1]) : Classification label
        Returns:
            True if the model can be trained and False otherwise.
        Raises:
            ValueError: If the length of the input array is not equal.
            ValueError: There are unexpected labels.
        '''
        if len({len(xData), len(yData), len(label)}) != 1:
            raise ValueError('输入数组长度不等')
        if set(label) != {0, 1}:
            raise ValueError('输入标签异常')
        xData = np.array(xData, dtype=float)
        yData = np.array(yData, dtype=float)
        lData = np.array([1 if i==1 else self.omega for i in label], dtype=float)
        fData = np.log(lData)
        try:
            self.cof = self.__polyFit(xData, yData, fData)
        except:
            return False
        return True
    def predict(self, xData, yData): #预测
        '''
        Args:
            xData (list[float]): X coordinate
            yData (list[float]): X coordinate
        Returns:
            predict_label (np.array[0|1]) : Prediction results
        '''
        xData, yData = np.array(xData), np.array(yData)
        return np.array(self.__fun_gauss(xData, yData, self.cof)*2, dtype=int).clip(0, 1)
    def score(self, xData, yData, label): #评估模型性能
        '''
        Args:
            xData (list[float]): X coordinate
            yData (list[float]): X coordinate
            label (list[0|1]) : Classification label
        Returns:
            ???
        '''
        pass
    def get_parameters(self): #获取模型参数
        #exp(ax^2+by^2+cx+dy+e)
        return [self.cof[3], self.cof[4], self.cof[1], self.cof[2], self.cof[0]]
    def generate_contour_map(self, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), samp=0.01, show=True): #生成等高线图
        '''
        Args:
            xlim (list[float]): X-axis upper and lower limits
            ylim (list[float]): Y-axis upper and lower limits
            samp (float) : Sampling interval
            show (bool)  : Function has 'plt.show()' if set True.
        '''
        X0 = np.arange(xlim[0], xlim[1], samp)
        Y0 = np.arange(ylim[0], ylim[1], samp)
        X0, Y0 = np.meshgrid(X0, Y0)
        Z0 = self.__fun_gauss(X0, Y0, self.cof)
        plt.figure(figsize=(8, 8))
        plt.contourf(X0, Y0, Z0, cmap=plt.cm.gray)
        C = plt.contour(X0, Y0, Z0, 8, colors='r')
        mid = int((len(C.levels)-2)/2)
        if mid >1 and mid < len(C.levels)-2:
            levels = [C.levels[1], C.levels[mid], C.levels[-2]]
        else:
            levels = [C.levels[1], C.levels[-2]]
        plt.clabel(C,levels=levels,inline=1,fontsize=10)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Contour Map')
        if show:
            plt.show()
    def __fun_gauss(self, x, y, cof): #高斯函数核
        return np.exp(cof[3]*np.power(x, 2) + cof[4]*np.power(y, 2) + cof[1]*x + cof[2]*y + cof[0])
    def __polyFit(self, xData, yData, fData): #最小二乘法求解高斯曲面函数参数
        n = 5
        s = np.zeros((n, len(xData)))
        s[0] = np.ones(len(xData))
        s[1] = xData
        s[2] = yData
        s[3] = xData**2
        s[4] = yData**2
        return self.__polyfit_mx(n, fData, s)
    def __polyfit_mx(self, n, fData, s):
        a = np.zeros((n,n))
        b = np.zeros(n)
        for i in range(n):
            a[i] = (s * s[i]).sum(axis=1)
            b[i] = (fData * s[i]).sum()
        return np.linalg.solve(a, b)

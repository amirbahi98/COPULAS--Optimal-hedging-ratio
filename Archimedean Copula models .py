#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import scipy.integrate
import pandas as pd

from scipy.special  import comb
from scipy.stats    import norm
from scipy.optimize import root
from scipy.optimize import newton

import numpy.random as npr
import scipy.stats as scs

import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Data contains Simulated Equity Returns 
Returns = pd.read_excel ('C:/Users/click\Desktop/PYTHON-GOLDD.xlsx', index_col=False)
print (Returns.head(2))
print (Returns.tail(2))
size = len(Returns)


# In[10]:


symbols = ['S', 'F']

plt.figure(figsize=(12, 10)) 

val = 0 
plt.subplot(211)
Returns[symbols[val]].hist(bins=70, density=True, label='Frequency')
plt.grid(True)
plt.xlabel('Return ' + symbols[val])
plt.ylabel('Freq ' + symbols[val])
x = np.linspace(plt.axis()[0], plt.axis()[1])
plt.plot(x, scs.norm.pdf(x, loc=np.mean(Returns[symbols[val]]), scale=np.std(Returns[symbols[val]])),'r', lw=2.0, label='normal density')
plt.legend()

val = 1
plt.subplot(212)
Returns[symbols[val]].hist(bins=70, density=True, label='Frequency')
plt.grid(True)
plt.xlabel('Return ' + symbols[val])
plt.ylabel('Freq ' + symbols[val])
x = np.linspace(plt.axis()[0], plt.axis()[1])
plt.plot(x, scs.norm.pdf(x, loc=np.mean(Returns[symbols[val]]), scale=np.std(Returns[symbols[val]])),'r', lw=2.0, label='normal density')
plt.legend()
plt.show()


# In[11]:


print(norm.cdf(0.394630007))
print(norm.cdf(-0.849660998))


# In[12]:


print(norm.ppf(0.653442018))


# In[13]:




ReturnsData = pd.DataFrame(norm.cdf(Returns))
ReturnsData.columns =['S', 'F']
print (ReturnsData.head(3))
print (ReturnsData.tail(2))
size = len(ReturnsData)


# In[14]:


print (np.corrcoef(Returns.S, Returns.F)[1,0])
print (np.corrcoef(ReturnsData.S, ReturnsData.F)[1,0])


# In[15]:


def KendallCorrelationNonParametric(X, Y):
    #Estimate Kendall's correlation coefficient using nonparametric estimator
    n = len(X)
    s = 0
    for i in range(n):
        for j in range(i,n):
            if j > i:
                a = (X[i] - X[j]) * (Y[i] - Y[j])
                s +=np.sign(a)            
    return comb(n, 2) ** (-1) * s


# In[16]:


tau_data =  (KendallCorrelationNonParametric(ReturnsData.S, ReturnsData.F))
print (tau_data)
print (KendallCorrelationNonParametric(Returns.S, Returns.F))


# In[17]:


def Kendall_NonParametric(X, Y, pos):
    # The actual Copula based on data
    pos = int(pos)    
    n = len(X)
    Ts = 0
    for j in range(n):
        if (X[j] < X[pos]) & (Y[j] < Y[pos]):
            Ts += 1        
    return  Ts / (n - 1)

#let's test what it does on few data point
ff = [0, 1, 2, 5, 10, 20]
for i in ff:
    print(Kendall_NonParametric(ReturnsData.S, ReturnsData.F,i))


# In[19]:


Margins = []
for i in range(265):
    Margins.append(Kendall_NonParametric(ReturnsData.S, ReturnsData.F,i))

# Create the dataframe for the Kendall function, called MarginsDF
MarginsDF = pd.DataFrame(Margins)
MarginsDF.columns =['K']
MarginsDF.head()
del(Margins)

# Now we prepare the Reporting Table
# 1 - the Distribution Points, called t
dt     = 0.00001
t      = [dt]
delta  = 0.05
for i in range(20):
    tt = delta + t[-1]
    t.append(min(1,tt))

Frequency = pd.DataFrame(t)
Frequency.columns =['t']
#Frequency

# 2 - the Cumulative points
Cumulative = []
j = 0
for i in range(21):
    value = Frequency['t'].loc[i]
    Cumulative.append(MarginsDF.loc[MarginsDF['K'] <= value].count(axis=0)[0])
del(value)

Frequency['KCumulative'] = Cumulative
del(Cumulative)

MarginsDF.loc[MarginsDF['K'] <=0.00001].count(axis=0)


# In[20]:


# 3 - the and the Density and the Sample Points
Array    = Frequency['KCumulative'].diff()
Array[0] = Frequency['KCumulative'].loc[0]
Frequency['KDensity'] = Array
Denominator =  Frequency['KCumulative'].loc[20]
Frequency['KSample'] = Frequency['KCumulative']/Denominator
del(Denominator)
del(Array)

Frequency


# In[21]:


from scipy.integrate import quad, dblquad
# Copula Class
class Copula(object):
    def __init__(self, tau):
        self.tau   = tau
        self.theta = self.f_theta(self.tau)
    
    def LoadData(self, X, Y):
        self.X     = X
        self.Y     = Y
    
    def CopulaFunction(self, u, v):
        x = self.Generator(u)
        y = self.Generator(v)  
        return self.GeneratorInverse(x+y)
    
    # Numerical implementation of the copula derivative.
    def CopuladensityNumerical(self, u, v):
        h = 0.0001
        u_plus = min(u + h,1)
        u_minus= max(0,u-h)
        v_plus = min(v + h,1)
        v_minus=max(0, v-h)
        
        x_plus = self.Generator(u_plus)
        y_plus = self.Generator(v_plus)  
        x_minus = self.Generator(u_minus)
        y_minus = self.Generator(v_minus)  
        
        ss =  self.GeneratorInverse(x_plus + y_plus)  - self.GeneratorInverse(x_plus + y_minus) -               self.GeneratorInverse(x_minus + y_plus) + self.GeneratorInverse(x_minus + y_minus)
        
        ss = max(0, ss)
        return ss / (4.0 * h * h)
    
    def Copula_tau(self):            
        def f(u,v):
            a = self.CopulaFunction(u,v)
            b = self.CopuladensityNumerical(u,v)                
            return a * b
        res = dblquad(f, 0, 1, lambda x: 0, lambda x: 1)[0]
        
        return 4.0 * res - 1
   
    def KendallDistribution(self, t):
        pass

    def derivativeKendallDistribution(self, t):
        pass

    def Generator(self, t):
        pass

    def GeneratorInverse(self, t):
        pass

    def derivativeGenerator(self, t):
        pass

    def Integral_tau(self):    
        def f(t): 
            return self.Generator(t)/self.derivativeGenerator(t) 
        Integrale = scipy.integrate.quad(f, 0, 1)[0]      
        return 1 + 4*Integrale

    def f_tau(self, theta):
        pass
    
    def f_theta(self, tau):
        pass

# Gumbel Copula
class Gumbel(Copula):    
    def __init__(self, tau):
        Copula.__init__(self, tau)
        
    def KendallDistribution(self, t):
        return t - (t*np.log(t)/self.theta)

    def derivativeKendallDistribution(self, t):
        return 1 -np.log(t)/self.theta - 1/ self.theta

    def Generator(self, t):
        return (-np.log(t))**self.theta

    def GeneratorInverse(self, t):
        return np.exp(-t**(1/self.theta))

    def derivativeGenerator(self, t):
        return -1/t*self.theta*(-np.log(t))**(self.theta-1)
    
    def f_tau(self, theta):
        return 1 - 1/theta
    
    def f_theta(self, tau):
        return 1/(1-tau)
        
# Clayton Copula
class Clayton(Copula):    
    def __init__(self, tau):
        Copula.__init__(self, tau)
    
    def KendallDistribution(self, t):
        return t - (t**(self.theta + 1) - t)/self.theta

    def derivativeKendallDistribution(self, t):
        return 1 + 1/self.theta - 1/self.theta*(t**self.theta *(self.theta + 1))

    def Generator(self, t):
        return (t**(-self.theta)-1)

    def GeneratorInverse(self, t):
        return (1+t)**(-1/self.theta)

    def derivativeGenerator(self, t):
        return (-self.theta)*(t)**(-self.theta-1)

    def f_tau(self, theta):
        return theta/(theta + 2)
    
    def f_theta(self, tau):
        return 2*tau / (1-tau)

# Frank Copula
class Frank(Copula):    
    def __init__(self, tau):
        Copula.__init__(self,tau)    
    
    def KendallDistribution(self, t):
        num = (np.exp(-self.theta*t) -1)
        den = (np.exp(-self.theta)   -1)
        return t - (np.exp(self.theta*t)-1)/self.theta * np.log(num/den)

    def derivativeKendallDistribution(self, t):
        return -np.exp(self.theta*t)*(np.log((np.exp(-self.theta*t)-1)/(np.exp(-self.theta)-1)) + self.theta*(1-t))
    
    def Generator(self, t):
        num = np.exp(-self.theta*t) - 1
        den = np.exp(-self.theta)   - 1
        return -np.log(num/den)

    def GeneratorInverse(self, t):
        return -(1/self.theta)*np.log(np.exp(-self.theta-t) - np.exp(-t)  + 1)
    
    def derivativeGenerator(self, t):
        return self.theta / (1-np.exp(self.theta*t))
    
    def f_tau(self, theta):
        def f(t):
            return t/(np.exp(t)-1)   
        Integrale = scipy.integrate.quad(f, 0, theta)[0]      
        Dtheta = 1/theta*Integrale    
        return 1 - 4*(1-Dtheta)/theta

    def f_theta(self,tau):
        def f(x):
            return self.f_tau(x) - tau
        sol = root(f, 0.5) 
        return sol.x[0]


# In[22]:



#Instance of a Class Frank


# In[23]:


frank = Frank(tau_data)
frank.LoadData(ReturnsData.S, ReturnsData.F)
frank.theta


# In[24]:


# A copula value, given u and v
frank.CopulaFunction(.1,.5)


# In[26]:


# Let's check the integral from 0 to 1 of the Frank copula density adds up to 1, as it is a probability distribution
np.round(dblquad(frank.CopuladensityNumerical, 0, 1, lambda x: 0, lambda x: 1)[0],9)


# In[ ]:



#Instance of a Class Gumbel


# In[34]:


# Instance of a Class Gumbel
gumbel = Gumbel(tau_data)
gumbel.LoadData(ReturnsData.S, ReturnsData.F)
# The theta parameter
print(gumbel.theta)


# In[27]:


# Instance of a Class Cayton
clayton = Clayton(tau_data)
clayton.LoadData(ReturnsData.S, ReturnsData.F)
# The theta parameter
print(clayton.theta)


# In[35]:


# A copula value, given u and v
gumbel.CopulaFunction(.1,.5)


# In[36]:


# Let's check the integral from 0 to 1 of the Gumbel copula density adds up to 1, as it is a probability distribution
# There are some runtime errors. We will investigate
np.round(dblquad(gumbel.CopuladensityNumerical, 0, 1, lambda x: 0, lambda x: 1)[0],9)


# In[28]:


# A copula value, given u and v
clayton.CopulaFunction(.1,.5)


# In[29]:


# Let's check the integral from 0 to 1 of the clayton copula density adds up to 1, as it is a probability distribution
# There are some runtime errors. We will investigate 
np.round(dblquad(clayton.CopuladensityNumerical, 0, 1, lambda x: 0, lambda x: 1)[0],9)


# In[30]:


# Again, Kendall tau is invariant!
print (KendallCorrelationNonParametric(ReturnsData.S, ReturnsData.F))
print (KendallCorrelationNonParametric(Returns.S, Returns.F))


# In[37]:


print(gumbel.Integral_tau())
print(clayton.Integral_tau())
print(frank.Integral_tau())


# In[38]:


print(gumbel.f_tau(gumbel.theta))
print(clayton.f_tau(clayton.theta))
print(frank.f_tau(frank.theta))


# In[ ]:


# There are some runtime warnings when integrating the numerical version of the Copula density. We aim to replace soon 
# the numerical density with a analytical implementation and so remove these errors too

# Clayton and Frank tau:
# They are almost identical to the previous tau vaues.

# Gumbel tau:
# some difference!

print(clayton.Copula_tau())
print(frank.Copula_tau())
print(gumbel.Copula_tau())


# In[ ]:



#Calculate the Kendall distribution function for Clayton, Gumbel and Frank copula


# In[40]:


ClaytonArray = []
for i in range(21):
    t = Frequency['t'].loc[i]
    ClaytonArray.append(clayton.KendallDistribution(t))
del(t)

GumbelArray = []
for i in range(21):
    t = Frequency['t'].loc[i]
    GumbelArray.append(gumbel.KendallDistribution(t))
del(t)

FrankArray = []
for i in range(21):
    t = Frequency['t'].loc[i]
    FrankArray.append(frank.KendallDistribution(t))
del(t)

Frequency['K_Clayton'] = ClaytonArray
Frequency['K_Gumbel']  = GumbelArray
Frequency['K_Frank']   = FrankArray
del(ClaytonArray)
del(GumbelArray)
del(FrankArray)

Frequency


# In[41]:


Frequency['Clayton'] = (Frequency['K_Clayton'] - Frequency['KSample'])**2
Frequency['Gumbel']  = (Frequency['K_Gumbel']  - Frequency['KSample'])**2 
Frequency['Frank']   = (Frequency['K_Frank']   - Frequency['KSample'])**2 

print(sum(Frequency.Gumbel))
print(sum(Frequency.Clayton))
print(sum(Frequency.Frank))


# In[42]:


# Print our DataFrame
Frequency


# In[43]:


# Copula Class
class ArchimedeanSimulation(object):
    def __init__(self, tau, CopulaType):
        # CopulaType can be either:
        # C for Clayton,
        # G for Gumbel
        
        self.copula = ""
        if(CopulaType == "C"):
            self.copula = Clayton(tau_data) # an instance of class Clayton 
        else:
            self.copula = Gumbel(tau_data)  # an instance of class Gumbel
        
    def KendallInverse(self, q):       
        def f(t):
            return self.copula.KendallDistribution(t) - q
        
        return newton(f, 0.00001, fprime=self.copula.derivativeKendallDistribution)
    
    def Simulations(self, Number):
        npr.seed(30)
        sq      = npr.rand(Number, 2)
        self.uv = np.zeros([Number,2])
        
        for i in range(Number):                    
            q = sq[i,1]
            s = sq[i,0] 
            t = self.KendallInverse(q)
            self.uv[i,0] = self.copula.GeneratorInverse(s     * self.copula.Generator(t))
            self.uv[i,1] = self.copula.GeneratorInverse((1-s) * self.copula.Generator(t))
        
        self.M = norm.ppf(self.uv)
        
class ArchimedeanSimulationFrank(ArchimedeanSimulation):
    def __init__(self, tau):        

        
        self.copula = Frank(tau_data)  
        
    def KendallInverse(self, q):       
        def f(t):
            return self.copula.KendallDistribution(t) - q
        
        return newton(f, 0.0000001)

Arch_C = ArchimedeanSimulation(tau_data,"C")
Arch_G = ArchimedeanSimulation(tau_data,"G")
Arch_F = ArchimedeanSimulationFrank(tau_data)


# In[44]:


t_K_Clayton = np.zeros([len(Frequency)])
t_K_Gumbel  = np.zeros([len(Frequency)])
t_K_Frank   = np.zeros([len(Frequency)])
for i in range(0, len(Frequency)):
    t_K_Clayton[i] = Arch_C.KendallInverse(Frequency.K_Clayton.loc[i])
    t_K_Gumbel[i]  = Arch_G.KendallInverse(Frequency.K_Gumbel.loc[i])
    t_K_Frank[i]   = Arch_F.KendallInverse(Frequency.K_Frank.loc[i])
    
arr = np.stack((t_K_Clayton, t_K_Gumbel, t_K_Frank), axis=1)
arr = np.round(arr,4)
print(arr)


# In[46]:


#Running the simulation algo for the Gumbel copula


# In[53]:


# Let's simulate bivariate numbers from the Gumbel copula 
Arch_G.Simulations(265)
Data_Arch_Returns     = pd.DataFrame(Arch_G.M)
Data_Arch_Univariates = pd.DataFrame(Arch_G.uv)
Data_Arch_Returns.columns =['S', 'F']
Data_Arch_Univariates.columns =['U', 'V']
Data_Arch_Returns.to_csv('GumbelReturns.csv', index = False)
Data_Arch_Univariates.to_csv('GumbelUnivariates.csv', index = False)

# Let's save both returns and univariates in a csv file 
Returns2    = pd.read_csv ('GumbelReturns.csv',     index_col=False)
Univariates = pd.read_csv ('GumbelUnivariates.csv', index_col=False)

# and notice the Kendall Coefficient. It is the same for both returns and univariate data,
# linear correlation is not the same 
print (np.corrcoef(Returns2.S, Returns2.F)[1,0])
print (KendallCorrelationNonParametric(Returns2.S, Returns2.F))
print(tau_data)


# In[49]:


# and notice the Kendall Coefficient. It is the same for both returns and univariate data,
# linear correlation is not the same 
print (np.corrcoef(Univariates.U, Univariates.V)[1,0])
print (KendallCorrelationNonParametric(Univariates.U, Univariates.V))
print(tau_data)


# In[50]:


del(Data_Arch_Returns)
del(Data_Arch_Univariates)
del(Returns2)
del(Univariates)


# In[ ]:



#Running the simulation algo for the Clayton copula


# In[52]:


Arch_C.Simulations(265)
Data_Arch_Returns     = pd.DataFrame(Arch_C.M)
Data_Arch_Univariates = pd.DataFrame(Arch_C.uv)
Data_Arch_Returns.columns =['S', 'F']
Data_Arch_Univariates.columns =['U', 'V']
Data_Arch_Returns.to_csv('ClaytonReturns.csv', index = False)
Data_Arch_Univariates.to_csv('ClaytonUnivariates.csv', index = False)

# Let's save both returns and univariates in a csv file 
Returns2    = pd.read_csv ('ClaytonReturns.csv',     index_col=False)
Univariates = pd.read_csv ('ClaytonUnivariates.csv', index_col=False)

# and notice the Kendall Coefficient
print (np.corrcoef(Returns2.S, Returns2.F)[1,0])
print (KendallCorrelationNonParametric(Returns2.S, Returns2.F))
print(tau_data)


# In[54]:


# and notice the Kendall Coefficient
print (np.corrcoef(Univariates.U, Univariates.V)[1,0])
print (KendallCorrelationNonParametric(Univariates.U, Univariates.V))
print(tau_data)


# In[55]:


del(Data_Arch_Returns)
del(Data_Arch_Univariates)
del(Returns2)
del(Univariates)


# In[56]:



#Running the simulation algo for the Frank copula


# In[60]:


# Let's simulate bivariate numbers from the Gumbel copula 
Arch_F.Simulations(265)
Data_Arch_Returns     = pd.DataFrame(Arch_F.M)
Data_Arch_Univariates = pd.DataFrame(Arch_F.uv)
Data_Arch_Returns.columns =['S', 'F']
Data_Arch_Univariates.columns =['U', 'V']
Data_Arch_Returns.to_csv('FrankReturns.csv', index = False)
Data_Arch_Univariates.to_csv('FrankUnivariates.csv', index = False)

# Let's save both returns and univariates in a csv file 
Returns2    = pd.read_csv ('FrankReturns.csv',     index_col=False)
Univariates = pd.read_csv ('FrankUnivariates.csv', index_col=False)

# and notice the Kendall Coefficient
print (np.corrcoef(Returns2.S, Returns2.F)[1,0])
print (KendallCorrelationNonParametric(Returns2.S, Returns2.F))
print(tau_data)


# In[61]:


# and notice the Kendall Coefficient
print (np.corrcoef(Univariates.U, Univariates.V)[1,0])
print (KendallCorrelationNonParametric(Univariates.U, Univariates.V))
print(tau_data)


# In[62]:


del(Data_Arch_Returns)
del(Data_Arch_Univariates)
del(Returns2)
del(Univariates)


# In[4]:


print(variance(returns.S, Returns.F))


# In[ ]:





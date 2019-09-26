#library(FMStable)

import warnings
import numpy as np
from numpy import pi, log
from scipy.special import digamma

def hmp_stat(p, w=None):
    if w is None:
        return 1/np.mean(1/p)
    else:
        return sum(w)/sum(w/p)

def p_hmp(p, w=None, L=None, w_sum_tolerance=0.000001, multilevel=True):
    if L is None:
        warnings.warn("L not specified: for multilevel testing set L to the total number of individual p-values.")
        L = len(p)
    if len(p) == 0:
        return None
    if len(p) > L:
        warnings.warn("The number of p-values cannot exceed L.")
    if w is None:
        w = np.repeat(1/L,len(p))
    elif any(weight < 0 for weight in w):
        raise ValueError("No weights can be negative.")
    w_sum = sum(w)
    if w_sum > 1 + w_sum_tolerance:
        raise ValueError("Weights cannot exceed 1.")
    HMP = hmp_stat(p, w)
    O_874 = 1 + digamma(1) - log(2/pi)
    if multilevel:
        return w_sum * pEstable(w_sum/HMP, 
                                setParam(alpha=1,
                                         location=(log(L) + O_874),
                                         logscale=log(pi/2),
                                         pm=0),
                                lower_tail = False)
    else:
        return pEstable(1/HMP, 
                        setParam(alpha = 1, 
                                 location = (log(len(p)) + O_874), 
                                 logscale = log(pi/2), 
                                 pm = 0), 
                        lower.tail = False)

def mamml_stat(R, w=None):
    if any(p < 1 for p in R):
        raise ValueError("Maximized likelihood ratios (R) cannot be less than one.")
    if w is None:
        return mean(R)
    else:
        w = w/sum(w)
        return sum(w*R)

def p_mamml(R, nu, w=None, L=None):
    if L is None:
        warnings.warn("L not specified, assuming L = len(R)")
        L = len(R)
    if len(nu) not in (1,len(R)):
        raise ValueError("Degrees of freedom (nu) must have length=1 or length=length(R)")
    Rbar = mamml_stat(R, w)
    if any(n <= 0 for n in nu):
        raise ValueError("Degrees of freedom (nu) must be strictly positive.")
    nu_max = max(nu)
    if nu_max < 2:
        c = pgamma(log(Rbar), nu_max/2, 1, lower.tail=False) * Rbar
    else:
        c = pgamma(log(len(R)*Rbar), nu_max/2 ,1, lower.tail=False)*len(R)*Rbar
    
    ########################################
    ## NEED TO TRANSLATE LOGIC BELOW HERE ##
    ########################################
    
    O_874 = 1 + digamma(1) - log(2/pi)
    return pEstable(Rbar,setParam(alpha=1,
                                  location=c*(log(L)+O_874),
                                  logscale=log(pi/2*c),
                                  pm=0), lower.tail=False)


################################################
##                                            ##
## NEED TO TRANSLATE ALL FUNCTIONS BELOW HERE ##
##                                            ##
################################################

def dLandau(x,mu=log(pi/2),sigma=pi/2,log=False):
    param = setParam(alpha=1, location=mu, logscale=log(sigma), pm=0)
    return dEstable(x,param,log=log)
    
def pLandau(x,mu=log(pi/2),sigma=pi/2,log=False,lower.tail=True):
    param = setParam(alpha=1, location=mu, logscale=log(sigma), pm=0)
    return pEstable(x,param,log=log,lower.tail=lower.tail)
    
def qLandau(p,mu=log(pi/2),sigma=pi/2,log=False,lower.tail=True):
    param = setParam(alpha=1, location=mu, logscale=log(sigma), pm=0)
    return qEstable(p,param,log=log,lower.tail=lower.tail)
    
def rLandau(n,mu=log(pi/2),sigma=pi/2):
    return qLandau(runif(n),mu,sigma)
    
def dharmonicmeanp(x, L, log=False):
    x=pmax(1e-300,x); # Would be better to calculate limit
    if log:
        return dLandau(1/x, mu=log(L)+1+psigamma(1)-log(2/pi), sigma=pi/2, log=True)-2*log(x)
    else:
        return dLandau(1/x, mu=log(L)+1+psigamma(1)-log(2/pi), sigma=pi/2, log=False)/x**2))
    
def pharmonicmeanp(x, L, log=False, lower.tail=True):
    return pLandau(1/x, mu=log(L)+1+psigamma(1)-log(2/pi), sigma=pi/2, log=log, lower.tail=!lower.tail)))
    
def qharmonicmeanp(p, L, log=False, lower.tail=True):
    return 1/qLandau(p, mu=log(L)+1+psigamma(1)-log(2/pi), sigma=pi/2, log=log, lower.tail=!lower.tail)))
    
def rharmonicmeanp(n, L):
    return qharmonicmeanp(runif(n),L)
    
def dmamml(x, L, df, log=False):
    if df == 2:
        c = 1
    elif df < 2:
        c = x*(1-pgamma(log(x),df/2,1))
    else:
        c = L*x*(1-pgamma(log(L*x),df/2,1))
    return dLandau(x, mu=c*(log(L)+1+psigamma(1)-log(2/pi)), sigma=c*pi/2, log=log)
    
def pmamml(x, L, df, log=False, lower.tail=True):
    if df == 2:
        c = 1
    elif df < 2:
        c = x*(1-pgamma(log(x),df/2,1))
    else:
        L*x*(1-pgamma(log(L*x),df/2,1))
    return pLandau(x, mu=c*(log(L)+1+psigamma(1)-log(2/pi)), sigma=c*pi/2, log=log, lower.tail=lower.tail)
    
def qmamml(p, L, df, log=False, lower.tail=True, xmin=1+1e-12, xmax=1e12):
    f = function(x) pmamml(x,L,df,log=log,lower.tail=lower.tail)-p
    return uniroot(f,c(xmin,xmax))$root)
    
def rmamml(n, L, df):
    return qmamml(runif(n),L,df)


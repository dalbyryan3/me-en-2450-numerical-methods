# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 23:18:41 2019
Lab3 colebrook_equation
@author: Ryan Dalby
"""
import numpy as np
def colebrook_equation(f,e,D,Re):
    """
    Returns a value of the colebrook equation given parameters passed in, 
    to satisfy colebrook equation returned value should equal to 0
    """
    return -2.0 * np.log10(((e/D) / 3.7) + (2.51 / (Re * np.sqrt(f)))) - (1/np.sqrt(f))

    

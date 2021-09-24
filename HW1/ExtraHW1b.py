# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 01:17:19 2019
Exercise 1 and 2 extra
@author: Ryan Dalby
"""
def toDecimal(num):
    '''Takes binary number in signed magnitude form as a string
    (first bit before number must be 0 or 1 to specify sign)
    and gives decimal form'''
    ans = int(num[1:], 2)
    if int(num[0]) == 1:
        return -1 * ans
    else:
        return ans

def toBinary(num):
    '''Takes decimal number and gives binary number in signed 
    magnitude form'''
    if num < 0:
        return '1' + (bin(abs(num))[2:])
    else:
        return bin(num)[2:]


print(toDecimal('0110100100'))
print(toDecimal('11101101'))

print(toBinary(-795))
print(toBinary(-109))


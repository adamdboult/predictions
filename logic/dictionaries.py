################
# Dictionaries #
################
numeralsL = []
numObj = []
empty=[]
import copy
from axioms import *
numObj.append(["i",str(0)])
numObj.append(["v",empty])
numObj.append(["t","n"])

numeralsL.append(numObj)
for i in range(0, 2):
    numObj = copy.deepcopy(numObj)
    numObj[0]=["i",str(i+1)]
    numObj[1]=["v",seq(numObj[1][1])]
    numeralsL.append(numObj)

plusObj = []
plusObj.append(["t","f"])
plusObj.append(["i","+"])
plusObj.append(["v","add"])
numeralsL.append(plusObj)

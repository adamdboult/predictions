##################
# more functions #
##################
from axioms import *

def matchByKey(obj, key, value):
    test = loopList(obj, keyIs, [key])
    return eq(test[1],value)

def dictExpand(obj, dictionary, key):
    keyValMatch = loopList(obj, keyIs, [key])

    dictObjMatch = loopList(dictionary, matchByKey,[key,keyValMatch[1]])
    print (dictObjMatch)
    print ("M")
    loop = 1
    while loop == 1:
        dictKey = loopList(dictObjMatch, retTrue,[])
        print (dictKey)
        print (key)
        objMatch = loopList(obj, keyIs, [dictKey[0]])
        if (objMatch == False):
            obj.append(dictKey)
        print (objMatch)
        loop = 0

def keyIs(obj, key):
    print ("A")
    retVal = arbFunc(eq,[],[obj,key],[],[0,1])
    print ("B")
    return retVal

def retTrue(a):
    return True

############
## Discuss #
############
# object exists
# prioritise where to look
# do test
# decision

import dictionaries

##########
# Axioms #
##########
def loopF(inp,func,index):
    blank_json = {}
    for i in range(0, index):
        if (isinstance(inp, list)):
            func(*inp)
        else:
            func(inp)
    return inp

def addToSet(S,n):
    S.append(n)

def incN(n):
    return n + 1

###########
# Derived #
###########
def add(a,b):
    return loopF(a, incN, b)

def mult(a,b):
    return loopF([a,a], add, b)

def power(a, b):
    return loopF([a,a], mult, b)

#############
# Functions #
#############
def importF(old, new):
    for i in range(0, len(old)):
        newObj={}
        newObj["i"]=old[i]
        new.append(newObj)
        
def cleanF(old):
    delList(old)
    orderList(old)

def runFunc(func,extras):
    args=[workings]
    for extra in extras:
        args.append(extra)
    func(*args)
    cleanF(workings)
        
def labelList(old,dictionary,key):
    for i in range(0, len(old)):
        for j in range(0, len(dictionary)):
            match=0
            for keyi in old[i]:
                for keyj in dictionary[j]:
                    if (keyi==keyj==key and old[i][keyi] == dictionary[j][keyj]):
                        match=1
            if (match==1):
                for keyj in dictionary[j]:
                    old[i][keyj]=dictionary[j][keyj]

def orderList(old):
    for i in range(0, len(old)):
        old[i]["o"]=i

def delList(old):
    j = 0
    for i in range(0, len(old)):
        for key in old[j]:
            if (key == "d"):
                del old[j]
                j-=1
        j +=1
###!!!!!

def checkMatch(a, b):
    return a==b

def mergeNum(old, base, iters):
    stop = 0
    learr = []

    for i in range(0, iters):
        learr.append(0)
    while stop == 0:
        for i in range(0, iters):
            if learr[i]==iters:
                learr[i]=0
                if i + 1 == iters:
                    stop = 1
                else:
                    learr[i+1]+=1
        if (stop == 0):
            match = 1
            checks =[]
            checkInputs =[]
            for i in range(0, iters):
                checkInputs.append(old[learr[i]])
            ####
            checks.append([checkInputs[0]["t"],"n"])
            checks.append([checkInputs[1]["t"],"n"])
            checks.append([checkInputs[0]["o"] + 1, checkInputs[1]["o"]])
            ####
            for check in checks:
                if match == 1:
                    match = checkMatch(check[0],check[1])
            if match ==1:
                updates = []
                updates.append([checkInputs[0]["d"],1])
                updates.append([checkInputs[1]["v"],checkInputs[1]["v"] + base * checkInputs[0]["v"]])
                ####
                checkInputs[0]["d"]=1
                checkInputs[1]["v"]=checkInputs[1]["v"] + base * checkInputs[0]["v"]
                ####
        learr[0]+=1
        
def mergeNumAction(old):
    print(1)
        
def solve(old):
    for i in range(0, len(old)):
        for j in range(0, len(old)):
            for k in range(0, len(old)):
                if old[i]["t"]=="n":
                    if (old[k]["o"] == old[i]["o"] + 2 and old[j]["o"] == old[i]["o"]+1):
                        old[i]["v"]+=old[k]["v"]
                        old[k]["d"]=1
                        old[j]["d"]=1

def splitN(old,base):
    j=0
    for i in range(0, len(old)):
        if (old[j]["t"]=="n"):
            upper=old[j]["v"]/base
            newObj={}
            newObj["v"]=upper
            newObj["t"]="n"
            lower=old[j]["v"]-upper*base
            old[j]["v"]=lower
            old.insert(j, newObj)
        j+=1

def makeOutput(old):
    for i in range(0, len(old)):
        for j in range(0, len(old)):
            if old[i]["o"]+1 == old[j]["o"]:
                old[i]["d"]=1
                old[j]["i"]=old[i]["i"] + old[j]["i"]

############
# Settings #
############    
inputString="11+3"
answer = "14"
workings=[]
base=10

#######
# Run #
#######
print (inputString)

importF(inputString,workings)
runFunc(labelList,[labelDict,"i"])
runFunc(mergeNum,[base,2])
runFunc(solve,[])

runFunc(splitN,[base])

runFunc(labelList,[labelDict,"v"])

runFunc(makeOutput,[])

answerEst = workings[0]["i"]
print ("Correct: "+answer)
print ("Calculation: "+answerEst)
print (answer == answerEst)

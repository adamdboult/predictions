##############
# Dictionary #
##############
labelDict=[]
for i in range(0, 9):
    newObj={}
    newObj["v"]=i
    newObj["i"]=str(i)
    newObj["t"]="n"
    labelDict.append(newObj)
addObj={}
addObj["v"]="add"
addObj["i"]="+"
addObj["t"]="f"
labelDict.append(addObj)

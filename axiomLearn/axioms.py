##########
# Axioms #
##########
def seq(set):
    toRet = []
    for obj in set:
        toRet.append(obj)
    toRet.append(set)
    return toRet

def loopList(loopSet, crit, extras):
    for obj in loopSet:
        args =[obj]
        for extra in extras:
            args.append(extra)
        if crit(*args):
            return obj
    return False

def eq(a,b):
    return a == b

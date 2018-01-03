#!/usr/bin/python3
##########
# Import #
##########

####################
# Define functions #
####################

# Consumer
###########
# U = ln(
#         Q      ^  pref      .
#        (1 - L) ^ (1 - pref)
#     )

def getUtil(pref, quantity, labour):
    utility = pref * Math.log(quantity) + (1 - pref) * Math.log(1 - labour)# - lam * (price * quantity - wage * labour)
    return utility

def getMargCons(pref, quantity, labour):
    margCons = pref / quantity - lam * (price)
    return margCons

def getMargLabour(pref, quantity, labour):
    margLab = - (1 - pref) / (1 - labour) - lam * (- wage)
    return margLab

# Note: Set to 0 and combinate above to get:
#    pref / (quantity * price) = (1 - pref) / ((1 - labour) * wage)
#   (pref / (1 - pref)) * (wage / price) = quantity / (1 - labour)
def getMargLam(price, quantity, wage, labour):
    margLam = - (price * quantity - wage * labour)
    
    # Note: set to 0 to get:
    #   labour = quantity * (price / wage)
    #   (pref / (1 - pref)) * (wage / price) = quantity / (1 - quantity * (price / wage))
    #   (pref / (1 - pref)) * (wage / price) * (1 - quantity * (price / wage)) = quantity

# consumer quantity demanded
def getConsQuant(price, wage, pref):
    quantity = pref * (wage / price)
    return quantity

# consumer labour demanded
def getConsLab(pref):
    labour = pref
    return labour

# Firm
#######

# q = l ^ a . k ^ b
# profit = p . l ^ a . k ^ b - w . l - r . k
#dl      = a . p . l ^ (a - 1) . k ^ b - w
#dk      = b . p . l ^ a . k ^ (b - 1) - r

def getProfit(l, k, a, b, w, r, p):
    q = l^a * k^b
    profit = p*q - w*l - r*k
    return profit

def firmMarg(price, quantity, wage, stock):
    margProf = price - 2 * wage * (quantity - stock)
    return margProf

def getFirmQuant(price, wage, stock):
    quantity = price / (2 * wage) + stock
    return quantity

def getFirmLab(quantity, stock):
    labour = (quantity - stock) ** 2
    return labour

# Market
#########
import math
def getEquilibrium(cost, pref):
    price = 1
    stock = 0
    wage = 1
    
    sumErr = 1
    i = 1
    delta = 0.001
    while (sumErr > 0.00001):
        quantFirmGoal = getFirmQuant(price, wage, stock)
        labFirmGoal = getFirmLab(quantFirmGoal, stock)
        
        quantConsGoal = getConsQuant(price, wage, pref)
        labConsGoal = getConsLab(pref)
        
        profit = getProfit(price, quantFirmGoal, wage, stock)
        profitUnits = profit / price
    
        goodsExcess = quantFirmGoal - quantConsGoal - profitUnits
        labourExcess = labConsGoal - labFirmGoal
        sumErr = goodsExcess ** 2 + labourExcess ** 2

        deltaPrice = - goodsExcess * delta
        deltaWage = - labourExcess * delta
        
        print ("\n\n#########\n# ROUND ", i, "#\n#########", i)
        print ("Price: ", price)
        print ("Wage:  ", wage)
        print ("P/W:   ", price / wage)

        print ("\n---Firm---")
        print ("* Goals")
        print ("Quant:  ", quantFirmGoal)
        print ("Labour: ", labFirmGoal)
        
        print ("\n---Consumer---")
        print ("* Goals")
        print ("Quant:  ", quantConsGoal)
        print ("Labour: ", labConsGoal)

        print ("\n---Capitalist---")
        print ("* Goals")
        print ("Profit: ", profit)
        print ("Pro/Pr: ", profitUnits)

        print ("\n---Excess supply---")
        print ("Quant:  ", quantFirmGoal - quantConsGoal - profitUnits)
        print ("Labour: ", labConsGoal - labFirmGoal)

        print ("\ndeltaP ", deltaPrice)
        print ("deltaW ", deltaWage)

        i += 1
        price = price + deltaPrice
        wage = wage + deltaWage

    return
    
#######
# Run #
#######
startPrice = 10
cost = 10
pref = 0.5
getEquilibrium(cost, pref)

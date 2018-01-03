#!/usr/bin/env python3
#############
# Libraries #
#############
import random
import math

###########
# Classes #
###########
class person:
    """A person"""
    instances = []
    def __init__(self):
        a = random.random()
        self.prefs = [0.5, 0.5]
        self.data = [a, 1 - a, 0]
        self.utilityUpdate()

        person.instances.append(self)

    def trade(self, market):
        self.marginalUtilityUpdate()
        self.goalUpdate(market)
        goods = [0, 1]
        for good in goods:
            toBuy = self.goal[good] - self.data[good]
            price = market.price[good]
            if (
                    self.data[good]   + toBuy         >= 0 and
                    self.data[2]      - toBuy * price >= 0 and
                    market.data[good] - toBuy         >= 0 and
                    market.data[2]    + toBuy * price >= 0
            ):
                market.data[good] -= toBuy
                market.data[2]    += toBuy * price
                self.data[good]   += toBuy
                self.data[2]      -= toBuy * price
        self.utilityUpdate()

    def goalUpdate(self, market):
        self.goal = [self.data[0], self.data[1]]
        if (self.marginalUtility[0]/market.price[0] > self.marginalUtility[1]/market.price[1]):
            self.goal[0] +=0.01
            self.goal[1] -=0.01
        elif (self.marginalUtility[0]/market.price[0] < self.marginalUtility[1]/market.price[1]):
            self.goal[0] -=0.01
            self.goal[1] +=0.01
        
    def utilityUpdate(self):
        self.utility = (self.data[0] ** self.prefs[0]) * (self.data[1] ** self.prefs[1])

    def marginalUtilityUpdate(self):
        self.marginalUtility = [
            self.prefs[0] * (self.data[0] ** (self.prefs[0] -1)) * (self.data[1] **  self.prefs[1]),
            self.prefs[1] * (self.data[0] **  self.prefs[0]    ) * (self.data[1] ** (self.prefs[1] - 1))
        ]

class market:
    """A amarket"""
    instances = []
    def __init__(self):
        self.data = [0, 0, 100]
        self.lastStock = self.data[:]
        self.price = [10.0, 10.0]
        market.instances.append(self)

    def updatePrices(self):
        goods = [0, 1]
        for good in goods:
            if self.data[good] < self.lastStock[good]:
                self.price[good] += 0.001
            else:
                self.price[good] -= 0.001

        self.lastStock = self.data[:]

#############
# Functions #
#############
def printStatus(entity):
    print ("Printing")
    for instance in entity.instances:
        print("Data: ", instance.data)
        #try:
        #    print("Utility: ", instance.utility)
        #except:
        #    continue
    print ()

#################
# Configuration #
#################
population = 3

##############
# Initialise #
##############
print ("Intialising...")
print ("--------------")
for i in range (population):
    person()

auctioneer = market()

#######
# Run #
#######
printStatus(person)
printStatus(market)
var = ""
i = 0
while (var != "q"):
    print ("STARTING")
    print (i, ".......")
    for instance in person.instances:
        instance.trade(auctioneer)
    auctioneer.updatePrices()
    var = input("Please enter something: ")
    print ("you entered", var)
    printStatus(person)
    printStatus(market)
    print (auctioneer.price)
    i+=1

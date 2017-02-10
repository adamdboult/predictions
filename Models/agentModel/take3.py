#!/usr/bin/env python3
#############
# Libraries #
#############
import random

####################
# Global variables #
####################
priceStep = 1
quantityStep = 1
learnRate = 0.01
            
#################
# Configuration #
#################
consumerPop = 1
firmPop = 1

#####################
# Initialise agents #
#####################
print ("------------------------")
print ("---Intialising world----")
print ("------------------------")
print ()

print ("------------------------")
print ("---Intialising agents---")
print ("------------------------")
consumers = []
firms = []
for i in range (consumerPop):
    consumers.append(
        consumer(i)
    )
    print ("Initiated consumer ", i)
for i in range (firmPop):
    firms.append(
        consumer(i)
    )
    print ("Initiated firm ", i)

print ()

#######
# Run #
#######
print ("------------------------")
print ("---Starting loop--------")
print ("------------------------")

var = ""
while (var != "q"):
    orders = {}
    for good in goods:
        orders[good] = []
    for i in agents:
        newOrder = i.getAction()
        for good in newOrder:
            newOrder[good].append(newOrder[good])
    print ()
    print ("-----")
    var = input("Please enter something: ")
    print ("-----")

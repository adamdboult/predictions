#!/usr/bin/env python3
#############
# Libraries #
#############
import random
import math

################
# Agent models #
################
agentDict = {
    "person": {
        "actions": [],
        "preferences": []
    },
    "world": {
        "actions": [],
        "preferences": []
    }
}

###########
# Classes #
###########
class agent:
    """An agent"""
    def __init__(self, template, world):
        model = agentDict[template]
        self.children = []
        self.name = template
        self.actions = model["actions"]
        self.preferences = model["preferences"]
        self.world = world

    def addChild(self, child):
        newChild = agent(
            child,
            self
        )
        self.children.append(newChild)

    def takeAction(self):
        for action in self.actions:
            action = 0

    def worldInput(self):
        print (1)

    def utilityUpdate(self):
        print (2)
    
    def update(self):
        print ("Updating ", self.name)
        self.worldInput()
        for child in self.children:
            child.update()
        self.utilityUpdate()
        self.takeAction()
        
#################
# Configuration #
#################
agentPopulation = 3

##############
# Initialise #
##############
print ("Intialising world....")
print ("---------------------")
print ()
world = agent(
    "world",
    None
    )

print ("Intialising agents...")
print ("---------------------")
print ()
for i in range (agentPopulation):
    world.addChild("person")

#######
# Run #
#######
print ("Starting loop........")
print ("---------------------")
print ()
var = ""
while (var != "q"):
    print ()
    var = input("Please enter something: ")
    world.update()

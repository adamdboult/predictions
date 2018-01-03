################
# Dependencies #
################
from classes import agentClass
from learnFunctions import *

#############
# Templates #
#############


################
# Child agents #
################
class consumer(agentClass):
    'A consumer is an agent'
    def __init__(self, goods, assets, frames, discount, alpha, beta, maxLeisure):

        self.alpha = alpha
        self.beta = beta
        self.maxLeisure = maxLeisure
        agentClass.__init__(self, goods, discount, assets, frames)

        self.updateAssets()

    def updateAssets(self):
        self.assets["consumption"] = 0
        self.assets["leisure"] = self.maxLeisure

    def getReward(self, state):
        consumption = state["consumption"]
        leisure = state["leisure"]
        reward = self.alpha * Math.log(consumption) + self.beta * Math.log(leisure)
        return (reward)
        
class firm(agentClass):
    'A firm is an agent'
    def __init__(self, goods, assets, frames, discount, alpha, beta):
        self.alpha = alpha
        self.beta = beta

        agentClass.__init__(self, goods, discount, assets, frames)

        self.updateAssets()

    def updateAssets(self):
        print (self.assets["consumption"])
        print ("--")
        print (self.assets["labour"])
        print (self.alpha)
        print (self.assets["labour"] ** self.alpha)
        print (self.assets["capital"] ** self.beta)
        self.assets["consumption"] = self.assets["consumption"] + (self.assets["labour"] ** self.alpha) * (self.assets["capital"] ** self.beta)
        self.assets["labour"] = 0
        self.assets["capital"] = 0

    def getReward(self, state):
        currentWealth = state[0].wealth
        previousWealth = state[1].wealth
        reward = currentWealth - previousWealth
        return (reward)

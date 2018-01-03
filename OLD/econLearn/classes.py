################
# Dependencies #
################
import math

from learnFunctions import *

#########
# Agent #
#########
class agentClass:
    "Agent"
    # Initiation
    #############
    def __init__(self, goods, discount, assets, frames):
        self.discount = discount


        actionTemplate = getTemplate("action", goods)
        stateTemplate = getTemplate("state", goods)
        goodsTemplate = getTemplate("goods", goods)

        self.assets = {}
        for good in goods:
            self.assets[good] = float(0)
        print (assets)
        for key, value in assets.items():
            self.assets[key] = value

        self.actions = actionTemplate

        self.state = []
        for i in range(frames):
            self.state.append(stateTemplate)


    # Policy
    #########
    def updateStateArray(self, stateTemplate, accept, prices):
        self.state.pop(0)

        newState = stateTemplate

        newState.exists = 1
        newState.assets = self.assets
        newState.actions = self.actions
        newState.prices = prices
        newState.accept = accept

        self.state.append(newState)
        
    # External functions
    #####################
    def getActions(self):
        # check actions are possible before returning!
        return (self.actions)

    def updateActions(self):
        self.updateAssets()
        self.updateState(accept)
        # get state
        # turn it into vector
        # do forward pass of vector through neural network
        # get output vector of raise, lower, same for price and quantity of each good
        for good in goods:
            priceOrder = getMaxIndex(priceMoves)
            quantityOrder = getMaxIndex(quantityMoves)
            # price
            epsilon = Math.random()
            if (epislon < x):
                directionEpsilon = Math.random()
                priceOrder = 2
                if (directionEpsilon < (2/3)):
                    priceOrder = 1
                if (directionEpsilon < (1/3)):
                    priceOrder = 0
                    
            # quantity
            epsilon = Math.random()
            if (epislon < x):
                directionEpsilon = Math.random()
                quantityOrder = 2
                if (directionEpsilon < (2/3)):
                    quantityOrder = 1
                if (directionEpsilon < (1/3)):
                    quantityOrder = 0
                            
    def learn(self, Qnow):
        # Reward is discounted flow of reward
        # Rt = r_t + d*r_t+1..

        # We define a function as the discounted reward conditional on taking action a at state s, and optimal actions from then.
        # Q(state, action) = max(Rt+1)

        # Bellman equation: a' is the optimal decision at s'
        # Q(s, a) = r + discount*Q(s',a')

        # How do we train?
        # Neural network predicts Q(s, a). It can also provide Q(s', a'), that is the Q at the next action taken.

        # Loss is how far away left is from right
        
        Qnow = Qnow # We already have this
        maxQnext = 0
        reward = self.getReward(state)
        
        loss = self.getLoss(reward, maxQnext, Qnow)

        # Now that we have the loss we can update the neural network using back propagation.

    # Train
    #########
    def getLoss(self, reward, maxQnext, Qnow):
        loss = (1/2) * [reward + maxQnext - Qnow]^2
        return (loss)


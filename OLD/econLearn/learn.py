################
# Dependencies #
################
import math

from classes import agentClass
from childClasses import firm, consumer
from learnFunctions import *

#################
# Configuration #
#################
maxLeisure = 24
agentFrames = 4
numConsumers = 3
numFirms = 3

goods = [
    "consumption",
    "labour",
    "saving",
    "capital",
    "gold"
]

stateTemplate = getTemplate("state", goods)

#####################
# Initialise agents #
#####################
for i in range(numConsumers):
    assets = {
    }
    frames = agentFrames
    discount = 0.05
    alpha = 0.5
    beta = 0.5
    consumer(goods, assets, frames, discount, alpha, beta, maxLeisure)

for i in range(numFirms):
    assets = {
        "gold": 10
    }
    frames = agentFrames
    discount = 0.05
    alpha = 0.5
    beta = 0.5
    firm(goods, assets, frames, discount, alpha, beta)

#######
# Run #
#######
buyOrders = getTemplate("order", goods)
sellOrders = getTemplate("order", goods)
while True:
    # Update orders MERGE WITH BELOW. DO DETERMINISTIC FOR USERS
    ################
    for agent in agentClass:
        agent.updateActions()
        
        for good in goods:
            
            for order in orderArray:
                if ():
                    buyOrders[good].append(order)
                else:
                    sellOrders[good].append(order)

        #DETqq21ERMINE ELIGIBLE BUNDLES
        
#+ experience replay. remember <s, a, r, s'>, use random mini batches to train

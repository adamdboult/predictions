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
    
#######
# Run #
#######
startPrice = 10
cost = 10
pref = 0.5
getEquilibrium(cost, pref)

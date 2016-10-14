#######################
# Import dependencies #
#######################
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

import random

###############
# Money agent #
###############
class simAgent(Agent):
    def __init__(self, unique_id):
        self.unique_id = unique_id
        self.wealth = 1
        
    def move(self, model):
        possible_steps = model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        new_position = random.choice(possible_steps)
        model.grid.move_agent(self, new_position)
        
    def give_money(self, model):
        cellmates = model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            other = random.choice(cellmates)
            other.wealth += 1
            self.wealth -= 1

    def step(self, model):
        self.move(model)
        if self.wealth > 0:
            self.give_money(model)

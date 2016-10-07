#######################
# Import dependencies #
#######################
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

import random

#################
# Import agents #
#################
from simLand.agents import simAgent

from wolf_sheep.agents import Sheep, Wolf, GrassPatch

###################
# Import schedule #
###################
from wolf_sheep.schedule import RandomActivationByBreed

###############
# Money model #
###############
class simModel(Model):
    height = 20
    width = 20
    def __init__(self, N, width, height):
        self.num_agents = N
        self.grid = MultiGrid(height, width, True)
        self.schedule = RandomActivation(self)
        # Create agents
        for i in range(self.num_agents):
            a = MoneyAgent(i)
            self.schedule.add(a)
            
            # Add the agent to a random grid cell
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        self.datacollector = DataCollector(
            agent_reporters={"Wealth": lambda a: a.wealth})
            
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

class WolfSheepPredation(Model):
    def __init__(self, height=20, width=20,
                 initial_sheep=100, initial_wolves=50,
                 sheep_reproduce=0.04, wolf_reproduce=0.05,
                 wolf_gain_from_food=20,
                 grass=False, grass_regrowth_time=30, sheep_gain_from_food=4):

        # Set parameters
        self.height = height
        self.width = width
        self.initial_sheep = initial_sheep
        self.initial_wolves = initial_wolves
        self.sheep_reproduce = sheep_reproduce
        self.wolf_reproduce = wolf_reproduce
        self.wolf_gain_from_food = wolf_gain_from_food
        self.grass = grass
        self.grass_regrowth_time = grass_regrowth_time
        self.sheep_gain_from_food = sheep_gain_from_food

        self.schedule = RandomActivationByBreed(self)
        self.grid = MultiGrid(self.height, self.width, torus=True)
        self.datacollector = DataCollector(
            {"Wolves": lambda m: m.schedule.get_breed_count(Wolf),
             "Sheep": lambda m: m.schedule.get_breed_count(Sheep)})

        # Create sheep:
        for i in range(self.initial_sheep):
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            energy = random.randrange(2 * self.sheep_gain_from_food)
            sheep = Sheep((x, y), self, True, energy)
            self.grid.place_agent(sheep, (x, y))
            self.schedule.add(sheep)

        # Create grass patches
        if self.grass:
            for agent, x, y in self.grid.coord_iter():

                fully_grown = random.choice([True, False])

                if fully_grown:
                    countdown = self.grass_regrowth_time
                else:
                    countdown = random.randrange(self.grass_regrowth_time)

                patch = GrassPatch((x, y), self, fully_grown, countdown)
                self.grid.place_agent(patch, (x, y))
                self.schedule.add(patch)

        self.running = True

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)
        if self.verbose:
            print([self.schedule.time,
                   self.schedule.get_breed_count(Wolf),
                   self.schedule.get_breed_count(Sheep)])

    def run_model(self, step_count=200):

        if self.verbose:
            print('Initial number wolves: ',
                  self.schedule.get_breed_count(Wolf))
            print('Initial number sheep: ',
                  self.schedule.get_breed_count(Sheep))

        for i in range(step_count):
            self.step()

        if self.verbose:
            print('')
            print('Final number wolves: ',
                  self.schedule.get_breed_count(Wolf))
            print('Final number sheep: ',
                  self.schedule.get_breed_count(Sheep))

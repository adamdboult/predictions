#######################
# Import dependencies #
#######################
import matplotlib.pyplot as plt
import numpy as np

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule

#################
# Import model #
################
from simLand.agents import simAgent
from simLand.model import simModel

#############
# Run model #
#############
def simLandPortrayal(agent):
    if agent is None:
        return

    portrayal = {"Shape": "circle",
                 "Filled": "true"}

    if type(agent) is simAgent:
        portrayal["Color"] = "#666666"
        portrayal["r"] = 0.8
        portrayal["Layer"] = 1

    elif type(agent) is Wolf:
        portrayal["Color"] = "#AA0000"
        portrayal["r"] = 0.5
        portrayal["Layer"] = 2
        portrayal["text"] = round(agent.energy, 1)
        portrayal["text_color"] = "Yellow"

    elif type(agent) is GrassPatch:
        if agent.fully_grown:
            portrayal["Color"] = "#00AA00"
        else:
            portrayal["Color"] = "#D6F5D6"
        portrayal["Shape"] = "rect"
        portrayal["Layer"] = 0
        portrayal["w"] = 1
        portrayal["h"] = 1

    return portrayal

canvas_element = CanvasGrid(simLandPortrayal, 20, 20, 500, 500)
chart_element = ChartModule([{"Label": "Wolves", "Color": "#AA0000"},
                             {"Label": "Sheep", "Color": "#666666"}])

server = ModularServer(simModel, [canvas_element, chart_element],
                       "simModel")

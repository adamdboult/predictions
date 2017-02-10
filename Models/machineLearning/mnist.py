#!/usr/bin/env python3
#############
# Libraries #
#############
import numpy as np
from numpy import inf
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import json
import cPickle
import gzip

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

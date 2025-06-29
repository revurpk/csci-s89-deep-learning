import math
from random import random

def rms_loss(y_true, y_pred):
    return math.sqrt((y_true - y_pred) ** 2)

def relu(x):
    """
    relu activation function
    :param x: independent variable
    :return: f(x) = x > 0 ? x : 0
    """
    if x <= 0:
        return 0
    return x

def grad_relu(x):
    """
    first derivative of relu activation f(x) at x
    :param x: input value where f'(x) is evaluated
    :return: f'(x) = x > 0 ? 1 : 0
    """
    if x > 0:
        return 1
    return 0

class Node:
    def __init__(self, input=None):
        if input is None:
            self.input = None
            self.output = float(0.0)
        else:
            self.input = input
            # output = activation function applied to input
            self.output = relu(self.input)

    def __repr__(self):
        return "input: {}, output {}".format(self.input, self.output)

    def setInput(self, input):
        self.input = input
        self.output = relu(self.input)

class Layer:
    def __init__(self, n=None):
        self.nodes = []
        if n is not None:
            for i in range(n):
                self.nodes.append(Node())

    def addNode(self, node):
        self.nodes.append(node)

class Connection:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.weight = random()

    def setWeight(self, weight):
        self.weight = weight

    def connect(self, node1, node2):
        self.node1 = node1
        self.node2 = node2

# build layers and connect

layerIn = Layer(3) # including bias node
layerHidden = Layer(3)  # hidden including bias
layerOut = Layer(1) # output

network = [layerIn, layerHidden, layerOut]

weights = []

# connect input layer to hidden layer
for i in range(len(layerIn.nodes)):
    for j in range(len(layerHidden.nodes)):
        if j != 0:  # skip bias node
            nodeFrom = layerIn.nodes[i]
            nodeTo = layerHidden.nodes[j]
            c = Connection(nodeFrom, nodeTo)
            weights.append(c)

# connect to hidden layer to output layer
for i in range(len(layerHidden.nodes)):
    for j in range(len(layerOut.nodes)):
        nodeFrom = layerHidden.nodes[i]
        nodeTo = layerOut.nodes[j]
        c = Connection(nodeFrom, nodeTo)
        weights.append(c)

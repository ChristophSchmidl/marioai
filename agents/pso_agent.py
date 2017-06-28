import numpy as np
import random
import marioai


__all__ = ['PSO_agent']


class PSO_net():

    def __init__(self, node_list):
        """
        Create the network, the values of the weights and biases are randomized
        
        :param size: a list containg the number of nodes in each layer
        """

        self.nodes = node_list
        self.W = [np.random.uniform(low=-0.5, high=0.5, size=(y, x)) for x, y in zip(node_list[:-1], node_list[1:])]
        self.B = [np.random.uniform(low=-0.5, high=0.5, size=(y,)) for y in node_list[1:]]


    def forward(self, x):
        """
        For a given input x, calculate the forward pass through the network and return the last layer
        
        :return: the values of the last layer
        """

        for W, B in zip(self.W, self.B):
            x = np.tanh(np.dot(W, x) + B)

        return x


    def nn2vec(self):
        """
        Create a vector from the weights and biases. 
        
        :return: the freaking vector
        """

        W_vec = np.array([])
        B_vec = np.array([])
        for W, B in zip(self.W, self.B):
            W_vec = np.hstack((W_vec, W.flatten()))
            B_vec = np.hstack((B_vec, B.flatten()))

        vec = np.hstack((W_vec, B_vec))
        return vec


    def vec2nn(self, vec):
        """
        Create the neural network from a vector
        
        :param vec: the vector
        :return: 
        """

        node_list = self.nodes
        W_vec, B_vec = np.split(vec, [len(vec) - np.sum(node_list[1:])])
        self.W = []
        self.B = []
        W_mem = 0
        B_mem = 0
        for x, y in zip(node_list[:-1], node_list[1:]):
            self.W.append(np.array(W_vec[W_mem:(W_mem + x*y)]).reshape((y, x)))
            W_mem += x*y

            self.B.append(B_vec[B_mem:(B_mem + y)])
            B_mem += y



class PSO_agent(marioai.Agent):

    def set_nn(self):
        self.nn = PSO_net([7*4 + 2, 5])
        self.prev_input = None
        self.action = None
        x = self.nn.nn2vec()
        self.nn.vec2nn(x)

    def act(self):
        if self.level_scene is None:
            return [0,0,0,0,0]

        # Obtain the input
        nn_input = self.level_scene[8:15, 10:14]
        nn_input[nn_input == -10] = 1
        nn_input[nn_input == 20] = 1
        nn_input[nn_input == -11] = 0
        nn_input[nn_input > 1] = 1
        nn_input = nn_input.flatten().astype(float)
        nn_input = np.hstack((nn_input, np.array([float(self.can_jump), float(self.on_ground)])))


        if (self.prev_input is not None) and (nn_input == self.prev_input).all():
            nn_output = self.action
        else:
            nn_output = np.array(self.nn.forward(nn_input))
            nn_output = (nn_output > 0.5).astype(int)

        self.prev_input = nn_input
        self.action = nn_output

        return nn_output
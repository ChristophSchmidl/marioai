import random
import marioai
import neat
import numpy as np

__all__ = ['NeatAgent']

class NeatAgent(marioai.Agent):
    def set_neuralnet(self, ff_neuralnet):
        self.ff_neuralnet = ff_neuralnet
        self.prev_input = None
        self.action = None

    def act(self):
        nn_input = self.level_scene.flatten()

        # get neural net output from memory if the envirionment is the same as in the last state
        # otherwise make a forward pass through the network
        if (self.prev_input is not None) and (nn_input == self.prev_input).all():
            nn_output = self.action
        else:
            nn_output = np.array(self.ff_neuralnet.activate(nn_input))
            nn_output = (nn_output > 0.5).astype(float)
        
        self.prev_input = nn_input
        self.action = nn_output
        return nn_output.tolist()
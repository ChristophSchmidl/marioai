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
        if self.level_scene is None:
            return [0,0,0,0,0]
        nn_input = self.level_scene[8:15,12:]
        nn_input[nn_input == -10] = 1
        nn_input[nn_input == 20] = 1
        nn_input[nn_input == -11] = 0
        nn_input[nn_input > 1] = 1
        #print nn_input
        nn_input = nn_input.flatten().astype(float)
        #if nn_input.std() != 0.0:
        #    nn_input = (nn_input - nn_input.mean()) / nn_input.std()
        # get neural net output from memory if the envirionment is the same as in the last state
        # otherwise make a forward pass through the network
        if (self.prev_input is not None) and (nn_input == self.prev_input).all():
            nn_output = self.action
        else:
            nn_output = np.array(self.ff_neuralnet.activate(nn_input))
            nn_output = (nn_output > 0.5).astype(int)
        
        self.prev_input = nn_input
        self.action = nn_output
        return [0, nn_output[0], 0, nn_output[1], 0]
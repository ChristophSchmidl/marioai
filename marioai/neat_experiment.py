import time
import os
import neat

__all__ = ['NeatExperiment']

def mario_activation(z):
    return max(0.0, min(1.0, z))

class NeatExperiment(object):
    '''Episodic Experiment'''
    
    def __init__(self, task, agent):
        self.task = task
        self.agent = agent
        self.max_fps = -1
        
        config_path = os.path.join('agents', 'neat_config.conf')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)
        config.genome_config.add_activation('mario_activation', mario_activation)

        self.p = neat.Population(config)
        
        # Add a stdout reporter to show progress in the terminal.
        self.p.add_reporter(neat.StdOutReporter(True))

    def _step(self):
        self.agent.sense(self.task.get_sensors())
        self.task.perform_action(self.agent.act())
        self.agent.give_rewards(self.task.reward, self.task.cum_reward)
    
    def _episode(self):
        self.agent.reset()
        self.task.reset()

        while not self.task.finished:
            start_time = time.time()
            self._step()

            if self.max_fps > 0:
                time.sleep(start_time + 1./self.max_fps - time.time())

    def doEpisodes(self, genomes, config):
        for genome_id, genome in genomes:
            ff_neuralnet = neat.nn.FeedForwardNetwork.create(genome, config)
            self.agent.set_neuralnet(ff_neuralnet)
            self._episode()
            #reward = self.task.reward + self.task.coins
            reward = self.task.reward
            print "Reward for episode ", genome_id, ": ", reward
            genome.fitness = reward
import time
import numpy as np

__all__ = ['PSOExperiment']


class PSOExperiment(object):
    '''Episodic Experiment'''

    def __init__(self, task, agent):
        self.task = task
        self.agent = agent
        self.max_fps = -1

    def _step(self):
        self.agent.sense(self.task.get_sensors())
        self.task.perform_action(self.agent.act())
        self.agent.give_rewards(self.task.reward, self.task.cum_reward)
        return self.task.reward

    def _episode(self):
        rewards = []

        self.agent.reset()
        self.task.reset()
        while not self.task.finished:
            start_time = time.time()
            r = self._step()
            rewards.append(r)

            # if self.max_fps > 0:
            #     time.sleep(start_time + 1. / self.max_fps - time.time())

        return rewards

    def doEpisodes(self, n=1, m=20):
        """
        
        :param n: number of iterations
        :param m: number of particles
        :return: 
        """

        # Set a couple of constants
        omega = 0.5
        alpha_1 = 2
        alpha_2 = 2

        # Initialize the variables
        global_reward = [0]
        local_reward = np.zeros(shape=(m,))
        temp_reward = np.zeros(shape=(m,))
        global_weight = np.zeros(shape=(30*3 + 3))
        weights = np.random.uniform(low=-0.5, high=0.5, size=(m, 30*3 + 3))
        local_weights = weights
        velocity = np.zeros(shape=(m, 30*3 + 3))

        for run in xrange(n):
            # Disclaimer, the code is 'suboptimal' some of the for loops could
            # be merged, but that would comprise the readability, so I opted against it

            print 'This is run: ', run, ' with ', m, ' particles'
            t0 = time.time()
            # Calculate/simulate the perfomance
            for particle in xrange(m):

                # Initialize the particle
                self.agent.set_nn()
                self.agent.nn.vec2nn(weights[particle,:])

                # Run the network
                self._episode()

                # Obtain performance
                temp_reward[particle] = -self.task.reward

            # Update local best
            for particle in xrange(m):
                if temp_reward[particle] < local_reward[particle]:
                    local_weights[particle, :] = weights[particle, :]
                    local_reward[particle] = temp_reward[particle]

            # Get the best particle
            best_particle = np.argmin(temp_reward)
            print best_particle, temp_reward[best_particle], '\n', temp_reward
            if temp_reward[best_particle] < np.min(global_reward):
                print 'global best updated'
                global_weight = weights[best_particle, :]
            global_reward.append(temp_reward[best_particle])

            # Show the best particle of the run
            self.task.env.visualization = True
            self.agent.set_nn()
            self.agent.nn.vec2nn(weights[best_particle, :])
            self._episode()
            print self.task.reward
            self.task.env.visualization = False

            # Update the velocities and the weights
            r_1 = np.random.uniform(low=0, high=1)
            r_2 = np.random.uniform(low=0, high=1)
            for particle in xrange(m):
                velocity[particle] = omega*velocity[particle] + alpha_1*r_1*(local_weights[particle, :] - weights[particle, :]) + alpha_2*r_2*(global_weight - weights[particle, :])
                weights[particle] = weights[particle] + velocity[particle]

            print 'average: ', np.average(temp_reward), 'best: ', np.min(temp_reward), 'this run took', time.time() - t0, ' seconds \n'

        return global_reward
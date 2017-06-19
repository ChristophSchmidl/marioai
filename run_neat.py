import os
import marioai
import agents
import neat
import visualize



def main():
    # setup our main classes
    agent = agents.NeatAgent()
    task = marioai.NeatTask(name="NeatAgent", visualization=True)
    exp = marioai.NeatExperiment(task, agent)

    # set some environment vars
    exp.max_fps = 24
    task.env.level_type = 0

    # load from checkpoint, uncomment if running from scratch
    exp.p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-14')

    # get the statistics of NEAT during training
    stats = neat.StatisticsReporter()
    exp.p.add_reporter(stats)
    exp.p.add_reporter(neat.Checkpointer(5))

    # print some statistics at the end
    winner = exp.p.run(exp.doEpisodes, 300)
    print('\nBest genome:\n{!s}'.format(winner))

    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

if __name__ == '__main__':
    main()

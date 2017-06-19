import marioai
import agents


def main():
    agent = agents.PSO_agent()
    task = marioai.PSOTask()
    exp = marioai.PSOExperiment(task, agent)

    exp.max_fps = 24
    task.env.level_type = 0
    exp.doEpisodes(n=10, m=50)


if __name__ == '__main__':
    main()
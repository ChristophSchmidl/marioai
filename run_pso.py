import marioai
import agents
import numpy as np
import matplotlib.pyplot as plt

def main():
    agent = agents.PSO_agent()
    task = marioai.PSOTask()
    exp = marioai.PSOExperiment(task, agent)

    exp.max_fps = 24
    task.env.level_type = 0

    #Obtain graph with global and average reward preformance
    trials = 5
    generations = 10
    global_reward = np.zeros(shape=(trials, generations))
    average_reward = np.zeros(shape=(trials, generations))
    for i in range(trials):
        print i, '\n'
        global_reward[i, :], average_reward[i, :] = exp.doEpisodes(n=generations, m=200, omega=0.5, alpha_1=1.5, alpha_2=1.5)

    print average_reward
    glb_avg = np.average(-global_reward, axis = 0)
    glb_std = np.std(-global_reward, axis = 0)
    avg_avg = np.average(-average_reward, axis = 0)
    avg_std = np.std(-average_reward, axis = 0)


    plt.errorbar(range(generations), glb_avg, yerr=glb_std)
    plt.errorbar(range(generations), avg_avg, yerr=avg_std)
    plt.savefig('avg_glb_gen.png')

    print 'done with first part'

    # Obtain the contour plot
    trials = 6*9
    x_axis = []
    y_axis = []
    gl_results_best = np.zeros(shape=(trials,))
    gl_results_last = np.zeros(shape=(trials,))
    avg_results_best = np.zeros(shape=(trials,))
    avg_results_last = np.zeros(shape=(trials,))
    for omega_large in range(6):
        omega = omega_large*1/5.
        for alpha_large in range(9):
            alpha = alpha_large*2/8.
            print '\n', 'currently running with ', omega_large*9 + alpha_large
            x_axis.append(omega)
            y_axis.append(alpha)
            for i in range(1):
                global_reward, average_reward = exp.doEpisodes(n=5, m=200, omega=omega, alpha_1=alpha, alpha_2=alpha)
                gl_results_best[omega_large*9 + alpha_large] += min(global_reward)
                gl_results_last[omega_large*9 + alpha_large] += global_reward[-1]
                avg_results_best[omega_large*9 + alpha_large] += min(average_reward)
                avg_results_last[omega_large*9 + alpha_large] += average_reward[-1]

    gl_results_best = gl_results_best/1
    gl_results_last = gl_results_last/1
    avg_results_best = avg_results_best/1
    avg_results_last = avg_results_last/1

    print gl_results_best

    f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].tripcolor(x_axis, y_axis, -gl_results_last)
    ax[1].tricontourf(x_axis, y_axis, -gl_results_last, 20)
    plt.savefig('gl_results_last.png')

    f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].tripcolor(x_axis, y_axis, -gl_results_best)
    ax[1].tricontourf(x_axis, y_axis, -gl_results_best, 20)
    plt.savefig('gl_results_best.png')


    f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].tripcolor(x_axis, y_axis, -avg_results_last)
    ax[1].tricontourf(x_axis, y_axis, -avg_results_last, 20)
    plt.savefig('avg_results_last.png')

    f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].tripcolor(x_axis, y_axis, -avg_results_best)
    ax[1].tricontourf(x_axis, y_axis, -avg_results_best, 20)
    plt.savefig('avg_results_best.png')


if __name__ == '__main__':
    main()
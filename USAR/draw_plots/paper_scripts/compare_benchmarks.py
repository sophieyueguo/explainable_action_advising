import matplotlib
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
from matplotlib import rcParams
rcParams['font.family'] = 'AppleMyungjo'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.ticker

env_type = 'Simple_4_Room'
evaluate_trial_cycle = 10
X_MAX = 500
Y_MAX = 11
exp_num = 3
margin = 0.1


benchmark_types = ['Decay Reusing Probability', 'Q-change Per Step', 'Reusing Budget', 'Learn to Teach']
advice_types = ['Early', 'Alternative', 'Importance', 'Mistake Correcting']
result_path = 'draw_plots/saved_results/no-transfer/' + env_type + '/'
compare_types = ['No Advice', 'Random Action']
colors = {'No Advice': 'blue', 'Random Action': 'pink', 'Early': 'red',
'Alternative': 'purple', 'Importance': 'orange', 'Mistake Correcting': 'green',
'Decay Reusing Probability': 'orange', 'Q-change Per Step': 'brown',
'Reusing Budget': 'purple', 'Learn to Teach': 'olive'}

markers = {'EAA - Best': 's', 'Decay Reusing Probability': '^', 'Q-change Per Step': 'd', 'Reusing Budget': '.', 'Learn to Teach': '*'}
plot_freq = 20

class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format

def smooth(l, r):
    if len(l) < 2:
        return list(l), np.std(l)

    if len(l) < r:
        return [np.mean(l)]* len(l), np.std(l)

    l_ = np.copy(l)
    std = np.zeros(len(l))
    for i in range(1, r):
        l_[i], std[i] = np.mean(l[0: i]), np.std(l[0: i])
    for i in range(len(l)-r, len(l)-1):
        l_[i], std[i] = np.mean(l[i: len(l)]), np.std(l[i: len(l)])
    for i in range(r, len(l)-r):
        l_[i], std[i] = np.mean(l[i-r: i+r]), np.std(l[i-r: i+r])

    return list(l_), list(std)

def process_entire_line(l, p, r):
    l1, s1 = smooth(l[:p], r)
    l2, s2 = smooth(l[p:], r)
    return l1 + l2, s1 + s2

def calc_avg_and_bound(dir, calc_p=True):
    avg_l = []
    avg_p = []
    for exp in range(exp_num):
        l = np.load(dir + str(exp) + '_episode_rewards.npy')
        # if 'Early' in dir:
        #     print (l)
        avg_l.append(l)
        if calc_p:
            p = np.load(str(exp) + '_trials_budget_used_up.npy')[0]
            p = int(p/evaluate_trial_cycle)
            avg_p.append(p)
    # deal with the early convergence...
    max_len = np.max([len(d) for d in avg_l])
    for exp in range(exp_num):
        while len(avg_l[exp]) < max_len:
            # print ('append converge tails', dir)
            avg_l[exp] = np.append(avg_l[exp], avg_l[exp][-1])

    std_l = np.std(avg_l, axis = 0)
    avg_l = np.mean(avg_l, axis = 0)
    if calc_p:
        avg_p = int(np.mean(avg_p))

    x = [i for i in range(0, len(avg_l), plot_freq)]
    if not calc_p:
        avg_p = 0
    avg_l, _ = process_entire_line(avg_l, avg_p, r=30)
    std_l, _ = process_entire_line(std_l, avg_p, r=30)
    y1 = [avg_l[i] + std_l[i] for i in range(len(avg_l))]
    y2 = [avg_l[i] - std_l[i] for i in range(len(avg_l))]
    y1 = [y1[i] + margin for i in range(0, len(y1), plot_freq)]
    y2 = [y2[i] - margin for i in range(0, len(y2), plot_freq)]


    return avg_l, avg_p, x, y1, y2

fig, axes = plt.subplots(2, 1, figsize=(9,7))
for p_id, ax in enumerate(axes):
    # result_path = env_type + '/'
    print(">> Starting: "+env_type)
    for c in benchmark_types:
        dir = result_path + c + '/'

        avg_l, avg_p, x, y1, y2 = calc_avg_and_bound(dir, calc_p=False)
        ax.plot(np.asarray(range(len(avg_l[:x[-1]]))) * evaluate_trial_cycle, avg_l[:x[-1]], '-.', color=colors[c], alpha=0.4, label=c)
        # ax.scatter(np.asarray(x)  * evaluate_trial_cycle, [avg_l[i] for i in range(0, len(avg_l), plot_freq)], s=30, marker=markers[c], c=colors[c])
        ax.fill_between(np.asarray(x)  * evaluate_trial_cycle, y1, y2, color=colors[c], alpha=0.1)

        # print (c, max(avg_l))
        for i in range(len(avg_l)):
            if avg_l[i] > 0.95 * max(avg_l) and i > 50:
                print ('reach nearly optimal', i)
                break




    # eaa_median = []
    # for type in advice_types: # tmp for now, for partial finished works
    #     dir = result_path + type + '/heuristic_with_tree_memory/'
    #     l1 = np.load(dir + 'episode_rewards.npy')
    #     p = np.load(dir + 'trials_budget_used_up.npy')[0]
    #     p = int(p/evaluate_trial_cycle)
    #     l1, std = process_entire_line(l1, 0)
    #
    #     x = [i for i in range(0, len(l1), r)]
    #     eaa_median.append(l1)
    #     # plt.plot(l1[:x[-1]], '--', color='red', alpha=0.4)
    #     # plt.scatter(x, [l1[i] for i in range(0, len(l1), 20)], s=30, marker=markers[type], c='red', label='EAA - ' + type + 'Advising')
    # eaa_median = np.max(eaa_median, axis=0)
    # eaa_median_dot = [eaa_median[i] for i in range(0, len(eaa_median), r)]
    # plt.plot(eaa_median[:x[-1]], '--', color='red', alpha=0.4)
    # # print ('len(x)', len(x), 'len(eaa_median_dot)', len(eaa_median_dot))
    # x = x[0: len(eaa_median_dot)] # incase eaa is shorter...
    # plt.scatter(x, eaa_median_dot, s=30, marker='s', c='red', label='EAA - Best')

    eaa_median = []
    y1_median = []
    y2_median = []
    min_len_1 = np.inf
    min_len_2 = np.inf
    for type in advice_types:
        dir = result_path + type + '/heuristic_with_tree_memory/'
        avg_l, avg_p, x, y1, y2 = calc_avg_and_bound(dir, calc_p=False)
        if len(avg_l) < min_len_1:
            min_len_1 = len(avg_l)
        if len(y1) < min_len_2:
            min_len_2 = len(y1)
        eaa_median.append(avg_l)
        y1_median.append(y1)
        y2_median.append(y2)
    for j in range(len(advice_types)):
        eaa_median[j] = eaa_median[j][0:min_len_1]
        y1_median[j] = y1_median[j][0:min_len_2]
        y2_median[j] = y2_median[j][0:min_len_2]

    eaa_median = np.median(eaa_median, axis=0)
    y1_median = np.median(y1_median, axis=0)
    y2_median = np.median(y2_median, axis=0)
    eaa_median_dot = [eaa_median[i] for i in range(0, len(eaa_median), plot_freq)]
    ax.plot(np.asarray(range(eaa_median[:x[-1]].shape[0])) * evaluate_trial_cycle, eaa_median[:x[-1]], '--', color='red', label='EAA - Median')

    x = x[0: len(eaa_median_dot)]
    # ax.scatter(np.asarray(x)  * evaluate_trial_cycle, eaa_median_dot, s=30, marker='s', c='red')
    ax.fill_between(np.asarray(x)  * evaluate_trial_cycle, y1_median, y2_median, color='red', alpha=0.1)

    print ('eaa median', max(eaa_median))
    for i in range(len(eaa_median)):
        if eaa_median[i] > 0.95 * max(eaa_median) and i > 50:
            print ('reach nearly optimal', i)
            break
    if p_id == 1:
        ax.set_xlabel('Episode', fontsize=25, labelpad=-2)
        ax.legend(loc="upper center", bbox_to_anchor=(0.4225, 1.675), ncol=2, fancybox=True, framealpha=1.0, fontsize=18)
    ax.set_ylabel('Reward', fontsize=25, labelpad=-15)
    ax.set_yticks([0, 5, 10])
    # ax.set_ylim([-1, Y_MAX])
    ax.set_xlim([0, X_MAX * evaluate_trial_cycle])
    ax.xaxis.set_major_formatter(OOMFormatter(3, "%01d"))

    if p_id == 0:
        ax.set_ylabel('Reward', fontsize=25, labelpad=-14)
        ax.annotate('(a) Four Room Environment', (0.25,0), (0, -25), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=25)
    elif p_id == 1:
        ax.annotate('(b) Fourteen Room Environment', (0.175,0), (0, -50), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=25)

    # Do it again with the other setup
    env_type = '14_Room'
    evaluate_trial_cycle = 40
    X_MAX = 400
    Y_MAX = 11
    exp_num = 2
    margin = 0.3

plt.subplots_adjust(left=0.085, right=0.98, top=0.998, bottom=0.15, wspace=0.975, hspace=0.975)
plt.savefig('draw_plots/plots/benchmarks.png', format='png')
plt.savefig('draw_plots/plots/benchmarks.pdf', format='pdf')

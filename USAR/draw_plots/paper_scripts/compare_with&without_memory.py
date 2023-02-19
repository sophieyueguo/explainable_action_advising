import matplotlib
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
from matplotlib import rcParams
rcParams['font.family'] = 'AppleMyungjo'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker
import numpy as np
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)


# env_type = '14_Room'
# evaluate_trial_cycle = 40
# X_MAX = 350
# Y_MAX = 10.5
# Y_MIN = 2
# exp_num = 2
# SMOOTH = 10

env_type = 'Simple_4_Room'
evaluate_trial_cycle = 10
X_MAX = 350
Y_MAX = 10.5
Y_MIN = 4
exp_num = 3
SMOOTH = 10

plot_freq = 20

result_path = 'draw_plots/saved_results/no-transfer/' + env_type + '/'

compare_types = ['No Advice', 'Random Action']
advice_types = ['Early', 'Alternative', 'Importance', 'Mistake Correcting']
colors = {'No Advice': 'grey', 'Random Action': 'pink', 'Early': 'red',
'Alternative': 'purple', 'Importance': 'orange', 'Mistake Correcting': 'green',
'Decay Reusing Probability': 'orange', 'Q-change Per Step': 'brown', 'Reusing Budget': 'cyan'}


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

def smooth(l, r=30):
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

def process_entire_line(l, p, r=30):
    l1, s1 = smooth(l[:p], r)
    l2, s2 = smooth(l[p:], r)
    return l1 + l2, s1 + s2


def calc_avg_and_bound(dir, calc_p=True, smooth_factor=10):
    avg_l = []
    avg_p = []
    for exp in range(exp_num):
        l = np.load(dir + str(exp) + '_episode_rewards.npy')
        avg_l.append(l)
        if calc_p:
            p = np.load(dir + str(exp) + '_trials_budget_used_up.npy')[0]
            p = int(p/evaluate_trial_cycle)
            avg_p.append(p)
    # deal with the early convergence...
    max_len = np.max([len(d) for d in avg_l])
    for exp in range(exp_num):
        while len(avg_l[exp]) < max_len:
            print ('append converge tails', dir)
            avg_l[exp] = np.append(avg_l[exp], avg_l[exp][-1])




    std_l = np.std(avg_l, axis = 0)
    avg_l = np.mean(avg_l, axis = 0)

    if 'Early' in dir and 'Heusitic' not in dir:
        print (avg_l[1:15])

    if calc_p:
        avg_p = int(np.mean(avg_p))

    x = [i for i in range(0, len(avg_l), plot_freq)]
    if not calc_p:
        avg_p = 0
    avg_l, _ = process_entire_line(avg_l, avg_p, r=smooth_factor)
    std_l, _ = process_entire_line(std_l, avg_p, r=smooth_factor)
    y1 = [avg_l[i] + std_l[i] for i in range(len(avg_l))]
    y2 = [avg_l[i] - std_l[i] for i in range(len(avg_l))]
    y1 = [y1[i] for i in range(0, len(y1), plot_freq)]
    y2 = [y2[i] for i in range(0, len(y2), plot_freq)]

    return avg_l, avg_p, x, y1, y2

fig, axes = plt.subplots(1, 4, sharey=True, figsize=(18,3.5))
for type, ax, p_id in list(zip(advice_types, axes, range(len(advice_types)))):
    print ()
    print (type)

    dir = result_path + type + '/heuristic_with_tree_memory/'
    avg_l, avg_p, x, y1, y2 = calc_avg_and_bound(dir, smooth_factor=SMOOTH)
    inset_eaa_l, inset_eaa_p, _, _, _ = calc_avg_and_bound(dir, smooth_factor=1)
    legend_label = type
    if type == "Mistake Correcting":
        legend_label = "MC"
    else:
        legend_label = type[0]
    ax.plot(np.asarray(range(len(avg_l[:x[-1]]))) * evaluate_trial_cycle, avg_l[:x[-1]], '-', color=colors[type], label='EAA-' + legend_label, alpha=0.5)
    # ax.scatter(x, [avg_l[i] for i in range(0, len(avg_l), plot_freq)], s=30, marker='', c=colors[type])
    ax.vlines(avg_p * evaluate_trial_cycle, ymin=-1, ymax=Y_MAX, linestyles='dashdot', color=colors[type])
    ax.fill_between(np.asarray(x)  * evaluate_trial_cycle, y1, y2, color=colors[type], alpha=0.15)

    # print ('EAA', int(avg_p))
    # print ('x', x)
    candidate_avg_numbers = [avg_l[i] for i in range(0, len(avg_l), plot_freq)]
    after_budget_used_up = []
    for i in range(len(x)):
        if x[i] > avg_p:
            after_budget_used_up.append(candidate_avg_numbers[i])
    # print ('EAA after_budget_used_up', after_budget_used_up)
    for i in range(len(after_budget_used_up)):
        if after_budget_used_up[i] > 0.95 * max(after_budget_used_up):
            print ('EAA', i * plot_freq)
            break

    dir = result_path + type + '/heuristic_with_no_memory/'
    avg_l, avg_p, x, y1, y2 = calc_avg_and_bound(dir, smooth_factor=SMOOTH)
    inset_aa_l, inset_aa_p, _, _, _ = calc_avg_and_bound(dir, smooth_factor=1)
    ax.plot(np.asarray(range(len(avg_l[:x[-1]]))) * evaluate_trial_cycle, avg_l[:x[-1]], '-', color='blue', label='AA-' + legend_label, alpha=0.5)
    # ax.scatter(x, [avg_l[i] for i in range(0, len(avg_l), plot_freq)], s=30, marker='', c='blue')
    ax.vlines(avg_p * evaluate_trial_cycle, ymin=-1, ymax=Y_MAX, linestyles='dashdot', color='blue')
    ax.fill_between(np.asarray(x)  * evaluate_trial_cycle, y1, y2, color='blue', alpha=0.15)

    # print ('raw', int(avg_p))
    # print ('x', x)
    candidate_avg_numbers = [avg_l[i] for i in range(0, len(avg_l), plot_freq)]
    after_budget_used_up = []
    for i in range(len(x)):
        if x[i] > avg_p:
            after_budget_used_up.append(candidate_avg_numbers[i])
    # print ('Heuristic after_budget_used_up', after_budget_used_up)
    for i in range(len(after_budget_used_up)):
        if after_budget_used_up[i] > 0.95 * max(after_budget_used_up):
            print ('Heuristic', i * plot_freq)
            break


    ax.set_ylim([Y_MIN, Y_MAX])
    ax.set_yticks(np.arange(Y_MIN, Y_MAX))
    ax.set_xlim([0, X_MAX])
    # plt.title(env_type + ' EAA-' + type, fontsize=18)
    ax.set_xlabel('Episode',fontsize=25, labelpad=-3)
    ax.set_xlim([0, X_MAX * evaluate_trial_cycle])
    ax.xaxis.set_major_formatter(OOMFormatter(3, "%01d"))
    if p_id == 0:
        ax.set_ylabel('Reward',fontsize=25, labelpad=-14)
    if p_id == 0 and env_type == "Simple_4_Room":
        ax2 = plt.axes([0,0,1,1])
        ip = InsetPosition(ax, [0.15,0.01,0.4,0.4]) #px, py, w, h
        ax2.set_axes_locator(ip)
        mark_inset(ax, ax2, loc1=1, loc2=3, fc="none", ec='0.5', lw=2)
        ax2.axes.xaxis.set_visible(False)
        ax2.axes.yaxis.set_visible(False)

        cuttoff = 20
        ax2.plot(np.asarray(range(len(inset_eaa_l[:cuttoff]))) * evaluate_trial_cycle, inset_eaa_l[:cuttoff], '-', color=colors[type], label='EAA-' + legend_label, alpha=0.5)
        ax2.vlines(inset_eaa_p * evaluate_trial_cycle, ymin=5, ymax=10, linestyles='dashdot', color=colors[type])

        ax2.plot(np.asarray(range(len(inset_aa_l[:cuttoff]))) * evaluate_trial_cycle, inset_aa_l[:cuttoff], '-', color='blue', label='AA-' + legend_label, alpha=0.5)
        ax2.vlines(inset_aa_p * evaluate_trial_cycle, ymin=5, ymax=10, linestyles='dashdot', color='blue')

        rect = patches.Rectangle((0, 4.75), cuttoff * evaluate_trial_cycle, 5.5, linewidth=3, edgecolor='black', facecolor='none', alpha=0.5)
        ax.add_patch(rect)

    ax.legend(loc="lower right", ncol=1, fancybox=True, framealpha=1.0, fontsize=18)
    ax.grid(axis="y", linestyle='-.', alpha=0.4)

    if p_id == 0:
        ax.set_ylabel('Reward', fontsize=25, labelpad=-14)
        ax.annotate('(a) ' + type, (0.3,0), (0, -50), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=25)
    elif p_id == 1:
        ax.annotate('(b) ' + type, (0.2,0), (0, -50), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=25)
    elif p_id == 2:
        ax.annotate('(c) ' + type, (0.18,0), (0, -50), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=25)
    elif p_id == 3:
        ax.annotate('(d) ' + type, (0.075,0), (0, -50), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=25)


# global legend
# lines_labels = [ax.get_legend_handles_labels() for ax in axes]
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# fig.legend(lines, labels)

plt.subplots_adjust(left=0.033, right=0.999, top=0.998, bottom=0.3, wspace=0.05)
plt.savefig("draw_plots/plots/EAA_Methods_{}.png".format(env_type), format='png', dpi=200)
plt.savefig("draw_plots/plots/EAA_Methods_{}.pdf".format(env_type), format='pdf')

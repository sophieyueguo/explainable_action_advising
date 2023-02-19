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

env_type = 'Simple_4_Room'
X_max = 31
Y_max = 11
result_path = 'draw_plots/saved_results/transfer/'

ext1 = '/2-no_advice_no_init/'
ext2 = '/2-no_advice_init_explore/'

evaluate_trial_cycle = 1000



plot_freq = 1

R = 3

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

def process_entire_line(l, p, r=R):
    l1, s1 = smooth(l[:p], r)
    l2, s2 = smooth(l[p:], r)
    return l1 + l2, s1 + s2


def calc_avg_and_bound(ls, calc_p=True, extra=None):
    avg_l = []
    avg_p = []
    # for exp in range(exp_num):
        # l = np.load(dir + str(exp) + '_episode_rewards.npy')
        # print (l)
    for l in ls:
        if extra != None: #hand record numbers...
            l = np.insert(l, 0, extra)

        avg_l.append(l)
        if calc_p:
            # p = np.load(dir + str(exp) + '_trials_budget_used_up.npy')[0]
            p = 0
            p = int(p/evaluate_trial_cycle)
            avg_p.append(p)

    # # deal with the early convergence...
    # max_len = np.max([len(d) for d in avg_l])
    # for exp in range(exp_num):
    #     while len(avg_l[exp]) < max_len:
    #         print ('append converge tails', dir)
    #         avg_l[exp] = np.append(avg_l[exp], avg_l[exp][-1])


    std_l = np.std(avg_l, axis = 0)
    avg_l = np.mean(avg_l, axis = 0)
    if calc_p:
        avg_p = int(np.mean(avg_p))


    if not calc_p:
        avg_p = 0
    # avg_l, _ = process_entire_line(avg_l, avg_p)
    # std_l, _ = process_entire_line(std_l, avg_p)

    return avg_l, avg_p, std_l


def find_x_y(avg_l, std_l):
    x = [i for i in range(0, len(avg_l), plot_freq)]
    y1 = [avg_l[i] + std_l[i] for i in range(len(avg_l))]
    y2 = [avg_l[i] - std_l[i] for i in range(len(avg_l))]
    y1 = [y1[i] for i in range(0, len(y1), plot_freq)]
    y2 = [y2[i] for i in range(0, len(y2), plot_freq)]

    return x, y1, y2



def abbre_vec(l, freq=5):
    ind = list(range(0, len(l), freq))
    return np.array(l)[ind]




EAAs = ['Early', 'Alternative', 'Importance', 'Mistake Correcting']
EAA_colors = {ext1: {'Early': 'red', 'Alternative': 'purple',
'Importance': 'orange', 'Mistake Correcting': 'green'},
ext2: {'Early': 'blue', 'Alternative': 'blue',
'Importance': 'blue', 'Mistake Correcting': 'blue'}}
fig, axes = plt.subplots(1, 4, sharey=True, figsize=(18,3.5))
for e, ax, p_id in list(zip(EAAs, axes, [0, 1, 2, 3])):
    for ext in [ext1, ext2]:
        dir = result_path + env_type + ext + e + '_'
        ls = []
        for exp in range(5):
        # for exp in [2]:
            if exp != '':
                l = np.load(dir + str(exp) + '/episode_rewards.npy')
                l = np.delete(l, 0)#ignore the repetivie first one...
                if ext == '/2-no_advice_no_init/' or ext == '/2-no_advice_init_explore/':
                    l = l[20:]
                ls.append(l)
        plt_label = ""
        if ext == "/2-no_advice_init_explore/":
            plt_label = " (Explore)"


        avg_l, avg_p, std_l = calc_avg_and_bound(ls)
        x, y1, y2 = find_x_y(avg_l, std_l)
        ax.plot(np.asarray(range(len(avg_l))) * evaluate_trial_cycle, avg_l, '-', color=EAA_colors[ext][e], label="EAA-" + (e[0] if e[0] != "M" else "MC") + plt_label)
        # plt.scatter(x, [avg_l[i] for i in range(0, len(avg_l), plot_freq)], s=30, marker='o', c=EAA_colors[ext][e])
        ax.fill_between(np.asarray(x)  * evaluate_trial_cycle, y1, y2, color=EAA_colors[ext][e], alpha=0.2)



    ax.set_ylim([0, Y_max])
    ax.set_yticks(np.arange(0, Y_max, 2))
    ax.set_xlim([-1, X_max * evaluate_trial_cycle])
    ax.set_xlabel('Episode', fontsize=25, labelpad=-3)
    ax.xaxis.set_major_formatter(OOMFormatter(3, "%01d"))
    ax.grid(axis="y", linestyle='-.', alpha=0.4)
    if p_id == 0:
        ax.set_ylabel('Reward', fontsize=25, labelpad=-14)
        ax.annotate('(a) EAA Early', (0.2,0), (0, -50), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=25)
    elif p_id == 1:
        ax.annotate('(b) EAA Alternative', (0.1,0), (0, -50), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=25)
    elif p_id == 2:
        ax.annotate('(c) EAA Importance', (0.12,0), (0, -50), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=25)
    elif p_id == 3:
        ax.annotate('(d) EAA MC', (0.25,0), (0, -50), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=25)
    ax.legend(loc="lower right", ncol=1, fancybox=True, framealpha=1.0, fontsize=18)

plt.subplots_adjust(left=0.035, right=0.999, top=0.998, bottom=0.3, wspace=0.05)
plt.savefig('draw_plots/plots/no_teacher.png', format='png')
plt.savefig('draw_plots/plots/no_teacher.pdf', format='pdf')

    # plt.savefig('publish_results/plots/' + env_type + 'no_rubble' + type +'.png')

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
X_max = 29
Y_max = 11
result_path = 'draw_plots/saved_results/transfer/'
# ext = '/3-dt/'
ext = '/1-dt+nn/'

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
    avg_l, _ = process_entire_line(avg_l, avg_p)
    std_l, _ = process_entire_line(std_l, avg_p)

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

fig, ax = plt.subplots(figsize=(9,5))

baselines = ['scratch', 'NNtransfer', 'pretrain']
b_colors = {'pretrain': 'blue', 'scratch': 'cyan', 'NNtransfer': 'gray'}
b_count = {'pretrain': 5, 'scratch': 5, 'NNtransfer': 3}
for b in baselines:
    print ()
    dir = result_path + env_type + '/common_30000trials/' + b + '_'
    ls = []
    for exp in range(b_count[b]):
        l = np.load(dir + str(exp) + '/episode_rewards.npy')
        l = np.delete(l, 0)#ignore the repetivie first one...
        ls.append(l)
    avg_l, avg_p, std_l = calc_avg_and_bound(ls)
    x, y1, y2 = find_x_y(avg_l, std_l)

    plt_label = b
    if plt_label == "scratch":
        plt_label = "Scratch"
    elif plt_label == "NNtransfer":
        plt_label = "NN Tns."
    elif plt_label == "pretrain":
        plt_label = "NN Pre."
    ax.plot(np.asarray(range(len(avg_l[1:]))) * evaluate_trial_cycle, avg_l[1:], '-', color=b_colors[b], label=plt_label)
    # plt.scatter(x, [avg_l[i] for i in range(0, len(avg_l), plot_freq)], s=30, marker='o', c=b_colors[b])
    ax.fill_between(np.asarray(x)[:-1]  * evaluate_trial_cycle, y1[:-1], y2[:-1], color=b_colors[b], alpha=0.1)

# #-------------------------------------------------------------------------------

EAAs = ['Early', 'Alternative', 'Importance', 'Mistake Correcting']
EAA_colors = {'Early': 'red', 'Alternative': 'purple',
'Importance': 'orange', 'Mistake Correcting': 'green'}
for e in EAAs:
    print ()
    dir = result_path + env_type + ext + e + '_'
    ls = []
    for exp in range(4):
    # for exp in [2]:
        if exp != '':
            l = np.load(dir + str(exp) + '/episode_rewards.npy')
            l = np.delete(l, 0)#ignore the repetivie first one...
            if ext == '/2-no_advice_no_init/publish_results/' or ext == '/2-no_advice_init_explore/publish_results/':
                l = l[20:]
            ls.append(l)


            print (e, l)
    avg_l, avg_p, std_l = calc_avg_and_bound(ls)
    x, y1, y2 = find_x_y(avg_l, std_l)
    ax.plot(np.asarray(range(len(avg_l[1:]))) * evaluate_trial_cycle, avg_l[1:], '-', color=EAA_colors[e], label='EAA-' + e[0])
    # plt.scatter(x, [avg_l[i] for i in range(0, len(avg_l), plot_freq)], s=30, marker='o', c=EAA_colors[e])
    ax.fill_between(np.asarray(x)[:-1]  * evaluate_trial_cycle, y1[:-1], y2[:-1], color=EAA_colors[e], alpha=0.1)


plt.subplots_adjust(left=0.075, right=0.998, top=0.998, bottom=0.14)
plt.ylim([0, Y_max])
plt.xlim([-1, X_max * evaluate_trial_cycle])
ax.xaxis.set_major_formatter(OOMFormatter(3, "%01d"))
plt.xlabel('Episodes', fontsize=25, labelpad=-1)
plt.ylabel('Reward', fontsize=25, labelpad=-12)
plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4, fancybox=True, framealpha=1.0, fontsize=18)
plt.savefig('draw_plots/plots/having_teacher_nn_dt.png', format='png')
plt.savefig('draw_plots/plots/having_teacher_nn_dt.pdf', format='pdf')

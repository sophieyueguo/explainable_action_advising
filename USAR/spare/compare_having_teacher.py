import matplotlib.pyplot as plt
import numpy as np

env_type = 'Simple_4_Room'
X_max = 31
Y_max = 12
result_path = 'draw_plots/saved_results/'

ext = '/1-dt+nn/'
'''Comment out for the 3-dt case'''
# ext = '/3-dt/'

evaluate_trial_cycle = 1000



plot_freq = 1

R = 3

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









plt.figure()


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
        print (b, l)
    avg_l, avg_p, std_l = calc_avg_and_bound(ls)
    x, y1, y2 = find_x_y(avg_l, std_l)

    plt.plot(avg_l, '--', color=b_colors[b], label=b)
    plt.scatter(x, [avg_l[i] for i in range(0, len(avg_l), plot_freq)], s=30, marker='o', c=b_colors[b])
    plt.fill_between(x, y1, y2, color=b_colors[b], alpha=0.2)

# #-------------------------------------------------------------------------------

EAAs = ['Early', 'Alternative', 'Importance', 'Mistake Correcting']
EAA_colors = {'Early': 'red', 'Alternative': 'purple',
'Importance': 'orange', 'Mistake Correcting': 'green'}
for e in EAAs:
    print ()
    dir = result_path + env_type + ext  + e + '_'
    ls = []
    for exp in range(4):
    # for exp in [2]:
        if exp != '':
            l = np.load(dir + str(exp) + '/episode_rewards.npy')
            l = np.delete(l, 0)#ignore the repetivie first one...
            # if ext == '/2-no_advice_no_init/publish_results/' or ext == '/2-no_advice_init_explore/publish_results/':
            #     l = l[20:]
            ls.append(l)


            print (e, l)
    avg_l, avg_p, std_l = calc_avg_and_bound(ls)
    x, y1, y2 = find_x_y(avg_l, std_l)
    plt.plot(avg_l, '--', color=EAA_colors[e], label='EAA-' + e)
    plt.scatter(x, [avg_l[i] for i in range(0, len(avg_l), plot_freq)], s=30, marker='o', c=EAA_colors[e])
    plt.fill_between(x, y1, y2, color=EAA_colors[e], alpha=0.2)



plt.ylim([0, Y_max])
plt.xlim([-1, X_max])
plt.title('USAR Mission on ' + env_type)
plt.xlabel('Episodes * ' + str(evaluate_trial_cycle))
plt.ylabel('Rewards')
plt.legend()
plt.show()
# plt.savefig('publish_results/plots/' + env_type + 'no_rubble' + type +'.png')

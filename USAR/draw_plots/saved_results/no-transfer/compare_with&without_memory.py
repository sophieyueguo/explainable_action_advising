import matplotlib.pyplot as plt
import numpy as np

# env_type = '14_Room'
# evaluate_trial_cycle = 40
# X_MAX = 400
# Y_MAX = 18
# exp_num = 2

env_type = 'Simple_4_Room'
evaluate_trial_cycle = 10
X_MAX = 500
Y_MAX = 18
exp_num = 3

plot_freq = 20

result_path = env_type + '/'

compare_types = ['No Advice', 'Random Action']
advice_types = ['Early', 'Alternative', 'Importance', 'Mistake Correcting']
colors = {'No Advice': 'grey', 'Random Action': 'pink', 'Early': 'red',
'Alternative': 'purple', 'Importance': 'orange', 'Mistake Correcting': 'green',
'Decay Reusing Probability': 'orange', 'Q-change Per Step': 'brown', 'Reusing Budget': 'cyan'}


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


def calc_avg_and_bound(dir, calc_p=True):
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
    avg_l, _ = process_entire_line(avg_l, avg_p)
    std_l, _ = process_entire_line(std_l, avg_p)
    y1 = [avg_l[i] + std_l[i] for i in range(len(avg_l))]
    y2 = [avg_l[i] - std_l[i] for i in range(len(avg_l))]
    y1 = [y1[i] for i in range(0, len(y1), plot_freq)]
    y2 = [y2[i] for i in range(0, len(y2), plot_freq)]

    return avg_l, avg_p, x, y1, y2



print ('!!!!!!!!!!!!!!!!!!')

for type in advice_types:
    plt.figure(figsize=(6, 5))

    print ()
    print (type)

    dir = result_path + type + '/heuristic_with_tree_memory/'
    avg_l, avg_p, x, y1, y2 = calc_avg_and_bound(dir)
    plt.plot(avg_l[:x[-1]], '--', color=colors[type])
    plt.scatter(x, [avg_l[i] for i in range(0, len(avg_l), plot_freq)], s=30, marker='o', c=colors[type], label='EAA - ' + type)
    plt.vlines(avg_p, ymin=-1, ymax=Y_MAX-5, linestyles='dashdot', color='black', label='EAA - ' + type + ' Budget Used Up')
    plt.fill_between(x, y1, y2, color=colors[type], alpha=0.2)

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
    avg_l, avg_p, x, y1, y2 = calc_avg_and_bound(dir)
    plt.plot(avg_l[:x[-1]], ':', color='blue')
    plt.scatter(x, [avg_l[i] for i in range(0, len(avg_l), plot_freq)], s=30, marker='*', c='blue', label='Heuristic - ' + type)
    plt.vlines(avg_p, ymin=-1, ymax=Y_MAX-5, linestyles='dashed', color='grey', label='Heuristic - ' + type + ' Budget Used Up')
    plt.fill_between(x, y1, y2, color='blue', alpha=0.2)

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


    plt.ylim([-1, Y_MAX])
    plt.xlim([0, X_MAX])
    plt.title(env_type + ' EAA-' + type, fontsize=18)
    plt.xlabel('Episodes * ' + str(evaluate_trial_cycle),fontsize=16)
    plt.ylabel('Rewards',fontsize=16)
    plt.legend(loc='upper center', ncol=1, fontsize=12)
    plt.savefig('plots/' + env_type + '_heuristic_memory' + type +'.png', bbox_inches='tight')
    plt.savefig('plots/' + env_type + '_heuristic_memory' + type, bbox_inches='tight', format='svg')

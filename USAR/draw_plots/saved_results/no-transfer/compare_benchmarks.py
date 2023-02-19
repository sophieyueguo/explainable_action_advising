import matplotlib.pyplot as plt
import numpy as np

# env_type = '14_Room'
# evaluate_trial_cycle = 40
# X_MAX = 385
# Y_MAX = 18
# exp_num = 2
# margin = 0.3

env_type = 'Simple_4_Room'
evaluate_trial_cycle = 10
X_MAX = 490
Y_MAX = 18
exp_num = 3
margin = 0.1



benchmark_types = ['Decay Reusing Probability', 'Q-change Per Step', 'Reusing Budget', 'Learn to Teach']
advice_types = ['Early', 'Alternative', 'Importance', 'Mistake Correcting']
result_path = env_type + '/'
compare_types = ['No Advice', 'Random Action']
colors = {'No Advice': 'blue', 'Random Action': 'pink', 'Early': 'red',
'Alternative': 'purple', 'Importance': 'orange', 'Mistake Correcting': 'green',
'Decay Reusing Probability': 'orange', 'Q-change Per Step': 'brown',
'Reusing Budget': 'purple', 'Learn to Teach': 'olive'}

markers = {'EAA - Best': 's', 'Decay Reusing Probability': '^', 'Q-change Per Step': 'd', 'Reusing Budget': '.', 'Learn to Teach': '*'}
plot_freq = 20
r = 30
def smooth(l):
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

def process_entire_line(l, p):
    l1, s1 = smooth(l[:p])
    l2, s2 = smooth(l[p:])
    return l1 + l2, s1 + s2

def calc_avg_and_bound(dir, calc_p=True):
    avg_l = []
    avg_p = []
    for exp in range(exp_num):
        l = np.load(dir + str(exp) + '_episode_rewards.npy')
        if 'Early' in dir:
            print (l)
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
    if calc_p:
        avg_p = int(np.mean(avg_p))

    x = [i for i in range(0, len(avg_l), plot_freq)]
    if not calc_p:
        avg_p = 0
    avg_l, _ = process_entire_line(avg_l, avg_p)
    std_l, _ = process_entire_line(std_l, avg_p)
    y1 = [avg_l[i] + std_l[i] for i in range(len(avg_l))]
    y2 = [avg_l[i] - std_l[i] for i in range(len(avg_l))]
    y1 = [y1[i] + margin for i in range(0, len(y1), plot_freq)]
    y2 = [y2[i] - margin for i in range(0, len(y2), plot_freq)]


    return avg_l, avg_p, x, y1, y2



plt.figure(figsize=(8, 6))






for c in benchmark_types:
    dir = result_path + c + '/'

    avg_l, avg_p, x, y1, y2 = calc_avg_and_bound(dir, calc_p=False)
    plt.plot(avg_l[:x[-1]], '-.', color=colors[c], alpha=0.4)
    plt.scatter(x, [avg_l[i] for i in range(0, len(avg_l), plot_freq)], s=30, marker=markers[c], c=colors[c], label=c)
    plt.fill_between(x, y1, y2, color=colors[c], alpha=0.1)

    print (c, max(avg_l))
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
plt.plot(eaa_median[:x[-1]], '--', color='red')

x = x[0: len(eaa_median_dot)]
plt.scatter(x, eaa_median_dot, s=30, marker='s', c='red', label='EAA - Median')
plt.fill_between(x, y1_median, y2_median, color='red', alpha=0.1)

print ('eaa median', max(eaa_median))
for i in range(len(eaa_median)):
    if eaa_median[i] > 0.95 * max(eaa_median) and i > 50:
        print ('reach nearly optimal', i)
        break



plt.ylim([-1, Y_MAX])
plt.xlim([0, X_MAX])
plt.title('USAR Mission on ' + env_type)
plt.xlabel('Episodes * ' + str(evaluate_trial_cycle))
plt.ylabel('Rewards')
plt.legend(loc='upper center', ncol=2, fontsize=15)
plt.savefig('plots/' + env_type + '_benchmarks' +'.png', bbox_inches='tight')
plt.savefig('plots/' + env_type + '_benchmarks', bbox_inches='tight', format='svg')

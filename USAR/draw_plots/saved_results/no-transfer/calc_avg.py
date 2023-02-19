import numpy as np


# # avg saturn no advice
# path = 'publish_results/A_Saturn_Section/No Advice/'
#
# l1 = np.load(path + 'multiple/new_episode_rewards.npy')
# l2 = np.load(path + 'multiple/old_episode_rewards.npy')
# for i in range(int(len(l2)/4)):
#     l2[i] = l2[i*4] #old training keeps 10 not 40 trials as evaluation cycle...
#
# length = 396
# avg = (l1[0:length] + l2[0: length])/2
# np.save('publish_results/A_Saturn_Section/No Advice/episode_rewards', avg)


# avg saturn no advice
path = 'publish_results/Simple_4_Room/No Advice/'


l = []
for i in range(6):
    l.append(np.load(path + 'multiple/' + str(i) + '_episode_rewards.npy'))

length = np.max([len(li) for li in l])
avg = np.mean([li[0:length] for li in l], axis=0)
np.save('publish_results/Simple_4_Room/No Advice/episode_rewards', avg)

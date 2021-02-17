import pickle
import matplotlib.pyplot as plt
import numpy as np


with open ('rewards.txt', 'rb') as fp:
    rewards = pickle.load(fp)

with open ('scores.txt', 'rb') as fp:
    scores = pickle.load(fp)

maximum_score = []
average_score = []

maximum_reward = []
average_reward = []

x = range(0, 32500, 2500)

for i in range(13):
    maximum_score.append(max(scores[i*10: (i+1)*10]))
    average_score.append(np.mean(scores[i * 10: (i + 1) * 10]))

    maximum_reward.append(max(rewards[i * 10: (i + 1) * 10]))
    average_reward.append(np.mean(rewards[i * 10: (i + 1) * 10]))

fig = plt.figure()
plt.plot(x, maximum_score, marker='x', label='Maximum Score')
plt.plot(x, average_score, marker='s', label='Average Score')

plt.xlabel('Episode', fontsize=16)
plt.ylabel('Score', fontsize=16)

plt.legend()
plt.savefig('test_scores.jpg')

fig = plt.figure()
plt.plot(x, maximum_reward, marker='x', label='Maximum Reward')
plt.plot(x, average_reward, marker='s', label='Average Reward')

plt.xlabel('Episode', fontsize=16)
plt.ylabel('Reward', fontsize=16)

plt.legend()
plt.savefig('test_rewards.jpg')



with open ('avg.txt', 'rb') as fp:
    average_per_episode = pickle.load(fp)

with open ('score.txt', 'rb') as fp:
    score_per_episode = pickle.load(fp)

with open ('sequence.txt', 'rb') as fp:
    seq_length = pickle.load(fp)

# plt.plot(average_per_episode)

a = int(len(score_per_episode) / 100)

b = []
c = []
d = []

rewards = [a * b for a, b in zip(average_per_episode, seq_length)]


for i in range(a):
    b.append(np.mean(score_per_episode[int(100*i): int(101*i + 100)]))

fig = plt.figure()
plt.plot(b)
plt.title('Average Score per Epoch', fontsize=18)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Score', fontsize=16)
plt.savefig('score.jpg')

for i in range(a):
    c.append(np.mean(rewards[int(100*i): int(100*i + 100)]))

fig = plt.figure()
plt.plot(c)
plt.title('Average Reward per Epoch', fontsize=18)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Reward', fontsize=16)
plt.savefig('reward.jpg')


for i in range(a):
    d.append(np.mean(seq_length[int(100*i): int(100*i + 100)]))

fig = plt.figure()
plt.plot(d)
plt.title('Average Number of Actions per Epoch', fontsize=18)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Number of Actions', fontsize=16)
plt.savefig('actions.jpg')

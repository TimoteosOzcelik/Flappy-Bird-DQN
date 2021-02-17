import math
import random
import pygame
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import namedtuple
from itertools import count
from PIL import Image
from pygame.surfarray import array3d, pixels_alpha
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class Environment():
    pygame.init()

    width, height = (288, 512)
    screen = pygame.display.set_mode((width, height))

    clock = pygame.time.Clock()
    fps = 30

    # Backgroun
    background = pygame.image.load("assets/sprites/background-day.png").convert()

    # Floor
    floor = pygame.image.load("assets/sprites/base.png").convert()
    floor_limit = background.get_width() - floor.get_width()

    # Bird
    bird_downflap = pygame.image.load("assets/sprites/yellowbird-downflap.png").convert_alpha()
    bird_midflap = pygame.image.load("assets/sprites/yellowbird-midflap.png").convert_alpha()
    bird_upflap = pygame.image.load("assets/sprites/yellowbird-upflap.png").convert_alpha()

    bird_index = 0
    bird_frames = [bird_downflap, bird_midflap, bird_upflap]
    bird = bird_frames[bird_index]

    bird_hitmask = [pixels_alpha(image).astype(bool) for image in bird_frames]

    init_pos = (
        int(width * 0.2),
        int(height / 2)
    )

    # Pipe
    pipe_surface = pygame.image.load("assets/sprites/pipe-green.png").convert_alpha()
    pipes = []

    '''
    pipe_gap = 100
    pipe_min = int(pipe_gap / 4)
    pipe_max = int(height * 0.79 * 0.6 - pipe_gap / 2)
    '''

    pipe_gap = 125
    pipe_min = 1
    pipe_max = 5

    def __init__(self):
        self.floorx, self.floory = (0, self.background.get_height() - self.floor.get_height())
        # self.floorx, self.floory = (0, self.height * 0.79)
        self.bird_rect = self.bird.get_rect(center=self.init_pos)

        # Game variables
        self.GRAVITY = 1
        self.FLAP_POWER = 9
        self.MAX_DROP_SPEED = 15

        # Velocity on y and x
        self.vel = 0
        self.speed = 4

        # Score
        self.score = 0

        self.tick = 0

        self.pipes.extend(self._generate_pipes(offset=(0.5 * self.width)))
        self.pipes.extend(self._generate_pipes(offset=int(0.5 * self.pipe_surface.get_width() + self.width)))


    def _generate_pipes(self, offset=0):
        # gap_start = random.randint(self.pipe_min, self.pipe_max)
        gap_start = random.randint(self.pipe_min, self.pipe_max + 1) * 25 + 50

        top_bottom = gap_start - self.pipe_surface.get_height()
        bottom_top = gap_start + self.pipe_gap

        top_pipe = self.pipe_surface.get_rect(topleft=(self.width + offset, top_bottom))
        bottom_pipe = self.pipe_surface.get_rect(topleft=(self.width + offset, bottom_top))

        return top_pipe, bottom_pipe


    def _is_collided(self):
        # out-of-screen
        if self.bird_rect.top < - self.bird.get_height() * 0.1 or self.bird_rect.bottom >= self.floory:
            return True

        # mask = self.bird_hitmask[self.bird_index]
        mask = pixels_alpha(self.rotate_bird()).astype(bool)
        rows, columns = mask.shape

        # pipe collison
        for pipe in self.pipes:
            lx, rx = pipe.x, pipe.x + self.pipe_surface.get_width()
            ty, by = pipe.y, pipe.y + self.pipe_surface.get_height()

            for i in range(rows):
                for j in range(columns):
                    # posx, posy = self.bird_rect.x + j, self.bird_rect.y + i
                    posx, posy = self.bird_rect.x + i, self.bird_rect.y + j
                    if mask[i, j] and lx < posx < rx and ty < posy < by:
                        return True

            '''      
            if self.bird_rect.colliderect(pipe):
                return True
            '''

        return False


    def rotate_bird(self):
        return pygame.transform.rotozoom(self.bird, -self.vel * 3, 1)


    def bird_animation(self):
        new_bird = self.bird_frames[self.bird_index]
        new_bird_rect = new_bird.get_rect(center=(100, self.bird_rect.centery))
        return new_bird, new_bird_rect


    def step(self, action):
        pygame.event.pump()

        # reward to stay alive
        reward = 0.1

        # terminal
        terminal = False

        self.tick += 1

        # Velocity updating
        if self.vel < self.MAX_DROP_SPEED:
            self.vel += self.GRAVITY

        if action == 1:
            self.vel = 0
            self.vel -= self.FLAP_POWER


        # Check whether bird passes the pipe or not
        for pipe in self.pipes:
            if pipe.centerx < self.bird_rect.centerx <= pipe.centerx + self.speed:
                reward = 1
                self.score += 1
                break

        # bird movement
        self.bird_rect.centery += self.vel

        # floor movement
        self.floorx -= 1
        if self.floorx < self.floor_limit:
            self.floorx = 0

        # pipes' movement
        for pipe in self.pipes:
            pipe.centerx -= self.speed

        # Update pipes
        if self.pipes[0].x <= -self.pipe_surface.get_width():
            self.pipes.extend(self._generate_pipes())

        # delete top and bottom pipes
        if self.pipes[0].x <= -self.pipe_surface.get_width():
            del self.pipes[0]
            del self.pipes[0]

        if (self.tick + 1) % 15 == 0:
            self.bird_index = (self.bird_index + 1) % 3
            self.bird, self.bird_rect = self.bird_animation()

        if self._is_collided():
            reward = -1
            terminal = True

        # draw
        self.screen.blit(self.background, (0, 0))

        for i, pipe in enumerate(self.pipes):
            if i % 2 == 0:
                flip_pipe = pygame.transform.flip(self.pipe_surface, False, True)
                self.screen.blit(flip_pipe, pipe)
            else:
                self.screen.blit(self.pipe_surface, pipe)

        self.screen.blit(self.floor, (self.floorx, self.floory))

        rotated_bird = self.rotate_bird()
        self.screen.blit(rotated_bird, self.bird_rect)

        pygame.display.update()
        screen = pygame.surfarray.array3d(pygame.display.get_surface())
        self.clock.tick(self.fps)

        return screen, reward, terminal


    def get_screen(self):
        return pygame.surfarray.array3d(pygame.display.get_surface())


    def reset(self):
        self.pipes.clear()
        self.__init__()
        
    
    def quit(self):
        pygame.quit()


# create env
env = Environment()

# if gpu is to be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        # 84x84x4
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(64)
        # 20x20x32
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        # 9x9x32
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        # 7x7x16
        self.fc1 = nn.Linear(7 * 7 * 32, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)

resize = T.Compose([T.ToPILImage(),
                    T.Grayscale(),
                    T.CenterCrop((288, 288)),
                    T.Resize(84),
                    T.ToTensor()])

n_actions = 2

# create networks
policy_net = DQN(n_actions)

model_scores = []
model_rewards = []

for i in range(13):

    policy_net.load_state_dict(torch.load('models/state_dict_model_' + str(int(2500 * i)) + '.pt', map_location=torch.device('cpu')))

    init_screen = env.get_screen()

    init_screen = resize(init_screen)
    #  (N, C, H, W): (1, 4, 84, 84)
    state = torch.cat(tuple(init_screen for _ in range(4))).unsqueeze(0).to(device)
    rewards = 0

    j = 0

    while True and j < 10:
        action = policy_net(state).max(1)[1].view(1, 1)
        screen, reward, terminal = env.step(action.item())
        rewards += reward

        screen = resize(screen).to(device)
        state = torch.cat((state[0, 1:], screen)).unsqueeze(0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if terminal == True:
            # print('Rewards: {}, Score: {}'.format(rewards, env.score))
            print('Episode: {}'.format(str(int(2500 * i))))

            model_rewards.append(rewards)
            model_scores.append(env.score)
            j += 1

            env.reset()
            init_screen = env.get_screen()
            rewards = 0
            init_screen = resize(init_screen)
            #  (N, C, H, W): (1, 4, 84, 84)
            state = torch.cat(tuple(init_screen for _ in range(4))).unsqueeze(0).to(device)


'''
with open('scores.txt', 'wb') as fp:
    pickle.dump(model_scores, fp)

with open('rewards.txt', 'wb') as fp:
    pickle.dump(model_rewards, fp)
'''
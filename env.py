import math
import random
import pygame
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image
from pygame.surfarray import array3d, pixels_alpha

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
    bird_downflap = pygame.image.load("assets/sprites/bluebird-downflap.png").convert_alpha()
    bird_midflap = pygame.image.load("assets/sprites/bluebird-midflap.png").convert_alpha()
    bird_upflap = pygame.image.load("assets/sprites/bluebird-upflap.png").convert_alpha()

    bird_index = 0
    bird_frames = [bird_downflap, bird_midflap, bird_upflap]
    bird = bird_frames[bird_index]

    bird_hitmask = [pixels_alpha(image).astype(bool) for image in bird_frames]

    # print(bird_hitmask[bird_index])

    init_pos = (
        int(width * 0.2),
        int(height / 2)
    )

    # Pipe
    pipe_surface = pygame.image.load("assets/sprites/pipe-green.png").convert_alpha()
    pipes = []

    pipe_gap = 100
    pipe_min = int(pipe_gap / 4)
    pipe_max = int(height * 0.79 * 0.6 - pipe_gap / 2)

    def __init__(self):
        self.floorx, self.floory = (0, self.background.get_height() - self.floor.get_height())
        self.bird_rect = self.bird.get_rect(center=self.init_pos)

        # Game variables
        self.GRAVITY = 1
        self.FLAP_POWER = 9
        self.MAX_DROP_SPEED = 10

        # Velocity on y and x
        self.vel = 0
        self.speed = 4

        # Score
        self.score = 0

        self.tick = 0

        self.pipes.extend(self._generate_pipes(self.width))
        self.pipes.extend(self._generate_pipes(offset=int(self.pipe_surface.get_width()/2 + 1.5 * self.width)))
        self.pipes.extend(self._generate_pipes(offset=int(self.pipe_surface.get_width() + 2 * self.width)))

        self.next_pipe = 0


    def _generate_pipes(self, offset=0):
        gap_start = random.randint(self.pipe_min, self.pipe_max)

        top_bottom = gap_start - self.pipe_surface.get_height()
        bottom_top = gap_start + self.pipe_gap

        top_pipe = self.pipe_surface.get_rect(topleft=(self.width + offset, top_bottom))
        bottom_pipe = self.pipe_surface.get_rect(topleft=(self.width + offset, bottom_top))

        return top_pipe, bottom_pipe


    def _is_collided(self):
        # out-of-screen
        if self.bird_rect.top < - self.bird.get_height() * 0.1 or self.bird_rect.bottom >= self.floory:
            return True

        mask = self.bird_hitmask[self.bird_index]
        rows, columns = mask.shape

        # pipe collison
        for pipe in self.pipes:
            lx, rx = pipe.x, pipe.x + self.pipe_surface.get_width()
            ty, by = pipe.y, pipe.y + self.pipe_surface.get_height()

            for i in range(rows):
                for j in range(columns):
                    posx, posy = self.bird_rect.x + j, self.bird_rect.y + i
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

        # update tick
        self.tick += 1

        # Velocity updating
        if self.vel < self.MAX_DROP_SPEED:
            self.vel += self.GRAVITY

        if action == 1:
            self.vel = 0
            self.vel -= self.FLAP_POWER

        # bird movement
        self.bird_rect.centery += self.vel

        # floor movement
        self.floorx -= 1
        if self.floorx < self.floor_limit:
            self.floorx = 0

        # pipes' movement
        for pipe in self.pipes:
            pipe.centerx -= self.speed

        # Check whether bird passes the pipe or not
        for pipe in self.pipes:
            if pipe.centerx < self.bird_rect.centerx <= pipe.centerx + self.speed:
                reward = 1
                self.score += 1
                self.next_pipe += 2
                break

        # Update pipes
        if self.pipes[0].x <= -self.pipe_surface.get_width():
            self.pipes.extend(self._generate_pipes(offset=int(0.5 * self.pipe_surface.get_width() + 0.5 * self.width)))

        # delete top and bottom pipes
        if self.pipes[0].x <= -self.pipe_surface.get_width():
            del self.pipes[0]
            del self.pipes[0]

            self.next_pipe -= 2

        # collision control
        if self._is_collided():
            reward = -1
            terminal = True

        if (self.tick + 1) % 15 == 0:
            self.bird_index = (self.bird_index + 1) % 3
            self.bird, self.bird_rect = self.bird_animation()  # TODO: Check functions

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

        # self.screen.blit(self.bird, self.bird_rect)

        pygame.display.update()
        # screen = pygame.surfarray.array3d(pygame.display.get_surface())
        screen = [
            self.bird_rect.centery,
            self.vel,
            self.pipes[self.next_pipe].centerx - self.bird_rect.centerx,
            self.pipes[self.next_pipe].y,
            self.pipes[self.next_pipe + 1].y,

            self.pipes[self.next_pipe + 2].centerx - self.bird_rect.centerx,
            self.pipes[self.next_pipe + 2].y,
            self.pipes[self.next_pipe + 3].y
        ]

        """
        Gets a non-visual state representation of the game.

        Returns
        -------

        dict
            * player y position.
            * players velocity.
            * next pipe distance to player
            * next pipe top y position
            * next pipe bottom y position
            * next next pipe distance to player
            * next next pipe top y position
            * next next pipe bottom y position


            See code for structure.

        """

        self.clock.tick(self.fps)

        return screen, reward, terminal


    # TODO : State of the game should be returned
    def get_screen(self):
        return array3d(pygame.display.get_surface())


    def reset(self):
        self.pipes.clear()
        self.__init__()


    def quit(self):
        pygame.quit()


def main():
    env = Environment()

    while True:
        action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1

        screen, reward, terminal = env.step(action)
        # print(reward)

        if terminal == True:
            env.reset()


if __name__ == '__main__':
    main()
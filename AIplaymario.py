## A Project for using Artificial Intelligence to play mario by Li Ka Lok, Jack

## Import the required library
## SIMPLE_MOVEMENT contain 7 actions in mario game(None, left, right, right+a+b, right+a, right+b, a)
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from numba import njit, jit
import numpy as np
import sys
import time
# Import Frame Stacker wrapper and GrayScaling Wrapper
from gym.wrappers import FrameStack, GrayScaleObservation # capture Frames, gray color for less data to process
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv # vectorize frames, 
import matplotlib.pyplot as plt # to show fame stacking
import torch as th 
## setup game
env = gym_super_mario_bros.make('SuperMarioBros-v1')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

## Create a flag (whether restart or not)
done = True# Loop through each frame in the game
for step in range(100000):
    '''Start the game to begin with'''
    if done:
        #Start the game
        env.reset()
    #Do random actions
    state, reward, done, info = env.step(env.action_space.sample())
    #Show the game
    env.render()
# Close the game
env.close()

## Preprocess Environment

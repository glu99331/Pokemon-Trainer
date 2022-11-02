import tensorflow as tf
from tensorflow import keras
from rl import callbacks

#action = 0

class RewardCallback(callbacks.Callback):
    moves = []
    opponent_types = ()

    def __init__(self):
        #Step Logs
        self.status_uses = 0
        self.super_effective_uses = 0
        self.step_actions = []
        self.step_rewards = []

        #Episode Logs
        self.episode_rewards = []

    def on_step_end(self, step, logs={}):
        #action_choice = logs['action']
        #print("STEP:",step+1)
        #print("ACTION:",action_choice)
        #print("MOVES:",self.moves)
        #if(len(self.moves)!=0):
        #    if(action_choice <= 15):
        #        try:
        #            print("CHOSEN MOVE:",self.moves[(action_choice%4)])
        #            print("MULTIPLIER:",self.moves[(action_choice%4)].type.damage_multiplier(opponent_types[0],opponent_types[1]))
        #            #print(self.moves[(action_choice%4)].status)
        #            #print(self.moves[(action_choice%4)].type.damage_multiplier(opponent_types[0],opponent_types[1]))
        #            if(self.moves[(action_choice%4)].status):
        #                self.status_uses+=1
        #                #print("STATUS MOVE USED")
        #            if(self.moves[(action_choice%4)].type.damage_multiplier(opponent_types[0],opponent_types[1]) > 1):
        #                self.super_effective_uses+=1
        #                #print("SUPER EFFECTIVE")
        #        except:
        #            print("move calculation error")
        #    else:
        #        print("no move used")
        self.step_rewards.append(logs['reward'])

    def on_episode_end(self, episode, logs={}):
        self.episode_rewards.append(logs['episode_reward'])

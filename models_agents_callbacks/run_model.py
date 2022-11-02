#Poke-Env Imports
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration

#Keras, Tensorflow Imports
import numpy as np
import tensorflow as tf
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

#Other Imports
import matplotlib.pyplot as plt
import time
import sys
import os

#Local Imports
from max_agent import MaxDamagePlayer
from QLearningPlayer import QLearningPlayer
from callbacks import RewardCallback

#Methods
def createTrainers(agents):
    q_player = QLearningPlayer(battle_format="gen8randombattle")
    players = [q_player]
    for agent in agents:
        if(agent == "random"):
            random_player = RandomPlayer(battle_format="gen8randombattle")
            players.append(random_player)
            continue
        if(agent == "max_damage"):
            max_player = MaxDamagePlayer(battle_format="gen8randombattle")
            players.append(max_player)
            continue
        else:
            raise Exception("player not found")
    return players

def buildArch(q_player, num_layers, m, actvation_function):
    n_action = len(q_player.action_space)
    model = Sequential()
    model.add(Dense(m, activation=activation_function, input_shape=q_player.state_shape)) #input layer
    model.add(Flatten())
    for i in range(num_layers):
        model.add(Dense(m, activation=activation_function))
    model.add(Dense(n_action, activation="softmax"))
    return model

def defineDQN_and_Policy(model, q_player, training_steps, learning_rate):
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=training_steps,
    )
    # Defining our DQN
    dqn = DQNAgent(
        model=model,
        nb_actions=len(q_player.action_space),
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    dqn.compile(Adam(lr=learning_rate), metrics=["mae"])
    return dqn

def training(dqn, q_player, opponent, training_steps, experiment_name, saveModel=False):
    def dqn_training(player, dqn, nb_steps, rewards_callback):
        dqn.fit(player, nb_steps=nb_steps, callbacks=rewards_callback)
        player.complete_current_battle()

    start = time.time()
    rewards_callback = RewardCallback()
    q_player.play_against(
        env_algorithm=dqn_training,
        opponent=opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_steps": training_steps,  "rewards_callback": [rewards_callback]},
    )
    if saveModel:
        model.save(f"saved_models/model_{experiment_name}_{training_steps}")
    end = time.time()
    elapsed = (end-start)/60
    return rewards_callback.episode_rewards, elapsed

def testing(dqn, q_player, opponents, testing_battles):
    win_percentages = []
    def dqn_evaluation(player, dqn, nb_episodes):
        player.reset_battles()
        dqn.test(player, nb_episodes=testing_battles, visualize=False, verbose=False)
        print(f"Model Evaluation: {player.n_won_battles} victories out of {testing_battles} episodes")
        win_percentage = player.n_won_battles/nb_episodes #This might be one off because of flooring and stuff
        win_percentages.append(win_percentage)

    for opponent in opponents:
        print(f"Results against {opponent}:")
        q_player.play_against(
            env_algorithm=dqn_evaluation,
            opponent=opponent,
            env_algorithm_kwargs={"dqn": dqn, "nb_episodes": testing_battles},
        )
    return win_percentages

def generateReport(experiment_name, training_steps, ep_rewards, win_percentages, opponents, elapsed_time):
    os.makedirs(f'../outputs/{experiment_name}_{training_steps}')

    plt.plot(ep_rewards)
    plt.title("Reward Change Every Episode During Training")
    plt.xlabel("Training Episode")
    plt.ylabel("Reward")
    plt.savefig(f"../outputs/{experiment_name}_{training_steps}/ep_reward.png")

    f = open(f'../outputs/{experiment_name}_{training_steps}/win_and_time.txt', 'w')
    f.write(f"{experiment_name} Results:\n")
    for i in range(len(opponents)):
        f.write(f"Win% Against {opponents[i]}: {win_percentages[i]}\n")
    f.write(f"Total Training Time: {int(elapsed_time)} minutes")
    f.close()

#Define Hyperparameters

#CMD Args
experiment_name = "lr-exp"
training_steps = 50000
test_episodes = 10
players = ['random', 'max_damage']#sys.argv[4].split(",")
saveModel = False
#if(sys.argv[5] in ['true', 'True']):
#    saveModel = True
learning_rate = [0.001, 0.0005, 0.0001, 0.00005, 0.00001] #0.00005 is optimal
num_layers = 1
m = 256
activation_function = 'relu'

#Other Args
tf.random.set_seed(0)
np.random.seed(0)
memory = SequentialMemory(training_steps, window_length=1)

#Note: You need to delete player configs before you grid search

#Main
for lr in learning_rate:
    print("TRAINING WITH LEARNING RATE OF:", lr)
    players = createTrainers(["random", "max_damage"]) #this supports 'random' and 'max_damage' right now
    arch = buildArch(players[0], num_layers, m, activation_function)
    dqn = defineDQN_and_Policy(arch, players[0], training_steps, lr)
    ep_rewards, training_time = training(dqn, players[0], players[1], training_steps, experiment_name, saveModel)
    win_percentages = testing(dqn, players[0], players[1:], test_episodes)
    generateReport(experiment_name+f'_{lr}', training_steps, ep_rewards, win_percentages, players[1:], training_time)

#Poke-Env imports
from poke_env.environment.pokemon_type import PokemonType
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer

#Keras, Tensorflow Imports
import numpy as np
import tensorflow as tf
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras import layers, models, activations
from tensorflow.keras.optimizers import Adam

#Local Imports
from max_agent import MaxDamagePlayer
from callbacks import RewardCallback

# Define some battle constants
TYPES_VECTOR_SIZE = 18
WEATHER_VECTOR_SIZE = 7

# Convert type(s) to one-hot (or two-hot) vector
def enum_to_vector(val1, val2=None, size=0):
    if size == 0:
        raise ValueError("size must be set on enum_to_vector")
    vec = np.zeros(size)
    # Weather case
    if type(val1) is dict:
        for key in val1.keys():
            vec[key.value-1] = 1
        return vec
    # Type case
    if val1:
        vec[val1.value-1] = 1
    if val2:
        vec[val2.value-1] = 1
    return vec

# Convert move to mapping vector
def move_to_mapping(move):
    if move is None:
        return np.zeros(2+TYPES_VECTOR_SIZE)
    mapping = [move.base_power, move.accuracy]
    mapping.extend(enum_to_vector(move.type, size=TYPES_VECTOR_SIZE))
    return mapping

# Convert pokemon to mapping vector
def pokemon_to_mapping(pokemon, moves=None):
    mapping = [pokemon.current_hp_fraction, pokemon.stats["atk"], pokemon.stats["def"], pokemon.stats["spa"], pokemon.stats["spd"], pokemon.stats["spe"]]
    mapping.extend(enum_to_vector(pokemon.type_1, pokemon.type_2, size=TYPES_VECTOR_SIZE))
    movelist = list(pokemon.moves.values()) if moves is None else list(moves)
    for i in range(4):
        mapping.extend(move_to_mapping(movelist[i] if i<len(movelist) else None))
    return mapping

def opponent_pokemon_to_mapping(pokemon):
    if pokemon is None:
        mapping = [1]
        mapping.extend(enum_to_vector(None, size=TYPES_VECTOR_SIZE))
        return mapping
    mapping = [pokemon.current_hp_fraction]
    mapping.extend(enum_to_vector(pokemon.type_1, pokemon.type_2, size=TYPES_VECTOR_SIZE))
    return mapping

# Define player for model input and reward calculation
class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def embed_battle(self, battle):
        # Add player team to mapping
        player_team_mapping = pokemon_to_mapping(battle.active_pokemon, moves=battle.available_moves)
        for player_pokemon in battle.team.values():
            if player_pokemon != battle.active_pokemon:
                player_team_mapping.extend(pokemon_to_mapping(player_pokemon))
        # Add opponent team to mapping
        # Opponent team needs empty mappings for unrevealed pokemon
        opponent_team = battle.opponent_team.copy()
        while len(opponent_team) < 6:
            opponent_team["empty{}".format(len(opponent_team))] = None
        opponent_team_mapping = opponent_pokemon_to_mapping(battle.opponent_active_pokemon)
        for opponent_pokemon in opponent_team.values():
            if opponent_pokemon != battle.opponent_active_pokemon:
                opponent_team_mapping.extend(opponent_pokemon_to_mapping(opponent_pokemon))
        # Add weather and other battle stats to mapping
        battle_stats = []
        battle_stats.extend(enum_to_vector(battle.weather if battle.weather else None, size=WEATHER_VECTOR_SIZE))

        return np.concatenate((player_team_mapping, opponent_team_mapping, battle_stats))

    def compute_reward(self, battle):
        return self.reward_computing_helper(battle, hp_value=.2, status_value=.2, victory_value=1)

class MaxDamagePlayer(RandomPlayer):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


NB_TRAINING_STEPS = 10000
NB_EVALUATION_EPISODES = 100

tf.random.set_seed(0)
np.random.seed(0)


# This is the function that will be used to train the dqn
def dqn_training(player, dqn, nb_steps, callback, rewards):
    dqn.fit(player, nb_steps=nb_steps, callbacks=callback)
    player.complete_current_battle()
    rewards.append(callback.rewards)
    return rewards

def dqn_evaluation(player, dqn, nb_episodes):
    # Reset battle statistics
    player.reset_battles()
    dqn.test(player, nb_episodes=nb_episodes, visualize=False, verbose=False)

    print(
        "DQN Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, nb_episodes)
    )


if __name__ == "__main__":
    env_player = SimpleRLPlayer(battle_format="gen8randombattle")

    opponent = RandomPlayer(battle_format="gen8randombattle")
    second_opponent = MaxDamagePlayer(battle_format="gen8randombattle")

    # Output dimension
    n_action = len(env_player.action_space)

    model = models.Sequential()
    model.add(layers.Dense(1024, activation=activations.relu, input_shape=(1,745,)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation=activations.relu))
    model.add(layers.Dense(512, activation=activations.relu))
    model.add(layers.Dense(n_action, activation=activations.softmax))

    memory = SequentialMemory(limit=10000, window_length=1)

    # Ssimple epsilon greedy
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=NB_TRAINING_STEPS,
    )

    # Defining our DQN
    dqn = DQNAgent(
        model=model,
        nb_actions=n_action,
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01
    )

    dqn.compile(Adam(lr=0.00025), metrics=["mse"])

    # Training
    rewards = []
    env_player.play_against(
        env_algorithm=dqn_training,
        opponent=opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_steps": NB_TRAINING_STEPS, "callback": rewards_callback, "rewards": rewards},
    )
    model.save("model_%d" % NB_TRAINING_STEPS)

    # Evaluation
    print("Results against random player:")
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    )

    print("\nResults against max player:")
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=second_opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    )

    #Analysis
    import matplotlib.pyplot as plt

    plt.plot(rewards)
    plt.title("Reward Over Time During Training")
    plt.xlabel("Training Step")
    plt.ylabel("Reward")
    plt.save("reward.png")
    plt.show()

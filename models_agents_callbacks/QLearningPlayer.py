#Poke-Env Imports
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer

#Keras, Tensorflow Imports
import numpy as np
import tensorflow as tf
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

class QLearningPlayer(Gen8EnvSinglePlayer):
    state_shape = (1,20)

    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_accuracy = -np.ones(4)
        moves_heal = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (move.base_power / 100)  # Simple rescaling to facilitate learning
            moves_accuracy[i] = move.accuracy
            moves_heal[i] = move.heal
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (len([mon for mon in battle.team.values() if mon.fainted]) / 6)
        remaining_mon_opponent = (len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6)

        dmg_mults = -np.ones(6)
        for i, pokemon in enumerate(battle.team.values()):
            for move in pokemon.moves.values():
                dmg_mults[i] = max(dmg_mults[i], move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                ))

        # Final vector with 10 components
        state = np.concatenate(
            [
                [battle.active_pokemon.current_hp_fraction],
                moves_base_power,
                #moves_accuracy,
                moves_heal,
                moves_dmg_multiplier,
                dmg_mults,
                #[remaining_mon_team, remaining_mon_opponent],
                [1 if battle.trapped else 0],
            ]
        )
        self.state_shape = state.shape
        return state

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(battle, fainted_value=2, hp_value=1, victory_value=30)

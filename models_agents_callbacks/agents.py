from poke_env.player.player import Player

class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)

class StatusPlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            for move in battle.available_moves:
                if move.status != None and battle.opponent_active_pokemon.status != None:
                    return self.create_order(move) #return status move if you can
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move) #else return max damage
        else:
            return self.choose_random_move(battle) #else return random

class SuperEffectivePlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power * battle.opponent_active_pokemon.damage_multiplier(move))
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)

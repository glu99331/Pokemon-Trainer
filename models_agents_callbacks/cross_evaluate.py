import asyncio
import time
from tabulate import tabulate

from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from poke_env.player_configuration import PlayerConfiguration
from poke_env.player.utils import cross_evaluate

from agents import MaxDamagePlayer, StatusPlayer, SuperEffectivePlayer

async def main():
    #Create Players
    Randy = RandomPlayer(player_configuration=PlayerConfiguration("Randy",None),battle_format="gen8randombattle")
    Max = MaxDamagePlayer(player_configuration=PlayerConfiguration("Max",None),battle_format="gen8randombattle")
    Stacy = StatusPlayer(player_configuration=PlayerConfiguration("Stacy",None),battle_format="gen8randombattle")
    Susy = SuperEffectivePlayer(player_configuration=PlayerConfiguration("Susy",None),battle_format="gen8randombattle")
    players = [Randy, Max, Stacy, Susy]

    #Run Matches
    start = time.time()
    cross_evaluation = await cross_evaluate(players, n_challenges=20)
    end = time.time()
    print("matches completed in: ", end-start, "seconds")

    #Tabulate Results
    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])
    print(tabulate(table))

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())

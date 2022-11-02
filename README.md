# The Very Best Deep Learning Bot
*By: Kenan Rustamov, Gordon Lu, Jake Baumbaugh, and Sean Steinle*

### What Is This?

We're glad you asked. We're a group of four students enrolled in CS1678 (Introduction to Deep Learning) at the University of Pittsburgh, and this is our 
term project for the class. For our project, we decided we wanted to play a game with Deep Learning. The game we settled on was Pokemon, and thus the project was born.
Our goal in this project is to create the best Pokemon trainer possible in a 6v6 Pokemon battle.

### Assumptions and Scenario

For our project, we assume that both our trainer and the opposing trainer both have six Pokemon. Additionally, the pokemon are randomly generated for each team, between levels of 70-90.
All Pokemon are from the Pokemon generation VIII. The opposing trainer that we are training or testing on is almost always a rule-based agent that we have written, but we are hoping to test
on real players on Pokemon-Showdown.

### How to Run

To run our project, you'll need to do three things: get a Pokemon-Showdown server running, set up an environment with the appropriate dependencies, and you'll need to clone our code. You can optionally try to run our code on a GPU.

1. Getting Pokemon-Showdown Running
	1. First, you'll need to download Node.js.
	2. Then, you'll need to clone Pokemon-Showdown's repository.
	3. Finally, go into the Pokemon-Showdown repository in your cmd, and run the server with *node pokemon-showdown start --no-security*
		1. Note that we run with the *no security* flag because we're training locally.
2. Setting Up Our Environment
	1. First, make a Python virtual environment with *python -m venv <env_name>*
		1. You can activate the environment with *<env_name>/Scripts/activate*
		2. You can deactivate the environment with *deactivate*
		3. We ran with Python versions 3.7 and 3.8, but we don't expect problems with more recent versions.
	2. Then, install our dependencies from the *requirements.txt* file with *pip install -r requirements.txt*
	3. Finally, run the agent using *python rl_agent.py*
3. Cloning Our Code
	1. Finally, clone our repository with *git clone <gh_link>*
		1. To get the link, click the green button that says "Code" and copy that link.

### Repo Contents

1. ***models_agents_callbacks*** - This is our largest folder, and like its name suggests, it contains
our models, agents, and callbacks. They are all grouped together because these files depend on
each other, and Python makes it difficult to import outside of the same directory.
	1. Models - Models are reinforcement learning models that we have trained on an agent.
		1. *default_model.py* - This is model was entirely written by Haris Sahovic, and can be found here https://poke-env.readthedocs.io/en/stable/rl_with_open_ai_gym_wrapper.html. This is an important baseline model for us.
		2. *OUR MODEL GOES HERE* - This is 
	2. Agents - Agents are rule-based pokemon trainers that we write for the purpose of training our models on.
		1. *random_agents.py* - Random agents are actually a part of the Poke-Env source code, but this file demonstrates how to instantiate them. Based on the tutorial from PokeEnv here: https://poke-env.readthedocs.io/en/stable/cross_evaluate_random_players.html.
		2. *max_agent* - This is our implementation of the max agent, based on the Poke-Env tutorial here: readthedocs.
		3. *OTHER AGENTS* - 
	3. Callbacks - Callbacks are classes that we write to extract information from our models during training.
		1. *callbacks.py* - All of our callbacks are in this file.
2. ***outputs*** - This folder contains our outputs and reports from our models and agents competing.
3. ***saved_models*** - This folder contains saved versions of our models.
4. ***background*** - This folder contains our reports throughout the semester, as well as our final presentation slides.

### Acknowledgements

We'd like to thank our instructors for the course, Dr. Kovashka and Ahmad Diab. We'd also like to thank the amazing folks at the Poke-Env library as well as those at the Pokemon-Showdown site who made
our project feasible. 

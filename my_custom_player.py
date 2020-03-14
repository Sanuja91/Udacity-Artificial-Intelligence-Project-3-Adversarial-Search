import random
import numpy as np
from sample_players import DataPlayer
from isolation import isolation

class Node:
	def __init__(self, state : isolation, parent = None, parent_action = None):
		# parent_action is None for root node
		self.visits = 0
		self.value = None
		self.state = state
		self.parent_action = parent_action
		self.unvisited_actions = state.actions()
		self.parent = parent
		self.children = {}
		self.player = state.player()

	def is_expandable(self):
		"""Checks if leaf node"""
		return len(self.unvisited_actions) != 0

	def is_terminal(self):
		"""Checks if terminal state"""
		return self.state.terminal_test()

	def value(self):
		"""Gets value of node"""
		if self.visits == 0:
			return 0
		return self.value / self.visits

	def expand(self):
		"""Expands existing leaf node"""
		action = random.choice(self.unvisited_actions)
		next_state = self.state.result(action)
		self.unvisited_actions.remove(action)
		node = Node(state = next_state, parent = self, parent_action = action)
		self.children[action] = node
		return node

class MCTS:
	"""
	Monte Carlo Tree Search algorithm 
	simulations = number of simulations to run
	exploration_factor = exploration priority
	expert_factory = expert knowledge priority
	expansion_threshold = after visiting a leaf node x times, 
	you expand it without using its rollout value
	"""
	def __init__(self, simulations, player_id, exploration_factor = 1.4, expert_factor = 2.0, expansion_threshold = 100, rolled_out_times = 10):
		self.simulations = simulations
		self.exploration_factor = exploration_factor
		self.expert_factor = expert_factor
		self.expansion_threshold = expansion_threshold
		self.rollout_times = rollout_times
		self.player_id = player_id

	def run(self, state : isolation):
		root = Node(state)
		root.expand()

		for _ in range(self.simulations):
			node = root
			search_path = [node]

			while not node.is_terminal():
				if node.visits >= self.expansion_threshold and node.is_expandable():
					node = node.expand()
					search_path.append(node)
				else:
					node = self.select_next_node(node)
					search_path.append(node)

	def select_next_node(self, node):
		"""Picks next child node based on highest value"""
		children = node.children
		children_values = {}
		for child_key in children.keys():
			child = children[child_key]
			# value for exploiting the child path
			exploitation_value = child.value()
			# value for exploring the child path
			exploration_value = self.exploration_factor * np.log(node.visits) / child.visits
			# value for expert knowledge
			# expert_value = self.expert_factor * expert_knowledge / (child.visits + 1)
			
			# children_values[child_key] = exploitation_value + exploration_value + expert_value
			children_values[child_key] = exploitation_value + exploration_value

		selected_child_key = max(children_values, key = children_values.get)
		return children[selected_child_key]

	def rollout(self, node):
		"""Rollouts the states randomly to find approximate value of the node"""
		# This needs to be backpropogated
		state = node.state
		action_values = []
		for action in state.actions:
			rollout_values = []
			while self.rollout_times > len(rollout_values):
				while not self.state.terminal_test():
					state = state.result(random.choice(state.actions()))
				rollout_values.append(state.utility(self.player_id))
				rolled_out_times += 1
			action_values.append(sum(rollout_values) / len (rollout_values))
		node.state

	def backpropogate(self, search_path, value):
		for node in reversed(search_path):
			node.value += value
			node.visits += 1


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            self.queue.put(self.mcts(state, 3))

	def alpha_beta_search(self, state, depth):    

        def min_value(state, depth, alpha, beta):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0 : return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1, alpha, beta))
                if value <= alpha: return value
                beta = min(beta, value)
            return value 

        def max_value(state, depth, alpha, beta):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0 : return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1, alpha, beta))
                if value >= beta: return value
                alpha = max(alpha, value)
            return value
        
        return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1, float('-inf'), float('inf')))     

	def mcts(self, state)
		
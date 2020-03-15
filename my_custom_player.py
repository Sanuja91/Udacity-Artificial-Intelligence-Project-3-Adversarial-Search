import random, sys
from time import time
import numpy as np
from copy import copy, deepcopy
from sample_players import DataPlayer
from isolation import isolation


class Node:
    def __init__(self, state: isolation, parent=None, parent_action=None):
        # parent_action is None for root node
        self.visits = 0
        self.value = 0
        self.state = state
        self.parent_action = parent_action
        self.unvisited_actions = state.actions()
        self.parent = parent
        self.children = {}
        self.player = state.player()

    def is_expandable(self):
        """Checks if leaf node"""
        return len(self.unvisited_actions) != 0

    def has_children(self):
        """Checks if leaf node"""
        return len(self.children) > 0

    def is_terminal(self):
        """Checks if terminal state"""
        return self.state.terminal_test()

    def get_value(self):
        """Gets value of node"""
        if self.visits == 0:
            return 0
        return self.value / self.visits

    def expand(self):
        """Expands existing leaf node"""
        while len(self.unvisited_actions) > 0:
            action = random.choice(self.unvisited_actions)
            next_state = self.state.result(action)
            self.unvisited_actions.remove(action)
            node = Node(state=next_state, parent=self, parent_action=action)
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

    def __init__(self, simulations, player_id, exploration_factor=1.4, expert_factor=2.0, expansion_threshold=100, rollout_times=10, simulation_time = 4.9):
        self.simulations = simulations
        self.exploration_factor = exploration_factor
        self.expert_factor = expert_factor
        self.expansion_threshold = expansion_threshold
        self.rollout_times = rollout_times
        self.player_id = player_id
        self.simulation_time = simulation_time
        self.time = time()

    def run(self, state: isolation):
        root = Node(state)
        root.expand()

        for _ in range(self.simulations):
            node = root
            search_path = [node]

            while not node.is_terminal() and time() - self.time < self.simulation_time * 0.8:

                if node.has_children() and node.visits > 0:
                    print('Selecting Node')
                    node = self.select_next_best_node(node)
                    search_path.append(node)
                elif node.visits >= self.expansion_threshold and node.is_expandable():
                    print('Expanding Node')
                    node = node.expand()
                    search_path.append(node)
                else:
                    print('Rolling Out Node')
                    node = self.rollout(node, search_path)
                    search_path.append(node)

        best_next_node = self.select_next_best_node(root)
        print('BEST ACTION', best_next_node)
        return best_next_node.parent_action

    def rollout(self, node, search_path):
        """Rollouts the states randomly to find approximate value of the node"""
        for action in node.state.actions():
            search_path_ = copy(search_path)
            search_path_.append(node)
            rolled_out_times = 0
            while self.rollout_times > rolled_out_times:
                state = deepcopy(node.state)
                count = 0
                while not state.terminal_test():
                    count += 1
                    state = state.result(random.choice(state.actions()))
                    print('SUB LOOP COUNT', count, 'NEXT ACTIONS', len(state.actions()), time() - self.time)
                print('EXIT SUB LOOP')
                value = state.utility(self.player_id)
                if value == float('inf'):
                    value = 1
                elif value == float('-inf'):
                    value = -1
                else:
                    value = 0                    
                self.backpropogate(search_path_, value)
                rolled_out_times += 1
                print('ROLLED OUT TIMES', rolled_out_times)
        return node

    def select_next_best_node(self, node):
        """Picks next child node based on highest value"""
        children = node.children
        children_values = {}
        for child_key in children.keys():
            child = children[child_key]
            # value for exploiting the child path
            exploitation_value = child.get_value()
            # value for exploring the child path
            print('EXP FACTOR', self.exploration_factor, 'NODE VISITS', node.visits, 'LOG VISITS', np.log(node.visits), 'CHILD VISITS', child.visits)
            exploration_value = self.exploration_factor * np.log(node.visits) / child.visits
            # value for expert knowledge
            # expert_value = self.expert_factor * expert_knowledge / (child.visits + 1)

            # children_values[child_key] = exploitation_value + exploration_value + expert_value
            print("CHILD KEY", child_key, "EXPLOIT", exploitation_value, "EXPLORE", exploration_value)
            children_values[child_key] = exploitation_value + exploration_value
        selected_child_key = max(children_values, key=children_values.get)
        print('SELECTED CHILD', selected_child_key)
        return children[selected_child_key]

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
        try:
            mcts = MCTS(100, self.player_id)
            if state.ply_count < 2:
                self.queue.put(random.choice(state.actions()))
            else:
                self.queue.put(mcts.run(state))
        except:
            print("Unexpected error:"+str(sys.exc_info()[0]))

    def alpha_beta_search(self, state, depth):
        def min_value(state, depth, alpha, beta):
            if state.terminal_test():
                return state.utility(self.player_id)
            if depth <= 0:
                return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(
                    state.result(action), depth - 1, alpha, beta))
                if value <= alpha:
                    return value
                beta = min(beta, value)
            return value

        def max_value(state, depth, alpha, beta):
            if state.terminal_test():
                return state.utility(self.player_id)
            if depth <= 0:
                return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(
                    state.result(action), depth - 1, alpha, beta))
                if value >= beta:
                    return value
                alpha = max(alpha, value)
            return value
        return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1, float('-inf'), float('inf')))

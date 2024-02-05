# This file is modified variant of https://gitlab.aicrowd.com/flatland/flatland/-/tree/master/flatland/envs/observations.py
# main changes: tree obs can be evaluated along given path

"""
Collection of environment-specific ObservationBuilder.
"""
import collections
import math
from typing import Optional, List, Dict, Tuple

import numpy as np

from fl_base.fast_methods import fast_argmax, fast_count_nonzero, fast_position_equal, fast_delete, fast_where, my_fast_where

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.step_utils.states import TrainState
from flatland.utils.ordered_set import OrderedSet

Node = collections.namedtuple('Node', 'dist_own_target_encountered '
                                      'dist_other_target_encountered '
                                      'dist_other_agent_encountered '
                                      'potential_conflict_dist '
                                      'potential_conflict_pos '
                                      'potential_conflict_time_step '
                                      'dist_unusable_switch '
                                      'dist_to_next_branch '
                                      'dist_min_to_target '
                                      'num_agents_same_direction '
                                      'num_agents_opposite_direction '
                                      'num_agents_malfunctioning '
                                      'speed_min_fractional '
                                      'tot_dist '
                                      'num_agents_ready_to_depart '
                                      'node_pos '
                                      'agent_direction '
                                      'childs')


class TreeObsForRailEnv(ObservationBuilder):
    """
    TreeObsForRailEnv object.

    This object returns observation vectors for agents in the RailEnv environment.
    The information is local to each agent and exploits the graph structure of the rail
    network to simplify the representation of the state of the environment for each agent.

    For details about the features in the tree observation see the get() function.
    """

    tree_explored_actions_char = ['L', 'F', 'R', 'B']

    def __init__(self, max_depth: int, predictor: PredictionBuilder = None):
        super().__init__()
        self.max_depth = max_depth
        self.observation_dim = 11
        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.predictor = predictor
        self.location_has_target = None

        self.max_prediction_depth = 0
        self.predicted_pos = {}
        self.predicted_dir = {}

    def reset(self):
        self.location_has_target = {tuple(agent.target): 1 for agent in self.env.agents}

    # TODO now where have re-planed an agent, i need to update the predictions for obs of consequent agents
    def get_many(self, handles: Optional[List[int]] = None) -> Dict[int, Node]:
        """
        Called whenever an observation has to be computed for the `env` environment, for each agent with handle
        in the `handles` list.
        """

        if handles is None:
            handles = []

        self.init_predictions(handles)

        self.initLUT()

        # observations = super().get_many(handles)
        observations = self.get_many_super(handles)

        return observations

    def initLUT(self):
        # Update local lookup table for all agents' positions
        # ignore other agents not in the grid (only status active and done)
        # self.location_has_agent = {tuple(agent.position): 1 for agent in self.env.agents if
        #                         agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE]}
        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.location_has_agent_speed = {}
        self.location_has_agent_malfunction = {}
        self.location_has_agent_ready_to_depart = {}
        for _agent in self.env.agents:
            if not _agent.state.is_off_map_state() and \
                _agent.position:
                self.location_has_agent[tuple(_agent.position)] = 1
                self.location_has_agent_direction[tuple(_agent.position)] = _agent.direction
                self.location_has_agent_speed[tuple(_agent.position)] = _agent.speed_counter.speed
                self.location_has_agent_malfunction[tuple(_agent.position)] = \
                    _agent.malfunction_handler.malfunction_down_counter

            # [NIMISH] WHAT IS THIS
            if _agent.state.is_off_map_state() and \
                _agent.initial_position:
                self.location_has_agent_ready_to_depart.setdefault(tuple(_agent.initial_position), 0)
                self.location_has_agent_ready_to_depart[tuple(_agent.initial_position)] += 1
            # self.location_has_agent_ready_to_depart[tuple(_agent.initial_position)] = \
            #     self.location_has_agent_ready_to_depart.get(tuple(_agent.initial_position), 0) + 1

    # used
    def update_observation_prediction(self, handles: Optional[List[int]] = None):
        if handles is None:
            handles = np.arange(0, len(self.env.agents))
        self.init_predictions(handles)

    def init_predictions(self, handles: Optional[List[int]] = None):
        if self.predictor:
            self.max_prediction_depth = 0
            self.predicted_pos = {}
            self.predicted_dir = {}
            self.predictions = self.predictor.get()
            n_agents = self.env.number_of_agents

            if self.predictions:
                for t in range(self.predictor.max_depth + 1):
                    # pos_list = []

                    pos_list = -np.ones(shape=(n_agents, 2))
                    dir_list = -np.ones(shape=(n_agents, 1))
                    # np.ones(shape=(self.max_depth + 1, 4))
                    # dir_list = []

                    for a in handles:
                        if a not in self.predictions or self.predictions[a][t][1] == math.inf:
                            continue
                        # pos_list.append(self.predictions[a][t][0:2])
                        # dir_list.append(self.predictions[a][t][2])
                        pos_list[a] = self.predictions[a][t][1:3]
                        dir_list[a] = self.predictions[a][t][3]
                    # if len(pos_list)==0:
                    #     continue

                    self.predicted_pos.update({t: coordinate_to_position(self.env.width, pos_list)})
                    self.predicted_dir.update({t: dir_list})
                self.max_prediction_depth = len(self.predicted_pos)

    def get_many_super(self, handles: Optional[List[int]] = None):
        # TODO sort agents call by priority
        observations = {}
        if handles is None:
            handles = []
        for h in handles:
            observations[h] = self.get(h)
        return observations

    def get(self, handle: int = 0, init_dist: int = 1, section_list: list = None) -> Node:
        """
        Computes the current observation for agent `handle` in env

        The observation vector is composed of 4 sequential parts, corresponding to data from the up to 4 possible
        movements in a RailEnv (up to because only a subset of possible transitions are allowed in RailEnv).
        The possible movements are sorted relative to the current orientation of the agent, rather than NESW as for
        the transitions. The order is::

            [data from 'left'] + [data from 'forward'] + [data from 'right'] + [data from 'back']

        Each branch data is organized as::

            [root node information] +
            [recursive branch data from 'left'] +
            [... from 'forward'] +
            [... from 'right] +
            [... from 'back']

        Each node information is composed of 9 features:

        #1:
            if own target lies on the explored branch the current distance from the agent in number of cells is stored.

        #2:
            if another agents target is detected the distance in number of cells from the agents current location\
            is stored

        #3:
            if another agent is detected the distance in number of cells from current agent position is stored.

        #4:
            possible conflict detected
            tot_dist = Other agent predicts to pass along this cell at the same time as the agent, we store the \
             distance in number of cells from current agent position

            0 = No other agent reserve the same cell at similar time

        #5:
            if an not usable switch (for agent) is detected we store the distance.

        #6:
            This feature stores the distance in number of cells to the next branching  (current node)

        #7:
            minimum distance from node to the agent's target given the direction of the agent if this path is chosen

        #8:
            agent in the same direction
            n = number of agents present same direction \
                (possible future use: number of other agents in the same direction in this branch)
            0 = no agent present same direction

        #9:
            agent in the opposite direction
            n = number of agents present other direction than myself (so conflict) \
                (possible future use: number of other agents in other direction in this branch, ie. number of conflicts)
            0 = no agent present other direction than myself

        #10:
            malfunctioning/blokcing agents
            n = number of time steps the oberved agent remains blocked

        #11:
            slowest observed speed of an agent in same direction
            1 if no agent is observed

            min_fractional speed otherwise
        #12:
            number of agents ready to depart but no yet active

        Missing/padding nodes are filled in with -inf (truncated).
        Missing values in present node are filled in with +inf (truncated).


        In case of the root node, the values are [0, 0, 0, 0, distance from agent to target, own malfunction, own speed]
        In case the target node is reached, the values are [0, 0, 0, 0, 0].
        """

        if handle > len(self.env.agents):
            print("ERROR: obs _get - handle ", handle, " len(agents)", len(self.env.agents))
        agent = self.env.agents[handle]  # TODO: handle being treated as index

        if agent.state.is_off_map_state():
            agent_virtual_position = agent.initial_position
        elif agent.state.is_on_map_state():
            agent_virtual_position = agent.position
        elif agent.state == TrainState.DONE:
            agent_virtual_position = agent.target
        else:
            return None

        possible_transitions = self.env.rail.get_transitions(*agent_virtual_position, agent.direction)
        num_transitions = fast_count_nonzero(possible_transitions)

        # if num_transitions > 1 and not exp_dir is None:
        #     exp_dir.pop(0)
        # actually, it is not clear, why this works...in at least one case
        # maybe, the agent has

        # Here information about the agent itself is stored
        distance_map = self.env.distance_map.get()

        # was referring to TreeObsForRailEnv.Node
        root_node_observation = Node(dist_own_target_encountered=0,
                                     dist_other_target_encountered=0,
                                     dist_other_agent_encountered=0,
                                     potential_conflict_dist=0,
                                     potential_conflict_pos=None,
                                     potential_conflict_time_step=-1,
                                     dist_unusable_switch=0,
                                     dist_to_next_branch=0,
                                     dist_min_to_target=distance_map[
                                         (handle, *agent_virtual_position,
                                          agent.direction)],
                                     num_agents_same_direction=0, num_agents_opposite_direction=0,
                                     num_agents_malfunctioning=agent.malfunction_handler.malfunction_down_counter,
                                     speed_min_fractional=agent.speed_counter.speed,
                                     tot_dist=init_dist,
                                     num_agents_ready_to_depart=0,
                                     node_pos=agent_virtual_position,
                                     agent_direction=agent.direction,
                                     childs={})
        # print("root node type:", type(root_node_observation))

        visited = OrderedSet()

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # If only one transition is possible, the tree is oriented with this transition as the forward branch.
        orientation = agent.direction

        if num_transitions == 1:
            orientation = fast_argmax(possible_transitions)

        if section_list is None:
            for i, branch_direction in enumerate([(orientation + i) % 4 for i in range(-1, 3)]):

                if possible_transitions[branch_direction]:
                    new_cell = get_new_position(agent_virtual_position, branch_direction)

                    # modified: init_dist to manipulte time_step in conflict prediction
                    branch_observation, branch_visited = self.explore_branch(handle, new_cell, branch_direction, init_dist, 1)
                    root_node_observation.childs[self.tree_explored_actions_char[i]] = branch_observation

                    visited |= branch_visited
                else:
                    # add cells filled with infinity if no transition is possible
                    root_node_observation.childs[self.tree_explored_actions_char[i]] = -np.inf
        else:
            dir = int(section_list[0][1][1])
            new_cell = get_new_position(agent_virtual_position, dir)
            test = section_list[0][1][0]
            if new_cell[0] != test[0] or new_cell[1] != test[1]:
                print("fehler tree root")
            # modified: init_dist to manipulte time_step in conflict prediction
            branch_observation, branch_visited = self.explore_branch(handle, new_cell, dir, init_dist, 1, section_list)

            # branch_label = self.tree_explored_actions_char[dir]
            branch_label = 'X'
            # branch_label = exp_dir[0][1]
            root_node_observation.childs[branch_label] = branch_observation
            # root_node_observation.childs["X"] = branch_observation

        self.env.dev_obs_dict[handle] = visited

        return root_node_observation

    def explore_branch(self, handle, position, direction, tot_dist, depth, section_list: list = None):
        """
        Utility function to compute tree-based observations.
        We walk along the branch and collect the information documented in the get() function.
        If there is a branching point a new node is created and each possible branch is explored.
        """

        # [Recursive branch opened]
        if depth >= self.max_depth + 1:
            return [], []

        # if handle == 0 and position[0] == 24 and position[1] == 9 and direction == 1:
        #     print("now")

        # if handle == 2 and position[0] == 11 and position[1] == 11 and direction == 2:
        #      print("now")
        # if handle == 2 and position[1]==11:
        #     print (position)

        # Continue along direction until next switch or
        # until no transitions are possible along the current direction (i.e., dead-ends)
        # We treat dead-ends as nodes, instead of going back, to avoid loops
        exploring = True
        last_is_switch = False
        last_is_dead_end = False
        last_is_terminal = False  # wrong cell OR cycle;  either way, we don't want the agent to land here
        last_is_target = False

        visited = OrderedSet()

        agent = self.env.agents[handle]
        distance_map_handle = self.env.distance_map.get()[handle]

        time_per_cell = 1.0 / agent.speed_counter.speed
        own_target_encountered = np.inf
        other_agent_encountered = np.inf
        other_target_encountered = np.inf

        potential_conflict = np.inf
        potential_conflict_pos = None
        potential_conflict_time_step = -1

        unusable_switch = np.inf
        other_agent_same_direction = 0
        other_agent_opposite_direction = 0
        malfunctioning_agent = 0
        min_fractional_speed = 1.
        num_steps = 1
        other_agent_ready_to_depart_encountered = 0

        debug = False

        while exploring:
            # if handle == 2 and position[0] == 12 and position[1] == 11 and direction == 2:
            #     print("now")
            #     debug = True
            debug = handle == 1
            debug = False

            # if debug:
            #     print("pos = {}, tot_dist = {}, pred_time = {}".format(position, tot_dist, predicted_time))

            # #############################
            # #############################
            # Modify here to compute any useful data required to build the end node's features. This code is called
            # for each cell visited between the previous branching node and the next switch / target / dead-end.
            if self.location_has_agent.get(position, 0) == 1:
                if tot_dist < other_agent_encountered:
                    other_agent_encountered = tot_dist

                # Check if any of the observed agents is malfunctioning, store agent with longest duration left
                if self.location_has_agent_malfunction[position] > malfunctioning_agent:
                    malfunctioning_agent = self.location_has_agent_malfunction[position]

                other_agent_ready_to_depart_encountered += self.location_has_agent_ready_to_depart.get(position, 0)

                if self.location_has_agent_direction[position] == direction:
                    # Cummulate the number of agents on branch with same direction
                    other_agent_same_direction += 1

                    # Check fractional speed of agents
                    current_fractional_speed = self.location_has_agent_speed[position]
                    if current_fractional_speed < min_fractional_speed:
                        min_fractional_speed = current_fractional_speed

                else:
                    # If no agent in the same direction was found all agents in that position are other direction
                    # Attention this counts to many agents as a few might be going off on a switch.
                    other_agent_opposite_direction += self.location_has_agent[position]

                # Check number of possible transitions for agent and total number of transitions in cell (type)
            cell_transitions = self.env.rail.get_transitions(*position, direction)
            transition_bit = bin(self.env.rail.get_full_transitions(*position))
            total_transitions = transition_bit.count("1")
            crossing_found = False
            if int(transition_bit, 2) == int('1000010000100001', 2):
                crossing_found = True

            # Register possible future conflict
            # this predicts only the required travel time, not the actual departure
            predicted_time = int(tot_dist * time_per_cell)

            # replaced by init_dist
            # offset time by agents earliest departure
            # env_step = self.env._elapsed_steps
            # if env_step < agent.earliest_departure:
            #     predicted_time += agent.earliest_departure-env_step
            # debug = True
            if self.predictor and predicted_time < self.max_prediction_depth:

                if debug:
                    print("pos = {}, tot_dist = {}, pred_time = {}".format(position, tot_dist, predicted_time))

                int_position = coordinate_to_position(self.env.width, [position])[0]

                # TODO check if this is really a bug. i think, it is not -> problem was the prediction not consider the earliest departure
                # modified -> agent cannot have any conflict before departure
                # need to be handled otherwise
                # at any time step in future, the prediction will start with pred_time = 1
                if tot_dist < self.max_prediction_depth:
                    # if tot_dist < self.max_prediction_depth and agent.earliest_departure < predicted_time:
                    # if agent.earliest_departure > predicted_time:
                    #     continue

                    # if agent.earliest_departure >= predicted_time:
                    #     # print("error")
                    #     debug = True

                    if debug:
                        print("agent({}): pos = {}, time = {}, ed = {}".format(agent.handle, position, predicted_time,
                                                                               agent.earliest_departure))

                    pre_step = max(0, predicted_time - 1)
                    post_step = min(self.max_prediction_depth - 1, predicted_time + 1)

                    # Look for conflicting paths at distance tot_dist
                    if int_position in fast_delete(self.predicted_pos[predicted_time], handle):
                        conflicting_agent = my_fast_where(self.predicted_pos[predicted_time], int_position)
                        for ca in conflicting_agent:
                            if direction != self.predicted_dir[predicted_time][ca] and cell_transitions[
                                self._reverse_dir(
                                    self.predicted_dir[predicted_time][ca])] == 1 and tot_dist < potential_conflict:
                                potential_conflict = tot_dist

                                potential_conflict_pos = self.predictions[ca][predicted_time]
                                potential_conflict_time_step = predicted_time
                                if debug:
                                    print("ca = {}, pred_time = {}".format(ca, predicted_time))

                                # potential_conflict_pos = int_position
                            if self.env.agents[ca].state == TrainState.DONE and tot_dist < potential_conflict:
                                potential_conflict = tot_dist

                                potential_conflict_pos = self.predictions[ca][predicted_time]
                                potential_conflict_time_step = predicted_time
                                if debug:
                                    print("ca = {}, pred_time = {}".format(ca, predicted_time))

                                # potential_conflict_pos = int_position
                    # Look for conflicting paths at distance num_step-1
                    elif int_position in fast_delete(self.predicted_pos[pre_step], handle):
                        conflicting_agent = my_fast_where(self.predicted_pos[pre_step], int_position)
                        for ca in conflicting_agent:
                            if direction != self.predicted_dir[pre_step][ca] \
                                and cell_transitions[self._reverse_dir(self.predicted_dir[pre_step][ca])] == 1 \
                                and tot_dist < potential_conflict:  # noqa: E125
                                potential_conflict = tot_dist

                                potential_conflict_pos = self.predictions[ca][pre_step]
                                potential_conflict_time_step = pre_step

                                if debug:
                                    print("ca = {}, pre_step = {}".format(ca, pre_step))
                                # potential_conflict_pos = int_position
                            if self.env.agents[ca].state == TrainState.DONE and tot_dist < potential_conflict:
                                potential_conflict = tot_dist

                                potential_conflict_pos = self.predictions[ca][pre_step]
                                potential_conflict_time_step = pre_step

                                if debug:
                                    print("ca = {}, pre_step = {}".format(ca, pre_step))
                                # potential_conflict_pos = int_position
                    # Look for conflicting paths at distance num_step+1
                    elif int_position in fast_delete(self.predicted_pos[post_step], handle):
                        conflicting_agent = my_fast_where(self.predicted_pos[post_step], int_position)
                        for ca in conflicting_agent:
                            if direction != self.predicted_dir[post_step][ca] and cell_transitions[self._reverse_dir(
                                self.predicted_dir[post_step][ca])] == 1 \
                                and tot_dist < potential_conflict:  # noqa: E125
                                potential_conflict = tot_dist

                                potential_conflict_pos = self.predictions[ca][post_step]
                                potential_conflict_time_step = post_step

                                if debug:
                                    print("ca = {}, post_step = {}".format(ca, post_step))
                                # potential_conflict_pos = int_position
                            if self.env.agents[ca].state == TrainState.DONE and tot_dist < potential_conflict:
                                potential_conflict = tot_dist

                                potential_conflict_pos = self.predictions[ca][post_step]
                                potential_conflict_time_step = post_step

                                # potential_conflict_pos = int_position
                                if debug:
                                    print("ca = {}, post_step = {}".format(ca, post_step))


            if self.location_has_target and position in self.location_has_target:
                if position != agent.target:
                    if tot_dist < other_target_encountered:
                        other_target_encountered = tot_dist

            if position == agent.target and tot_dist < own_target_encountered:
                own_target_encountered = tot_dist

            # #############################
            # #############################
            if (position[0], position[1], direction) in visited:
                last_is_terminal = True
                break
            visited.add((position[0], position[1], direction))

            # If the target node is encountered, pick that as node. Also, no further branching is possible.
            if fast_position_equal(position, self.env.agents[handle].target):
                last_is_target = True
                break

            # Check if crossing is found --> Not an unusable switch
            if crossing_found:
                # Treat the crossing as a straight rail cell
                total_transitions = 2
            num_transitions = fast_count_nonzero(cell_transitions)

            exploring = False

            # Detect Switches that can only be used by other agents.
            if total_transitions > 2 > num_transitions and tot_dist < unusable_switch:
                unusable_switch = tot_dist

            if num_transitions == 1:
                # Check if dead-end, or if we can go forward along direction
                nbits = total_transitions
                if nbits == 1:
                    # Dead-end!
                    last_is_dead_end = True

                if not last_is_dead_end:
                    # Keep walking through the tree along `direction`
                    exploring = True
                    # convert one-hot encoding to 0,1,2,3
                    direction = fast_argmax(cell_transitions)
                    position = get_new_position(position, direction)
                    num_steps += 1
                    tot_dist += 1
            elif num_transitions > 0:
                # Switch detected
                last_is_switch = True
                break

            elif num_transitions == 0:
                # Wrong cell type, but let's cover it and treat it as a dead-end, just in case
                print("WRONG CELL TYPE detected in tree-search (0 transitions possible) at cell", position[0],
                      position[1], direction)
                last_is_terminal = True
                break

        # `position` is either a terminal node or a switch

        # #############################
        # #############################
        # Modify here to append new / different features for each visited cell!

        if last_is_target:
            dist_to_next_branch = tot_dist
            dist_min_to_target = 0
        elif last_is_terminal:
            dist_to_next_branch = np.inf
            dist_min_to_target = distance_map_handle[position[0], position[1], direction]
        else:
            dist_to_next_branch = tot_dist
            dist_min_to_target = distance_map_handle[position[0], position[1], direction]

        # both seems to have no effect
        if False and dist_min_to_target == 0:
            depth = self.max_depth

        if False and tot_dist > 150:
            depth = self.max_depth

        # TreeObsForRailEnv.Node
        node = Node(dist_own_target_encountered=own_target_encountered,
                    dist_other_target_encountered=other_target_encountered,
                    dist_other_agent_encountered=other_agent_encountered,
                    potential_conflict_dist=potential_conflict,
                    potential_conflict_pos=potential_conflict_pos,
                    potential_conflict_time_step=potential_conflict_time_step,
                    dist_unusable_switch=unusable_switch,
                    dist_to_next_branch=dist_to_next_branch,
                    dist_min_to_target=dist_min_to_target,
                    num_agents_same_direction=other_agent_same_direction,
                    num_agents_opposite_direction=other_agent_opposite_direction,
                    num_agents_malfunctioning=malfunctioning_agent,
                    speed_min_fractional=min_fractional_speed,
                    tot_dist=tot_dist,
                    num_agents_ready_to_depart=other_agent_ready_to_depart_encountered,
                    node_pos=position,
                    agent_direction=direction,
                    childs={})

        # #############################
        # #############################
        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # Get the possible transitions
        if section_list is None:
            # if tot_dist< 150:

            possible_transitions = self.env.rail.get_transitions(*position, direction)
            for i, branch_direction in enumerate([(direction + 4 + i) % 4 for i in range(-1, 3)]):
                if last_is_dead_end and self.env.rail.get_transition((*position, direction),
                                                                     (branch_direction + 2) % 4):

                    # Swap forward and back in case of dead-end, so that an agent can learn that going forward takes
                    # it back
                    new_cell = get_new_position(position, (branch_direction + 2) % 4)
                    branch_observation, branch_visited = self.explore_branch(handle,
                                                                             new_cell,
                                                                             (branch_direction + 2) % 4,
                                                                             tot_dist + 1,
                                                                             depth + 1)
                    node.childs[self.tree_explored_actions_char[i]] = branch_observation
                    if len(branch_visited) != 0:
                        visited |= branch_visited
                elif last_is_switch and possible_transitions[branch_direction]:
                    new_cell = get_new_position(position, branch_direction)
                    branch_observation, branch_visited = self.explore_branch(handle,
                                                                             new_cell,
                                                                             branch_direction,
                                                                             tot_dist + 1,
                                                                             depth + 1)
                    node.childs[self.tree_explored_actions_char[i]] = branch_observation
                    if len(branch_visited) != 0:
                        visited |= branch_visited
                else:
                    # no exploring possible, add just cells with infinity
                    node.childs[self.tree_explored_actions_char[i]] = -np.inf
        else:
            if depth < len(section_list):
                dir = int(section_list[depth][1][1])
                new_cell = get_new_position(position, dir)
                test = section_list[depth][1][0]
                if new_cell[0] != test[0] or new_cell[1] != test[1]:
                    print("fehler exp branch")

                branch_observation, branch_visited = self.explore_branch(handle,
                                                                         new_cell,
                                                                         dir,
                                                                         tot_dist + 1,
                                                                         depth + 1,
                                                                         section_list)
                # branch_label = self.tree_explored_actions_char[dir]
                # branch_label = exp_dir[depth][1]
                branch_label = 'X'
                node.childs[branch_label] = branch_observation
                # node.childs["X"] = branch_observation
            else:
                # print("target?!")
                pass

        if depth == self.max_depth:
            node.childs.clear()
        return node, visited

    def util_print_obs_subtree(self, tree: Node):
        """
        Utility function to print tree observations returned by this object.
        """
        self.print_node_features(tree, "root", "")
        for direction in self.tree_explored_actions_char:
            self.print_subtree(tree.childs[direction], direction, "\t")

    @staticmethod
    def print_node_features(node: Node, label, indent):
        print(indent, "Direction ", label, ": ", node.dist_own_target_encountered, ", ",
              node.dist_other_target_encountered, ", ", node.dist_other_agent_encountered, ", ",
              node.dist_potential_conflict, ", ", node.dist_unusable_switch, ", ", node.dist_to_next_branch, ", ",
              node.dist_min_to_target, ", ", node.num_agents_same_direction, ", ", node.num_agents_opposite_direction,
              ", ", node.num_agents_malfunctioning, ", ", node.speed_min_fractional, ", ",
              node.num_agents_ready_to_depart)

    def print_subtree(self, node, label, indent):
        if node == -np.inf or not node:
            print(indent, "Direction ", label, ": -np.inf")
            return

        self.print_node_features(node, label, indent)

        if not node.childs:
            return

        for direction in self.tree_explored_actions_char:
            self.print_subtree(node.childs[direction], direction, indent + "\t")

    def set_env(self, env: Environment):
        super().set_env(env)
        if self.predictor:
            self.predictor.set_env(self.env)

    def _reverse_dir(self, direction):
        return int((direction + 2) % 4)


class GlobalObsForRailEnv(ObservationBuilder):
    """
    Gives a global observation of the entire rail environment.
    The observation is composed of the following elements:

        - transition map array with dimensions (env.height, env.width, 16),\
          assuming 16 bits encoding of transitions.

        - obs_agents_state: A 3D array (map_height, map_width, 5) with
            - first channel containing the agents position and direction
            - second channel containing the other agents positions and direction
            - third channel containing agent/other agent malfunctions
            - fourth channel containing agent/other agent fractional speeds
            - fifth channel containing number of other agents ready to depart

        - obs_targets: Two 2D arrays (map_height, map_width, 2) containing respectively the position of the given agent\
         target and the positions of the other agents targets (flag only, no counter!).
    """

    def __init__(self):
        super(GlobalObsForRailEnv, self).__init__()

    def set_env(self, env: Environment):
        super().set_env(env)

    def reset(self):
        self.rail_obs = np.zeros((self.env.height, self.env.width, 16))
        for i in range(self.rail_obs.shape[0]):
            for j in range(self.rail_obs.shape[1]):
                bitlist = [int(digit) for digit in bin(self.env.rail.get_full_transitions(i, j))[2:]]
                bitlist = [0] * (16 - len(bitlist)) + bitlist
                self.rail_obs[i, j] = np.array(bitlist)

    def get(self, handle: int = 0) -> (np.ndarray, np.ndarray, np.ndarray):

        agent = self.env.agents[handle]
        if agent.state.is_off_map_state():
            agent_virtual_position = agent.initial_position
        elif agent.state.is_on_map_state():
            agent_virtual_position = agent.position
        elif agent.state == TrainState.DONE:
            agent_virtual_position = agent.target
        else:
            return None

        obs_targets = np.zeros((self.env.height, self.env.width, 2))
        obs_agents_state = np.zeros((self.env.height, self.env.width, 5)) - 1

        # TODO can we do this more elegantly?
        # for r in range(self.env.height):
        #     for c in range(self.env.width):
        #         obs_agents_state[(r, c)][4] = 0
        obs_agents_state[:, :, 4] = 0

        obs_agents_state[agent_virtual_position][0] = agent.direction
        obs_targets[agent.target][0] = 1

        for i in range(len(self.env.agents)):
            other_agent: EnvAgent = self.env.agents[i]

            # ignore other agents not in the grid any more
            if other_agent.state == TrainState.DONE:
                continue

            obs_targets[other_agent.target][1] = 1

            # second to fourth channel only if in the grid
            if other_agent.position is not None:
                # second channel only for other agents
                if i != handle:
                    obs_agents_state[other_agent.position][1] = other_agent.direction
                obs_agents_state[other_agent.position][2] = other_agent.malfunction_handler.malfunction_down_counter
                obs_agents_state[other_agent.position][3] = other_agent.speed_counter.speed
            # fifth channel: all ready to depart on this position
            if other_agent.state.is_off_map_state():
                obs_agents_state[other_agent.initial_position][4] += 1
        return self.rail_obs, obs_agents_state, obs_targets


class LocalObsForRailEnv(ObservationBuilder):
    """
    !!!!!!WARNING!!! THIS IS DEPRACTED AND NOT UPDATED TO FLATLAND 2.0!!!!!
    Gives a local observation of the rail environment around the agent.
    The observation is composed of the following elements:

        - transition map array of the local environment around the given agent, \
          with dimensions (view_height,2*view_width+1, 16), \
          assuming 16 bits encoding of transitions.

        - Two 2D arrays (view_height,2*view_width+1, 2) containing respectively, \
        if they are in the agent's vision range, its target position, the positions of the other targets.

        - A 2D array (view_height,2*view_width+1, 4) containing the one hot encoding of directions \
          of the other agents at their position coordinates, if they are in the agent's vision range.

        - A 4 elements array with one hot encoding of the direction.

    Use the parameters view_width and view_height to define the rectangular view of the agent.
    The center parameters moves the agent along the height axis of this rectangle. If it is 0 the agent only has
    observation in front of it.

    .. deprecated:: 2.0.0
    """

    def __init__(self, view_width, view_height, center):

        super(LocalObsForRailEnv, self).__init__()
        self.view_width = view_width
        self.view_height = view_height
        self.center = center
        self.max_padding = max(self.view_width, self.view_height - self.center)

    def reset(self):
        # We build the transition map with a view_radius empty cells expansion on each side.
        # This helps to collect the local transition map view when the agent is close to a border.
        self.max_padding = max(self.view_width, self.view_height)
        self.rail_obs = np.zeros((self.env.height,
                                  self.env.width, 16))
        for i in range(self.env.height):
            for j in range(self.env.width):
                bitlist = [int(digit) for digit in bin(self.env.rail.get_full_transitions(i, j))[2:]]
                bitlist = [0] * (16 - len(bitlist)) + bitlist
                self.rail_obs[i, j] = np.array(bitlist)

    def get(self, handle: int = 0) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        agents = self.env.agents
        agent = agents[handle]

        # Correct agents position for padding
        # agent_rel_pos[0] = agent.position[0] + self.max_padding
        # agent_rel_pos[1] = agent.position[1] + self.max_padding

        # Collect visible cells as set to be plotted
        visited, rel_coords = self.field_of_view(agent.position, agent.direction, )
        local_rail_obs = None

        # Add the visible cells to the observed cells
        self.env.dev_obs_dict[handle] = set(visited)

        # Locate observed agents and their coresponding targets
        local_rail_obs = np.zeros((self.view_height, 2 * self.view_width + 1, 16))
        obs_map_state = np.zeros((self.view_height, 2 * self.view_width + 1, 2))
        obs_other_agents_state = np.zeros((self.view_height, 2 * self.view_width + 1, 4))
        _idx = 0
        for pos in visited:
            curr_rel_coord = rel_coords[_idx]
            local_rail_obs[curr_rel_coord[0], curr_rel_coord[1], :] = self.rail_obs[pos[0], pos[1], :]
            if pos == agent.target:
                obs_map_state[curr_rel_coord[0], curr_rel_coord[1], 0] = 1
            else:
                for tmp_agent in agents:
                    if pos == tmp_agent.target:
                        obs_map_state[curr_rel_coord[0], curr_rel_coord[1], 1] = 1
            if pos != agent.position:
                for tmp_agent in agents:
                    if pos == tmp_agent.position:
                        obs_other_agents_state[curr_rel_coord[0], curr_rel_coord[1], :] = np.identity(4)[
                            tmp_agent.direction]

            _idx += 1

        direction = np.identity(4)[agent.direction]
        return local_rail_obs, obs_map_state, obs_other_agents_state, direction

    def get_many(self, handles: Optional[List[int]] = None) -> Dict[
        int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Called whenever an observation has to be computed for the `env` environment, for each agent with handle
        in the `handles` list.
        """

        return super().get_many(handles)

    def field_of_view(self, position, direction, state=None):
        # Compute the local field of view for an agent in the environment
        data_collection = False
        if state is not None:
            temp_visible_data = np.zeros(shape=(self.view_height, 2 * self.view_width + 1, 16))
            data_collection = True
        if direction == 0:
            origin = (position[0] + self.center, position[1] - self.view_width)
        elif direction == 1:
            origin = (position[0] - self.view_width, position[1] - self.center)
        elif direction == 2:
            origin = (position[0] - self.center, position[1] + self.view_width)
        else:
            origin = (position[0] + self.view_width, position[1] + self.center)
        visible = list()
        rel_coords = list()
        for h in range(self.view_height):
            for w in range(2 * self.view_width + 1):
                if direction == 0:
                    if 0 <= origin[0] - h < self.env.height and 0 <= origin[1] + w < self.env.width:
                        visible.append((origin[0] - h, origin[1] + w))
                        rel_coords.append((h, w))
                    # if data_collection:
                    #    temp_visible_data[h, w, :] = state[origin[0] - h, origin[1] + w, :]
                elif direction == 1:
                    if 0 <= origin[0] + w < self.env.height and 0 <= origin[1] + h < self.env.width:
                        visible.append((origin[0] + w, origin[1] + h))
                        rel_coords.append((h, w))
                    # if data_collection:
                    #    temp_visible_data[h, w, :] = state[origin[0] + w, origin[1] + h, :]
                elif direction == 2:
                    if 0 <= origin[0] + h < self.env.height and 0 <= origin[1] - w < self.env.width:
                        visible.append((origin[0] + h, origin[1] - w))
                        rel_coords.append((h, w))
                    # if data_collection:
                    #    temp_visible_data[h, w, :] = state[origin[0] + h, origin[1] - w, :]
                else:
                    if 0 <= origin[0] - w < self.env.height and 0 <= origin[1] - h < self.env.width:
                        visible.append((origin[0] - w, origin[1] - h))
                        rel_coords.append((h, w))
                    # if data_collection:
                    #    temp_visible_data[h, w, :] = state[origin[0] - w, origin[1] - h, :]
        if data_collection:
            return temp_visible_data
        else:
            return visible, rel_coords

import math
from typing import Tuple

import numpy as np
from stopwatch import Stopwatch

from TreeCrawler import TreeCrawler
from fl_base.fast_methods import my_fast_where, fast_delete, fast_count_nonzero
from fl_base.observations import TreeObsForRailEnv, Node
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_shortest_paths import get_shortest_paths, get_action_for_move, get_new_position_for_action, get_k_shortest_paths
from flatland.envs.step_utils.states import TrainState
from flatland.utils.ordered_set import OrderedSet


class TreePathPlaner(TreeObsForRailEnv):
    max_explore_depth = 999

    def __init__(self, predictor, env: RailEnv, prediction_depth, tree_depth: int = 10, is_dyn_depth: bool = False, is_log: bool = True):
        super().__init__(self.max_explore_depth, predictor)
        self.tree_depth = tree_depth
        self.predictor = predictor
        # self.predictor.set_env(env)
        self.env = env
        self.tc = TreeCrawler()
        self.handles = []
        self.is_log = is_log
        # self.max_depth = 10
        self.prediction_depth = prediction_depth
        self.debug = False
        self.use_dynamic_tree_depth = is_dyn_depth
        self.label_to_action = {"L": 1, "F": 2, "R": 3, "B": 0}
        self.sw_path = Stopwatch(2)
        self.sw_plan = Stopwatch(2)
        self.departures = {}

    def get_agent_prio(self, handle, min_travel_time):

        agent: EnvAgent = self.env.agents[handle]
        min_dist = self.env.distance_map.get()[agent.handle, agent.initial_position[0], agent.initial_position[1], agent.initial_direction]
        min_dist *= self.get_time_per_cell(handle)
        prio = ((agent.latest_arrival - agent.earliest_departure) - min_dist) / min_dist
        # prio = agent.latest_arrival - ((agent.latest_arrival - agent.earliest_departure) - min_dist) / min_dist
        return prio

    def plan(self):
        env = self.env
        stopwatch = Stopwatch(2)
        stopwatch.restart()

        distance_map: DistanceMap = env.distance_map
        shortest_paths = get_shortest_paths(distance_map, max_depth=self.prediction_depth)

        prio = {}
        agents = env.agents
        for a in agents:
            # prio[a.handle] = (a.latest_arrival - a.earliest_departure) - (len(shortest_paths[a.handle]) - 1) # reverse = True

            if shortest_paths[a.handle]:
                min_travel_time = len(shortest_paths[a.handle]) / a.speed_counter.speed
            else:
                min_travel_time = 999

            prio[a.handle] = self.get_agent_prio(a.handle, min_travel_time)
        prio = sorted(prio.items(), key=lambda x: x[1], reverse=True)

        self.reset()
        self.initLUT()
        if self.is_log:
            print("###")
            print("Start initial path planning...")

        self.sw_path.reset()
        self.sw_plan.reset()

        self.departures = {}
        n_agents = len(prio)
        for i in range(0, n_agents):
            handle = prio[i][0]
            if self.is_log:
                print("Planning {}/{}: agent({}) - tw = {}-{},({})->({})".format(i + 1, n_agents, handle, agents[handle].earliest_departure,
                                                                                agents[handle].latest_arrival,
                                                                                 agents[handle].initial_position,
                                                                                 agents[handle].target))
            self.plan_agent_path(agents[handle])

            for k, v in enumerate(self.predictor.predictions[handle]):
                if 0 < v[4] < 4:
                    self.departures[k] = v[1:3]
                    if self.is_log:
                        print("departure: ", v)
                    break
        if self.is_log:
            print("Initial planning done!")
            print("###")
            print("duration = ", stopwatch)
            print("duration path = {}, planing = {}".format(self.sw_path, self.sw_plan))

        return self.predictor.predictions

    def plan_agent_path(self, agent: EnvAgent):

        if agent.state == TrainState.DONE:
            print("agent {} already done!".format(agent.handle))
            return

        handle = agent.handle
        is_departured = True
        # if agent is not departured...
        if agent.position is None:
            is_departured = False
            # env bug!?
            agent.earliest_departure = max(1, agent.earliest_departure)

            departure = agent.earliest_departure
            pos = agent.initial_position
            departure = self.get_valid_departure(agent.handle, pos, departure)

            time_step = self.env._elapsed_steps
            departure = max(departure, time_step - 1)
            agent.earliest_departure = departure

        # [t, pos.x, pos.y, dir, action taken to get here]

        self.sw_path.start()
        prediction = self.get_shortest_path(handle)
        self.sw_path.stop()

        self.sw_plan.start()
        predicted_path_nodes = self.get_path_nodes_from_prediction(handle, prediction)
        has_conflict = self.check_node_conflicts(predicted_path_nodes)

        if not has_conflict:
            self.predictor.set_prediction(handle, prediction)
            # self.update_observation_prediction()
        else:
            # look for alternative path
            if is_departured and self.search_tree(agent, prediction):
                if self.is_log:
                    print("agent {} uses alternative path".format(agent.handle))
            else:
                if True:
                    final_prediction = self.wait_in_sections(agent, predicted_path_nodes, prediction)
                    if self.check_for_conflicts(agent, final_prediction):
                        final_prediction, delay = self.wait_in_place(agent, prediction)
                        if self.is_log:
                            print("agent {} waits in place for {}".format(agent.handle, delay))
                    else:
                        if self.is_log:
                            print("agent {} waits in sections".format(agent.handle))
                else:
                    final_prediction, delay = self.wait_in_place(agent, prediction)
                    if self.is_log:
                        print("agent {} waits in place for {}".format(agent.handle, delay))

                self.predictor.set_prediction2(handle, final_prediction)
            # self.update_observation_prediction()

        self.update_observation_prediction()
        self.sw_plan.stop()

    def get_valid_departure(self, handle, pos, departure):
        is_valid = False
        while not is_valid:
            is_valid = True
            for k, v in self.predictor.departures.items():
                if k != handle:
                    if v is None:
                        print("depature pos err: agent {} at {}, dep. pos = {} ".format(handle, pos,v))
                    else:
                        if v[0] == departure + 1 and self.is_pos_equal(v[1:3], pos):
                            departure += 1
                            is_valid = False
        return departure

    def search_tree(self, agent: EnvAgent, prediction):
        init_dist = (agent.earliest_departure + 1) / agent.speed_counter.speed
        time_step = self.env._elapsed_steps

        if init_dist < time_step:
            init_dist = max(init_dist, time_step)

        pos = agent.position
        if pos == None:
            pos = agent.initial_position
            init_dist = self.get_valid_departure(agent.handle, pos, init_dist)

        # --- compute tree obs and get path sections ---
        self.max_depth = self.tree_depth
        target_obs = False
        while not target_obs:
            tree_obs = super().get(agent.handle, init_dist)
            tree_sections = self.get_path_nodes_from_tree_obs(agent.target, tree_obs, prediction)
            l = len(tree_sections) - 1
            if (len(tree_sections) > self.max_depth or tree_sections[l][0].dist_min_to_target != 0):
                # print("Error: Max. depth reached: {}".format(len(path_sections)))
                if not self.use_dynamic_tree_depth:
                    target_obs = True
                if self.max_depth < 10:
                    self.max_depth += 1
                else:
                    self.max_depth += 1
            else:
                target_obs = True

            self.max_depth = self.max_explore_depth
        # -------

        # time depends on tree obs -> init_dist
        alt_paths = []
        replan_from_node = tree_sections[0][0]
        alt_paths = self.tc.crawl(replan_from_node)

        if len(alt_paths) > 0:
            min = math.inf
            ap = None
            for v in alt_paths:
                if v[0] < min:
                    min = v[0]
                    ap = v

            prediction = self.update_path(agent.handle, ap[1])
            self.predictor.set_prediction(agent.handle, prediction)
            # self.update_observation_prediction()
            return True
        else:
            return False

    def update_path(self, handle, way_nodes):
        # waynodes = [pos, action str, node] -> result of path finding in tree obs

        # to init data structure
        ap = self.get_path_actions_for_section_by_node(handle, way_nodes[0], way_nodes[1], 2)
        for i in range(2, len(way_nodes)):
            ap = np.append(ap, self.get_path_actions_for_section_by_node(handle, way_nodes[i - 1], way_nodes[i], 2)[1:], axis=0)
        alt_path = ap
        # start_pos = way_nodes[0][0]
        start_time = way_nodes[0][2].tot_dist
        pred = []
        for i in range(1, len(way_nodes)):
            from_n = way_nodes[i - 1]
            to_n = way_nodes[i]
            # section_prediction = self.get_path_actions_section([from_n[1], from_n[3], from_n[0]], [to_n[1], to_n[3], to_n[0]])
            section_prediction = self.get_path_actions_for_section_by_node(handle, from_n, to_n, 2)

            pred.append(section_prediction)

        new_path = {}
        time = int(start_time)
        for i in range(0, len(pred)):
            if i == 0:
                s = 0
            else:
                s = 1
            for j in range(s, len(pred[i])):
                new_path[time] = ([time, *pred[i][j]])
                time += 1

        return self.predictor.set_prediction2(handle, new_path)

    #########################

    def wait_in_sections(self, agent, path_sections, prediction):

        pred = []
        delay = []

        is_departured = True
        start_pos = agent.position
        start_dir = agent.direction
        if start_pos is None:
            start_pos = agent.initial_position
            start_dir = agent.initial_direction
            is_departured = False

        t0 = -1
        for k, v in enumerate(prediction):
            if self.is_pos_equal(v[1:3], start_pos) and v[3] == start_dir:
                t0 = k
                break

        t_start = t0
        is_valid = False
        is_valid = is_departured
        while not is_valid:
            is_valid = True
            if t_start in self.departures:
                if self.is_pos_equal(self.departures[t_start], start_pos):
                    # t_start += 1
                    t_start += 1  # test
                    is_valid = False
                else:
                    is_valid = True

        # get waypoints + action for 1. section
        target = path_sections[0].node_pos

        section_prediction = self.get_path_actions_for_section_by_pos(agent.handle, start_pos, start_dir, target, prediction)
        pred.append(section_prediction)

        # get valid departure time
        ts = t_start
        is_valid = False
        while not is_valid:
            is_valid = True
            # if self.departures and ts in self.departures:
            if ts in self.departures:
                if self.is_pos_equal(self.departures[ts], start_pos):
                    ts += 1
                    is_valid = False
                else:
                    is_valid = True

        # agent_departure = path_sections[0].tot_dist
        agent_departure = ts
        tot_delay = path_sections[0].tot_dist + ts - t0
        # tot_delay =t_start
        for i in range(1, len(path_sections)):
            from_n = path_sections[i - 1]
            to_n = path_sections[i]

            section_prediction = self.get_path_actions_for_section_by_pos(agent.handle, from_n.node_pos, from_n.agent_direction,
                                                                          to_n.node_pos,
                                                                          prediction)
            section_delay = self.get_section_delay_by_nodes(agent, tot_delay, prediction, from_n, to_n)
            pred.append(section_prediction[1:])

            # this is the delay the agent must wait before entering this section
            # -1 -> transitions
            delay.append(section_delay - 1)
            # delay.append(section_delay)
            tot_delay += section_delay

        delay.append(0)

        new_path = {}
        t = agent_departure
        for k, s in enumerate(pred):
            for k1, pt in enumerate(s):
                new_path[t] = pt
                t += 1

                if k1 == len(s) - 2:
                    idx = max(len(s) - 2, 0)
                    wp = s[idx].copy()
                    wp[3] = 4
                    for i in range(0, delay[k]):
                        new_path[t] = wp
                        t += 1

        final_pred = {}
        for k, v in new_path.items():
            final_pred[int(k)] = [int(k), *v]

        return final_pred

    def wait_in_place(self, agent, prediction):

        is_departured = True
        start_pos = agent.position
        if start_pos is None:
            start_pos = agent.initial_position
            is_departured = False

        t0 = -1
        for k, v in enumerate(prediction):
            if self.is_pos_equal(v[1:3], start_pos):
                t0 = k
                break

        time_per_cell = self.get_time_per_cell(agent.handle)
        init_dist = t0 / time_per_cell
        # delay = self.get_delay_for_path(agent, init_dist, prediction)
        delay = self.get_delay_for_path_old(agent, init_dist, prediction)
        delay *= self.get_time_per_cell(agent.handle)

        is_valid = False
        is_valid = is_departured
        while not is_valid:
            valid_start = self.get_valid_departure(agent.handle, start_pos, t0 + delay)
            if valid_start == t0 + delay:
                is_valid = True
            else:
                delay += 1

        new_path = {}
        t_start = t0 + delay
        if not is_departured:
            t_start += 1

        if is_departured:
            if self.is_log:
                print("departured agent replaned ###########################################################################")
            for i in range(0, delay):
                new_path[t0 + i] = np.array([t0 + i, agent.position[0], agent.position[1], agent.direction, 4])

        for i in range(t0, len(prediction)):
            new_path[t_start] = prediction[i].copy()
            new_path[t_start][0] = t_start
            t_start += 1
            if t_start == self.max_prediction_depth - 1:
                break

        return new_path, delay

    def get_delay_for_path(self, agent, init_dist, prediction):
        has_conflict = True
        delay = -1
        while has_conflict:
            delay += 1
            has_conflict = self.check_for_conflicts_delay(agent, prediction, delay)

        return delay

    def get_delay_for_path_old(self, agent, init_dist, prediction):

        sections = self.get_sections(prediction)
        has_conflict = True
        i = 0
        while has_conflict:
            t1 = super().get(agent.handle, init_dist + i, sections)
            path_sections = self.get_path_nodes(agent.target, t1)
            conflicts = []
            has_conflict = False
            for n in path_sections:
                # n = sec[0]
                is_conflict = 0 < n.potential_conflict_dist < math.inf and n.node_pos != agent.target
                if is_conflict:
                    has_conflict = True
                conflicts.append(is_conflict)

            if has_conflict:
                i = i + 1
            else:
                break

        return i

    def get_sections(self, prediction):
        sections = []
        # p0, p1
        t0 = -1
        for p in prediction:
            if p[1] > 0:
                t0 = int(p[0])
                break

        p0 = prediction[t0][1:3]
        d0 = prediction[t0][3]
        t0 += 1

        t0 = min(t0, len(prediction)-1)
        p1 = prediction[t0][1:3]
        d1 = prediction[t0][3]
        sections.append([[p0, d0], [p1, d1]])
        # t0 += 1

        for t in range(t0, len(prediction) - 1):
            if prediction[t][1] < 0:
                break
            cell_transitions = self.env.rail.get_transitions(int(prediction[t][1]), int(prediction[t][2]), int(prediction[t][3]))
            num_transitions = fast_count_nonzero(cell_transitions)
            if num_transitions > 1:
                p0 = prediction[t][1:3]
                d0 = prediction[t][3]
                p1 = prediction[t + 1][1:3]
                d1 = prediction[t + 1][3]
                sections.append([[p0, d0], [p1, d1]])
                # t = t + 1

        return sections

    def get_path_nodes(self, target, t1):
        nodes = []
        is_target_reached = False
        cur_node = t1
        while not is_target_reached:
            nodes.append(cur_node)
            if len(cur_node.childs) > 0:
                cur_node = list(cur_node.childs.values())[0]
            else:
                if not self.is_pos_equal(cur_node.node_pos, target):
                    if self.is_log:
                        print("err path nodes - node/target = {} / {}".format(cur_node.node_pos, target))
                    break
                else:
                    is_target_reached = True

        return nodes

    def get_section_delay_by_nodes(self, agent, init_dist, prediction, from_node: Node, to_node: Node):
        pos = from_node.node_pos
        dir = from_node.agent_direction
        # init_dist = path_sections[0][0].tot_dist

        target = to_node.node_pos
        handle = agent.handle

        has_conflict = False
        i = 0
        while has_conflict:
            # t1, _ = super().explore_branch(handle, pos, dir, init_dist + i, 0, path_dirs)
            t1, _ = super().explore_branch(handle, pos, dir, init_dist + i, self.max_depth - 1)
            sections = self.get_path_nodes_from_tree_obs(target, t1, prediction)
            conflicts = []
            has_conflict = False
            for sec in sections:
                n = sec[0]
                is_conflict = 0 < n.potential_conflict_dist < math.inf and n.node_pos != agent.target
                is_conflict1 = 0 < n.potential_conflict_dist < math.inf

                if is_conflict1 != is_conflict:
                    if self.is_log:
                        print("Auch das kommt vor...is_conflict = {}, is_conflict1={}".format(is_conflict, is_conflict1))
                    is_conflict = is_conflict or is_conflict1

                if is_conflict:
                    has_conflict = True
                conflicts.append(is_conflict)
                # if is_conflict:
                #     print("conflict")

            if has_conflict:
                i = i + 1
            else:
                break
        # why i + 1?
        return i + 1

    def get_path_actions_for_section_by_node(self, handle, start_way_node: Tuple[Tuple, str, Node],
                                             target_way_node: Tuple[Tuple, str, Node],
                                             init_action: int = 0) -> object:
        # returns: [time_step, *new_position(r,c), new_direction, action]

        start_pos = start_way_node[0]
        target_pos = target_way_node[0]
        time_per_cell = self.get_time_per_cell(handle)

        # l,r,f,b
        start_action = self.label_to_action[start_way_node[1]]
        start_node = start_way_node[2]

        predictions = []
        # predictions.append([*start_pos, start_node.agent_direction, start_action])
        for i in range(0, time_per_cell):
            predictions.append([*start_pos, start_node.agent_direction, init_action])

        next_pos_dir = get_new_position_for_action(start_pos, start_node.agent_direction, start_action, self.env.rail)
        pos = next_pos_dir[0]
        dir = next_pos_dir[1]

        for i in range(0, time_per_cell):
            predictions.append([*pos, dir, start_action])

        while not self.is_pos_equal(pos, target_pos):
            next_pos_dir = get_new_position_for_action(pos, dir, RailEnvActions.MOVE_FORWARD, self.env.rail)
            pos = next_pos_dir[0]
            dir = next_pos_dir[1]
            for i in range(0, time_per_cell):
                predictions.append([*pos, dir, int(RailEnvActions.MOVE_FORWARD)])

        # print("done")

        return predictions

    def get_path_actions_for_section_by_pos(self, handle, start_pos, start_dir, target_pos, prediction) -> object:
        return self.get_path_actions_for_section_by_pos_speed(handle, start_pos, start_dir, target_pos, prediction)

        # returns: [time_step, *new_position(r,c), new_direction, action]

        t0 = -1
        for k, v in enumerate(prediction):
            if self.is_pos_equal(v[1:3], start_pos) and int(v[3]) == int(start_dir):
                t0 = k
                break

        path_actions = []
        init_action = 2
        cur_dir = int(prediction[t0][3])
        path_actions.append([*start_pos, cur_dir, init_action])

        for t in range(t0, len(prediction)):
            cur_pos = prediction[t][1:3]
            cur_pos = (int(cur_pos[0]), int(cur_pos[1]))
            cur_dir = int(prediction[t][3])

            next_pos = prediction[t + 1][1:3]
            next_pos = (int(next_pos[0]), int(next_pos[1]))
            next_dir = int(prediction[t + 1][3])
            action = int(get_action_for_move(cur_pos, cur_dir, next_pos, next_dir, self.env.rail))

            path_actions.append([*next_pos, next_dir, action])
            if self.is_pos_equal(next_pos, target_pos):
                break

        return path_actions

    def get_path_actions_for_section_by_pos_speed(self, handle, start_pos, start_dir, target_pos, prediction) -> object:
        # returns: [time_step, *new_position(r,c), new_direction, action]

        t0 = -1
        for k, v in enumerate(prediction):
            if self.is_pos_equal(v[1:3], start_pos) and int(v[3]) == int(start_dir):
                t0 = k
                break

        time_per_cell = self.get_time_per_cell(handle)
        path_actions = []
        init_action = 2
        cur_dir = int(prediction[t0][3])

        for i in range(0, time_per_cell):
            path_actions.append([*start_pos, cur_dir, init_action])

        t0 = t0 - 1 + time_per_cell

        for t in range(t0, len(prediction) - 1, time_per_cell):
            cur_pos = prediction[t][1:3]
            cur_pos = (int(cur_pos[0]), int(cur_pos[1]))
            cur_dir = int(prediction[t][3])

            next_pos = prediction[t + 1][1:3]
            next_pos = (int(next_pos[0]), int(next_pos[1]))
            next_dir = int(prediction[t + 1][3])

            tmp_a = get_action_for_move(cur_pos, cur_dir, next_pos, next_dir, self.env.rail)
            if tmp_a is None:
                tmp_a = 4
            action = int(tmp_a)

            for i in range(0, time_per_cell):
                path_actions.append([*next_pos, next_dir, action])

            if self.is_pos_equal(next_pos, target_pos):
                break

        return path_actions


    def get_path_nodes_from_tree_obs(self, target, treeObs, predictions):

        nodes = []
        # root key does not matter
        cur_node = ["F", treeObs]
        d = ["L", "F", "R", "B"]
        tmp_pos = (-1, -1)
        prev_node = None
        i = 0
        for pred in predictions:
            pos = pred[1:3]
            dir = pred[3]
            action = pred[4]
            node_label = d[int(action - 1)]

            if len(cur_node) == 0 and not (tmp_pos[0] == pos[0] and tmp_pos[1] == pos[1]):
                cur_node.append(node_label)
                if node_label in node.childs:
                    cur_node.append(node.childs[node_label])
                else:
                    break

                nodes.append([prev_node, prev_node.node_pos, int(pred[3]), node_label])

                i = 1
                if prev_node.node_pos == target:
                    break

            if self.debug:
                print("pos={}, action={}".format(pos, node_label))

            if action == 4 or len(cur_node) == 0:
                continue

            if self.is_pos_equal(pos, cur_node[1].node_pos) and dir == cur_node[1].agent_direction:
                node = cur_node[1]
                prev_node = node
                tmp_pos = pos
                cur_node.clear()
                if len(node.childs) == 0:
                    nodes.append([node, node.node_pos, node.agent_direction, "F"])
                    break

        if self.debug:
            print("#nodes=", len(nodes))

        return nodes

    def get_path_nodes_from_prediction(self, handle, prediction):

        # {t, r, c, dir, action taken to come here}
        for t in range(0, len(prediction)):
            # if (prediction[t][1] >= 0):
            if (0 < prediction[t][4] < 4):
                break
        pos = prediction[t][1:3]
        pos = (int(pos[0]), int(pos[1]))
        dir = int(prediction[t][3])
        tot_dist = prediction[t][0]
        nodes = []
        while t < len(prediction) - 2:
            if pos is None or len(pos) != 2:
                break

            n, exp = self.explore_branch(handle, pos, dir, tot_dist, self.max_depth)
            n: Node
            nodes.append(n)
            d = 0
            t = int(n.tot_dist)  # /agent.speed
            l = len(prediction)
            if t + 1 < l:
                action = int(prediction[t + 1][4])
            else:
                action = 4

            while (action == 4 and t + d + 1 < l):
                # increment delay
                d += 1
                action = int(prediction[t + d][4])

            t = int(n.tot_dist + d)  # /agent.speed
            if t + 1 >= l:
                break

            pos = prediction[t][1:3]
            pos = (int(pos[0]), int(pos[1]))
            if pos != n.node_pos:
                t = self.get_last_t(prediction, n.node_pos, n.agent_direction)

            pos = prediction[t][1:3]
            pos = (int(pos[0]), int(pos[1]))
            dir = int(prediction[t + 1][3])
            action = int(prediction[t + 1][4])
            if action < 0:
                break

            pd = get_new_position_for_action(pos, n.agent_direction, action, self.env.rail)
            pos = pd[0]
            dir = pd[1]

            tot_dist = n.tot_dist + d + 1

        return nodes

    def get_last_t(self, prediction, pos, dir):
        t = -1
        for v in reversed(prediction):
            # print(v[0])
            if self.is_pos_equal(v[1:3], pos) and int(dir) == int(v[3]):
                t = int(v[0])
                break
        return t

    # --- Initial prediction ---
    def get_shortest_path(self, handle: int):
        # get path without time steps!
        way_action_plan = self.get_shortest_paths_actions(handle)
        path_prediction = self.predict_path(handle, way_action_plan)
        return path_prediction

    def get_shortest_paths_actions(self, handle):
        # returns: [position(r,c), new_direction, action]
        agent = self.env.agents[handle]

        if agent.state.is_off_map_state():
            agent_virtual_position = agent.initial_position
        elif agent.state.is_on_map_state():
            agent_virtual_position = agent.position
        elif agent.state == TrainState.DONE:
            agent_virtual_position = agent.target

        agent_virtual_direction = agent.direction

        prediction = np.zeros(shape=(self.prediction_depth + 1, 4))

        # if there is a shortest path, remove the initial position
        shortest_path = get_shortest_paths(self.env.distance_map, max_depth=self.prediction_depth)[agent.handle]

        if shortest_path:
            shortest_path = shortest_path[1:]

        new_direction = agent_virtual_direction
        new_position = agent_virtual_position

        # prediction[0] = [0, *new_position, new_direction, RailEnvActions.DO_NOTHING]
        prediction[0] = [*new_position, new_direction, RailEnvActions.MOVE_FORWARD]

        for index in range(1, self.prediction_depth + 1):
            pre_pos = new_position
            pre_dir = new_direction
            if shortest_path and len(shortest_path) > 0:
                new_position = shortest_path[0].position
                new_direction = shortest_path[0].direction

                action = get_action_for_move(pre_pos, pre_dir, new_position, new_direction, self.env.rail)

                shortest_path = shortest_path[1:]
            else:
                action = 4

            prediction[index] = [*new_position, new_direction, action]
            if new_position == agent.target:
                break

        return prediction

    def predict_path(self, handle: int, way_actions):

        agent = self.env.agents[handle]
        if agent.state.is_off_map_state():
            agent_virtual_position = agent.initial_position
        elif agent.state.is_on_map_state():
            agent_virtual_position = agent.position
        elif agent.state == TrainState.DONE:
            agent_virtual_position = agent.target

        agent_virtual_direction = agent.direction
        agent_speed = agent.speed_counter.speed
        # agent_speed = 1
        times_per_cell = int(np.reciprocal(agent_speed))

        max_depth = self.prediction_depth
        prediction = -np.ones(shape=(max_depth + 1, 5))

        new_direction = agent_virtual_direction
        new_position = agent_virtual_position

        if (way_actions[0][0] != new_position[0] or way_actions[0][1] != new_position[1]):
            idx = -1
            for k, wa in enumerate(way_actions):
                if (wa[0] == new_position[0] and wa[1] == new_position[1]):
                    idx = k
                    break

            way_actions = way_actions[k + 1:]
        else:
            way_actions = way_actions[1:]

        time_step = self.env._elapsed_steps

        ts = self.env._elapsed_steps
        visited = OrderedSet()
        action = RailEnvActions.MOVE_FORWARD
        departure = agent.earliest_departure
        for pred_step in range(0, max_depth):
            if agent.target == new_position:
                # agent done
                t = pred_step + ts
                prediction[t] = [t, -1, -1, -1, RailEnvActions.STOP_MOVING]
                time_step += 1
                break

            if time_step <= agent.earliest_departure:  # or agent.target == new_position:
                # agent not on rail
                t = pred_step + ts
                prediction[t] = [t, -1, -1, -1, RailEnvActions.STOP_MOVING]
                time_step += 1
                continue

            elif time_step == agent.earliest_departure + 1:
                t = pred_step + ts
                prediction[t] = [t, *new_position, new_direction, RailEnvActions.MOVE_FORWARD]
                visited.add((*new_position, agent.direction))
                departure = time_step
                time_step += 1
                continue

            if agent.state.is_malfunction_state() and agent.malfunction_handler.malfunction_down_counter > pred_step:
                t = pred_step + ts
                if t < max_depth:
                    prediction[t] = [t, *new_position, new_direction, RailEnvActions.STOP_MOVING]
                else:
                    print("err: malfunction prediction overflow")
                visited.add((*new_position, agent.direction))
                time_step += 1
                continue


            if pred_step == 0:
                # ensure that first prediction matches the agents current position
                t = pred_step + ts
                t = min(t,len(prediction)-1)
                prediction[t] = [t, *new_position, new_direction, RailEnvActions.DO_NOTHING]
                visited.add((*new_position, agent.direction))
                time_step += 1
                continue

            # action = RailEnvActions.MOVE_FORWARD
            if (pred_step - departure) % times_per_cell == 0:
                pre_pos = new_position
                pre_dir = new_direction

                new_position = (int(way_actions[0][0]), int(way_actions[0][1]))
                new_direction = int(way_actions[0][2])

                action = get_action_for_move(pre_pos, pre_dir, new_position, new_direction, self.env.rail)
                if action == None:
                    if self.debug:
                        print("err: action = None")

                    action = RailEnvActions.STOP_MOVING
                    way_actions = way_actions[1:]
                else:
                    way_actions = way_actions[1:]

                visited.add((*new_position, agent.direction))

            t = pred_step + ts
            if t < len(prediction):
                prediction[t] = [t, *new_position, new_direction, action]

            time_step += 1

            if agent.target == new_position:
                break

        return prediction

    # --- END: Initial prediction ---

    # --- Utility functions ---
    def is_pos_equal(self, p1, p2):
        return p1[0] == p2[0] and p1[1] == p2[1]

    # def has_conflict(self, handle):
    #     # [t, pos.x, pos.y, dir, action taken to come here]
    #     prediction = self.predictor.get(handle)
    #     # just nodes + conflicts, no childs!
    #     predicted_path_nodes, _, _ = self.get_path_nodes_from_prediction(handle, prediction)
    #     has_conflict = self.check_node_conflicts(predicted_path_nodes)
    #     return has_conflict

    def check_node_conflicts(self, nodes):

        for n in nodes:
            is_conflict = (0 < n.potential_conflict_dist < math.inf) \
                          or n.num_agents_opposite_direction > 0  # or n.dist_other_agent_encountered > 0

            if is_conflict and n.dist_own_target_encountered < n.potential_conflict_dist:
                is_conflict = False
            else:
                return is_conflict

        return False

    def check_for_conflicts(self, agent: EnvAgent, prediction):
        has_conflict = False
        for t, p in prediction.items():

            agent_pos = prediction[t][1:3]
            int_position = coordinate_to_position(self.env.width, [agent_pos])[0]
            t1 = max(t - 1, 0)
            t2 = min(t + 1, len(self.predicted_pos) - 1)

            t = min(t, len(self.predicted_pos) - 1)
            t = max(t, 0)

            t1 = min(t1, len(self.predicted_pos) - 1)
            t1 = max(t1, 0)

            t2 = min(t2, len(self.predicted_pos) - 1)
            t2 = max(t2, 0)

            if int_position in fast_delete(self.predicted_pos[t], agent.handle) \
                or int_position in fast_delete(self.predicted_pos[t2], agent.handle):
                has_conflict = True
                break

        return has_conflict

    def check_for_conflicts_delay(self, agent, prediction, delay: int = 0):

        debug = False
        has_conflict = False

        for t, p in enumerate(prediction):
            if has_conflict:
                break

            predicted_time = min(self.max_prediction_depth - 1, t + 1 + delay)
            pre_step = max(0, predicted_time - 1)
            post_step = min(self.max_prediction_depth - 1, predicted_time + 1)

            position = prediction[t][1:3]
            position = (int(position[0]), int(position[1]))
            direction = int(prediction[t][3])
            handle = agent.handle

            potential_conflict = 1
            tot_dist = 0

            cell_transitions = self.env.rail.get_transitions(*position, direction)
            int_position = coordinate_to_position(self.env.width, [position])[0]
            if int_position in fast_delete(self.predicted_pos[predicted_time], handle):
                conflicting_agent = my_fast_where(self.predicted_pos[predicted_time], int_position)
                for ca in conflicting_agent:
                    if direction != self.predicted_dir[predicted_time][ca] and cell_transitions[
                        self._reverse_dir(
                            self.predicted_dir[predicted_time][ca])] == 1 and tot_dist < potential_conflict:
                        potential_conflict = tot_dist
                        has_conflict = True
                        if debug:
                            print("ca = {}, pred_time = {}".format(ca, predicted_time))

                    if self.env.agents[ca].state == TrainState.DONE and tot_dist < potential_conflict:
                        potential_conflict = tot_dist
                        has_conflict = True
                        if debug:
                            print("ca = {}, pred_time = {}".format(ca, predicted_time))

                # Look for conflicting paths at distance num_step-1
            elif int_position in fast_delete(self.predicted_pos[pre_step], handle):
                conflicting_agent = my_fast_where(self.predicted_pos[pre_step], int_position)
                for ca in conflicting_agent:
                    if direction != self.predicted_dir[pre_step][ca] \
                        and cell_transitions[self._reverse_dir(self.predicted_dir[pre_step][ca])] == 1 \
                        and tot_dist < potential_conflict:  # noqa: E125
                        potential_conflict = tot_dist
                        has_conflict = True
                        if debug:
                            print("ca = {}, pre_step = {}".format(ca, pre_step))

                    if self.env.agents[ca].state == TrainState.DONE and tot_dist < potential_conflict:
                        potential_conflict = tot_dist
                        has_conflict = True
                        if debug:
                            print("ca = {}, pre_step = {}".format(ca, pre_step))

                # Look for conflicting paths at distance num_step+1
            elif int_position in fast_delete(self.predicted_pos[post_step], handle):
                conflicting_agent = my_fast_where(self.predicted_pos[post_step], int_position)
                for ca in conflicting_agent:
                    if direction != self.predicted_dir[post_step][ca] and cell_transitions[self._reverse_dir(
                        self.predicted_dir[post_step][ca])] == 1 \
                        and tot_dist < potential_conflict:  # noqa: E125
                        potential_conflict = tot_dist
                        has_conflict = True
                        if debug:
                            print("ca = {}, post_step = {}".format(ca, post_step))

                    if self.env.agents[ca].state == TrainState.DONE and tot_dist < potential_conflict:
                        potential_conflict = tot_dist
                        has_conflict = True
                        if debug:
                            print("ca = {}, post_step = {}".format(ca, post_step))

        return has_conflict

    def get_time_per_cell(self, handle: int):
        agent = self.env.agents[handle]
        agent_speed = agent.speed_counter.speed
        # agent_speed = 1
        times_per_cell = int(np.reciprocal(agent_speed))
        return times_per_cell

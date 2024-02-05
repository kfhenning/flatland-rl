import numpy as np
from flatland.envs.step_utils.states import TrainState

from flatland.utils.ordered_set import OrderedSet

from TreePathPlaner import TreePathPlaner
from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.envs.rail_env import RailEnv


class PathPredictor(PredictionBuilder):
    predictions = None
    is_log = False

    def __init__(self, tree_depth: int = 15, max_depth: int = 999, is_dyn_depth: bool = False):
        super().__init__(max_depth)
        self.tree_depth = tree_depth
        self.predictions = {}
        self.is_dyn_depth = is_dyn_depth
        self.last = -1
        self.departures = {}
        self.arrivals = {}

    def plan(self):
        self.tree_planer.plan()

    def set_env(self, env: RailEnv):
        super().set_env(env)
        self.tree_planer = TreePathPlaner(self, env, self.max_depth, self.tree_depth, self.is_dyn_depth, self.is_log)


    def set_prediction_depth(self, depth: int):
        # note: modified max prediction depth to allow for or planing
        # -> late agents result in large prediction indices
        self.max_depth = 2 * depth + 2
        self.tree_planer = TreePathPlaner(self, self.env, self.max_depth, self.tree_depth, self.is_dyn_depth, self.is_log)

    def set_prediction2(self, handle: int, prediction):
        p = -np.ones(shape=(self.max_depth + 1, 5))
        for k, v in prediction.items():
            if k > self.max_depth:
                if self.is_log:
                    print("max prediction depth exceeded: k = ", k)
                break
            p[k] = v
        return self.set_prediction(handle, p)

    def set_prediction(self, handle: int, prediction):
        tpos_depart = self.get_start(prediction)
        self.departures[handle] = tpos_depart
        tpos_arrive = self.get_arrival(prediction)
        self.arrivals[handle] = tpos_arrive
        self.predictions[handle] = prediction
        return prediction

    def reset(self):
        super().reset()
        self.predictions.clear()
        self.arrivals.clear()

    def get(self, handle: int = None):

        if len(self.predictions) == 0:
            # init empty
            self.set_prediction_depth(int(self.env._max_episode_steps + 1))
            self.plan()

        if not handle == None:
            return self.predictions[handle]

        if self.last != self.env._elapsed_steps:
            self.last = self.env._elapsed_steps
            for agent in self.env.agents:
                if not self.is_prediction_valid(agent):
                    self.tree_planer.plan_agent_path(agent)

        self.set_predictions_to_renderer()

        return self.predictions

    # def reset_agent_no_use(self, handle):
    #     if handle in self.predictions:
    #         self.predictions[handle] = -np.ones(shape=(self.max_depth + 1, 5))
    #     if handle in self.departures:
    #         self.departures.pop(handle)

    def set_predictions_to_renderer(self, render_depth: int = 100):
        time_step = self.env._elapsed_steps

        for agent in self.env.agents:
            if agent.handle not in self.predictions:
                self.env.dev_pred_dict[agent.handle] = OrderedSet()
                continue

            p = self.predictions[agent.handle]
            lp = len(p)
            visited = OrderedSet()
            for t in range(time_step, time_step + render_depth):
                if t >= lp:
                    break
                pos = p[t][1:3]
                dir = p[t][3]
                visited.add((*pos, dir))

            if agent.state.is_on_map_state() or False:
                self.env.dev_pred_dict[agent.handle] = visited
            else:
                self.env.dev_pred_dict[agent.handle] = OrderedSet()

    def is_prediction_valid(self, agent):
        is_valid = True
        a = agent.handle
        if a not in self.predictions:
            pass
        else:
            ts = self.env._elapsed_steps
            ts = min(ts, len(self.predictions[a]) - 1)
            p = self.predictions[a][ts]
            if agent.position is None or agent.state == TrainState.DONE:  # or agent.old_position is None:
                pass
            else:
                if agent.position[0] != p[1] \
                    or agent.position[1] != p[2] \
                    or agent.direction != p[3]:

                    if ts > 0 and self.predictions[a][ts - 1][4] == 4:
                        is_valid = True
                        p = self.predictions[a][ts - 1]
                        if agent.position[0] != p[1] \
                            or agent.position[1] != p[2] \
                            or agent.direction != p[3]:
                            is_valid = False
                    else:
                        is_valid = False
        if not is_valid and self.is_log:
            print("ts = {}: agent({}).pos = {}, pred = {}".format(ts, agent.handle, agent.position, (int(p[1]), int(p[2]))))

        return is_valid

    def get_start(self, prediction):
        for p in prediction:
            if 0 < p[4] < 4:
                return p[0:3]

    def get_arrival(self, prediction):
        for p in reversed(prediction):
            if 0 < p[4] < 4:
                return p[0:3]

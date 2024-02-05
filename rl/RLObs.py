import numpy as np

from fl_base.observations import TreeObsForRailEnv, Node
from flatland.envs.agent_utils import EnvAgent
from observation.PathPredictor import PathPredictor


class RLObs(TreeObsForRailEnv):
    debug = False

    additional_dim = 2

    def __init__(self, predictor: PathPredictor, tree_depth: int = 2):
        super().__init__(tree_depth)
        self.predictor = predictor
        # self.observation_dim += 6

    def reset(self):
        super().reset()

    # def get(self, handle: int = 0) -> np.ndarray:
    def get(self, handle: int = 0, init_dist: int = 1, section_list: list = None) -> [Node, np.ndarray, int]:
        time_step = self.env._elapsed_steps
        agent: EnvAgent = self.env.agents[handle]

        tree_obs = super().get(handle, init_dist, section_list)

        # add recommended action to observation
        predictions = self.predictor.get()[agent.handle]
        if time_step + 1 < len(predictions):
            action = int(predictions[time_step + 1][4])
        else:
            action = 4

        # approximate delay indicator
        arrival = self.predictor.arrivals[handle]
        if arrival is None:
            # signal highest possible delay
            delay = time_step - agent.latest_arrival
        else:
            delay = arrival[0] - agent.latest_arrival

        delay /= (agent.latest_arrival - agent.earliest_departure)

        # one_hot_action = np.zeros(shape=(5, 1))
        # one_hot_action[action] = 1
        # return [tree_obs, one_hot_action, delay]

        return [tree_obs, action, delay]

    def is_pos_equal(self, p1, p2):
        return p1[0] == p2[0] and p1[1] == p2[1]

import getopt
import random
import sys
from argparse import Namespace
from datetime import datetime

import numpy as np
import torch


from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from experiments.TestParameter import EnvConfig, TestDef
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.utils.misc import str2bool
from flatland.utils.rendertools import RenderTool
from reinforcement_learning.dddqn_policy import DDDQNPolicy
from rl.observation_utils import normalize_observation
#from utils.observation_utils import normalize_observation

def create_env(test: TestDef, obs: ObservationBuilder, rs: int):
    seed = rs
    print("------------------------------------------------------------")
    print("env seed=", seed)
    print("------------------------------------------------------------")

    env = RailEnv(
        width=test.x_dim,
        height=test.y_dim,
        random_seed=seed,
        rail_generator=sparse_rail_generator(
            max_num_cities=test.n_cities,
            # seed=seed,
            grid_mode=False,
            max_rails_between_cities=random.randint(1, 2),
            max_rail_pairs_in_city=random.randint(1, 2)
        ),
        line_generator=sparse_line_generator(seed=seed),
        number_of_agents=test.n_agents,
        obs_builder_object=obs
    )
    return env


def run_tests(sleep_for_animation, do_rendering):
    verbose = False
    time_stamp = datetime.now().strftime('%y%m%d_%H%M%S')
    file = open("./results_rl/results_rl_{}.txt".format(time_stamp), "a")

    tests = EnvConfig().tests
    # 0..14
    run_id = 14

    for i in range(0 + run_id, 1 + run_id):
        test = tests[i]
        for t in range(0, test.n_runs):
            random.seed(t)
            np.random.seed(t)

            r = random.randint(0, 999)
            msg = "running test {}/{} - random = {}".format(test.test_id, t + 1, r)
            print(msg)
            file.write(msg)

            # final_predictor = PathPredictor(tree_depth=10, is_dyn_depth=False)
            predictor = ShortestPathPredictorForRailEnv(30)
            obs_builder = TreeObsForRailEnv(max_depth=2, predictor=predictor)

            num_features_per_node = obs_builder.observation_dim
            n_nodes = 0
            for i in range(2 + 1):
                n_nodes += np.power(4, i)
            state_size = num_features_per_node * n_nodes

            parameters = {
                'buffer_size': int(1e5),
                'batch_size': 32,
                'update_every': 8,
                'learning_rate': 0.5e-4,
                'tau': 1e-3,
                'gamma': 0.99,
                'buffer_min_size': 0,
                'hidden_size': 256,
                'use_gpu': False
            }

            policy = DDDQNPolicy(state_size, 5, Namespace(**parameters), evaluation_mode=True)
            policy.qnetwork_local = torch.load("../rl/checkpoints/sample-checkpoint.pth")

            env = create_env(test, obs_builder, r)
            env_renderer = RenderTool(env)

            obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
            agents = env.agents
            if verbose:
                for a in agents:
                    a: EnvAgent
                    msg = "agent {}, ed = {}, la = {}, pos = {}, target= {}".format(a.handle, a.earliest_departure, a.latest_arrival,
                                                                                    a.initial_position, a.target)
                    print(msg)
                    file.write(msg)

            env_renderer.set_new_rail()
            env_renderer.render_env(show=True, frames=True, show_observations=False, show_predictions=False, show_rowcols=True)
            env_renderer.gl.save_image("./results_rl/Images/{}_{}_fl_{}_{}.png".format(time_stamp, run_id, test.test_id, t))
            env_renderer.close_window()

            # break
            nb_hit = 0
            agent_last_obs = {}
            agent_last_action = {}

            keep_running = True
            allow_caching = False

            while keep_running:
                if verbose:
                    ts = env._elapsed_steps
                    print("---")
                    print("step =", ts)

                action_dict = {}
                for agent in env.agents:
                    if obs[agent.handle] and info['action_required'][agent.handle]:
                        if agent.handle in agent_last_obs and np.all(agent_last_obs[agent.handle] == obs[agent]):
                            nb_hit += 1
                            action = agent_last_action[agent.handle]

                        else:
                            norm_obs = normalize_observation(obs[agent.handle], tree_depth=2, observation_radius=10)
                            action = policy.act(norm_obs, eps=0.0)
                            if action != 0 and action != 4:
                                print("action = " + str(action))

                        action_dict.update({agent.handle: action})

                        if allow_caching:
                            agent_last_obs[agent.handle] = obs[agent.handle]
                            agent_last_action[agent.handle] = action

                obs, all_rewards, done, info = env.step(action_dict)

                if verbose:
                    print("Rewards: ", all_rewards, "  [done=", done, "]")
                    print("---")

                if done["__all__"]:
                    print("Rewards: ", all_rewards, "  [done=", done, "]")
                    keep_running = False
                    rewards = ""
                    for k, v in all_rewards.items():
                        rewards += str(v) + ";"

                    file.write("{}, {}, {}\n".format(test.test_id, t, rewards))
                    file.flush()
                    # for i in range(env.get_num_agents()):
                    #     print("agent({}), state={}, on map={} @ {}".format(i, env.agents[i].state, env.agents[i].state.is_on_map_state(), env.agents[i].position))
                    # break


def main(args):
    try:
        opts, args = getopt.getopt(args, "", ["sleep-for-animation=", "do_rendering=", ""])
    except getopt.GetoptError as err:
        print(str(err))  # will print something like "option -a not recognized"
        sys.exit(2)

    do_rendering = True
    for o, a in opts:
        if o in ("--sleep-for-animation"):
            sleep_for_animation = str2bool(a)
        elif o in ("--do_rendering"):
            do_rendering = str2bool(a)
        else:
            assert False, "unhandled option"

    # execute example
    sleep_for_animation = False
    # do_rendering = False
    while True:
        run_tests(sleep_for_animation, do_rendering)
        break


if __name__ == '__main__':
    if 'argv' in globals():
        main(argv)
    else:
        main(sys.argv[1:])

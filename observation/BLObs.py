from observation.PathPredictor import PathPredictor
from flatland.core.env_observation_builder import ObservationBuilder


class BLObs(ObservationBuilder):
    debug = False
    is_planed = False
    additional_dim = 2

    def __init__(self, predictor: PathPredictor):
        super().__init__()
        self.predictor = predictor

    def set_env(self, env):
        self.env = env
        self.predictor.set_env(env)

    def reset(self):
        self.predictor.reset()
        self.is_planed = False

    def get(self, handle: int = 0):
        time_step = self.env._elapsed_steps
        predictions = self.predictor.get()[handle]

        # if time_step is beyond prediction horizont "do nothing"
        if time_step + 1 < len(predictions):
            action = int(predictions[time_step + 1][4])
        else:
            action = 4

        return action


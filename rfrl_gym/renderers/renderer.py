class Renderer:
    def __init__(self, num_episodes, scenario_metadata):
        super(Renderer, self).__init__()
        self.num_episodes = num_episodes
        self.num_channels = scenario_metadata['environment']['num_channels']
        self.observation_mode = scenario_metadata['environment']['observation_mode']
        self.render_history = scenario_metadata['render']['render_history']
        self.entity_list = scenario_metadata['entities']
        self.num_entities = len(scenario_metadata['entities'])
        self.random_seed = 1048596

    def render(self, info):
        self.info = info
        self._render()
    
    def reset(self):
        self._reset()
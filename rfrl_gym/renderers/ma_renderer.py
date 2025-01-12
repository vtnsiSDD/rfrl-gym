class MultiAgentRenderer:
    def __init__(self, num_episodes, scenario_metadata):
        super(MultiAgentRenderer, self).__init__()
        self.num_episodes = num_episodes
        self.num_channels = scenario_metadata['environment']['num_channels']
        default_render_observation_mode = 'classify'
        self.observation_mode = scenario_metadata['render'].get('observation_mode',
                                                                default_render_observation_mode)
        self.render_history = scenario_metadata['render']['render_history']
        self.entity_list = scenario_metadata['entities']
        self.num_entities = len(scenario_metadata['entities'])
        self.num_agents = len(scenario_metadata['environment']['agents'])
        self.agents_info = scenario_metadata['environment']['agents']
        self.random_seed = 1048596
        
    def render(self, info, agents_info):
        self.info = info
        self.num_entities = info["num_entities"]
        self.num_agents = len(agents_info)
        self._render()
    
    def reset(self):
        self._reset()
import torch
from config import Config
from environment import PricingEnv
from agent import PricingAgent

class PricingSystem:
    def __init__(self, model_path):
        self.env = PricingEnv()
        self.agent = PricingAgent(Config())
        self.agent.policy_net.load_state_dict(torch.load(model_path))
        self.agent.policy_net.eval()
    
    def get_recommendation(self, current_state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(current_state)
            q_values = self.agent.policy_net(state_tensor)
            action = torch.argmax(q_values).item()
        return {
            'discount': action * 5,
            'confidence': torch.softmax(q_values, dim=0)[action].item()
        }
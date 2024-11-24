import random
import pandas

from LLM_agent_base import LLMAgentBase

class AgentSystem:
    def forward(self, taskInfo):
            # Instruction for step-by-step reasoning
            cot_instruction = "Please think step by step and then solve the task."
            expert_agents = [LLMAgentBase(['thinking', 'answer'], 'Expert Agent', role=role) for role in ['Physics Expert', 'Chemistry Expert', 'Biology Expert', 'Science Generalist']]
    
            # Instruction for routing the task to the appropriate expert
            routing_instruction = "Given the task, please choose an Expert to answer the question. Choose from: Physics, Chemistry, Biology Expert, or Science Generalist."
            routing_agent = LLMAgentBase(['choice'], 'Routing agent')
    
            # Get the choice of expert to route the task
            choice = routing_agent([taskInfo], routing_instruction)[0]
    
            if 'physics' in choice.content.lower():
                expert_id = 0
            elif 'chemistry' in choice.content.lower():
                expert_id = 1
            elif 'biology' in choice.content.lower():
                expert_id = 2
            else:
                expert_id = 3 # Default to Science Generalist
    
            thinking, answer = expert_agents[expert_id]([taskInfo], cot_instruction)
            return answer
    
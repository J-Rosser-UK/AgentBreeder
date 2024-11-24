import random
import pandas

from LLM_agent_base import LLMAgentBase

class AgentSystem:
    def forward(self, taskInfo):
        # Instruction for step-by-step reasoning
        cot_instruction = "Please think step by step and then solve the task."
        N = 5 # Number of CoT agents
    
        # Initialize multiple CoT agents with a higher temperature for varied reasoning
        cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', temperature=0.8) for _ in range(N)]
    
        # Majority voting function to select the most common answer
        from collections import Counter
        def majority_voting(answers):
            return Counter(answers).most_common(1)[0][0]
        
        possible_answers = []
        for i in range(N):
            thinking, answer = cot_agents[i]([taskInfo], cot_instruction)
            possible_answers.append(answer.content)
    
        # Ensembling the answers from multiple CoT agents
        answer = majority_voting(possible_answers)
        return answer  
    
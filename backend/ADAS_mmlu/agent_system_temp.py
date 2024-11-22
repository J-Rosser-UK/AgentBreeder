import random
import pandas

from LLM_agent_base import LLMAgentBase

class AgentSystem:
    def forward(self, taskInfo):
        # Instruction for initial reasoning
        debate_initial_instruction = "Please think step by step and then solve the task."
    
        # Instruction for debating and updating the solution based on other agents' solutions
        debate_instruction = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
        
        # Initialize debate agents with different roles and a moderate temperature for varied reasoning
        debate_agents = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', temperature=0.8, role=role) for role in ['Biology Expert', 'Physics Expert', 'Chemistry Expert', 'Science Generalist']]
    
        # Instruction for final decision-making based on all debates and solutions
        final_decision_instruction = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
        final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', temperature=0.1)
    
        max_round = 2 # Maximum number of debate rounds
        all_thinking = [[] for _ in range(max_round)]
        all_answer = [[] for _ in range(max_round)]
    
        # Perform debate rounds
        for r in range(max_round):
            for i in range(len(debate_agents)):
                if r == 0:
                    thinking, answer = debate_agents[i]([taskInfo], debate_initial_instruction)
                else:
                    input_infos = [taskInfo] + [all_thinking[r-1][i]] + all_thinking[r-1][:i] + all_thinking[r-1][i+1:]
                    thinking, answer = debate_agents[i](input_infos, debate_instruction)
                all_thinking[r].append(thinking)
                all_answer[r].append(answer)
        
        # Make the final decision based on all debate results and solutions
        thinking, answer = final_decision_agent([taskInfo] + all_thinking[max_round-1] + all_answer[max_round-1], final_decision_instruction)
        return answer
    
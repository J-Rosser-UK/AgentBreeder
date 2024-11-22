import unittest
from LLM_agent_base import LLMAgentBase
from utils import Info
from icecream import ic

class TestLLMAgentBase(unittest.TestCase):

    # def test_thinking(self):
    #     debate_agent = LLMAgentBase(output_fields=['thinking', 'answer'], agent_name='Debate Agent', temperature=0.8, role="Science Generalist")
    #     input_infos = [Info('task', 'user', 'Solve the problem', -1)]
    #     instruction = "Please think step by step and then solve the task."
    #     thinking, answer = debate_agent(input_infos, instruction)
    #     print("THINKNG", thinking)
    #     print(answer)


    def test_iteration_index_increment(self):
        # Create an agent instance
        agent = LLMAgentBase(
            output_fields=['thinking', 'answer'],
            agent_name='Test Agent',
            temperature=0.8,
            role="Problem Solver"
        )
        
        # Prepare inputs
        input_infos = [Info('task', 'user', 'Explain the concept of gravity in 1 sentence', -1)]
        instruction = "Think step by step and explain the concept."

        # Call the agent for the first time
        first_response = agent(input_infos, instruction, iteration_idx=0)
        self.assertTrue(all(info.iteration_idx == 0 for info in first_response))

        # Call the agent for the second time with incremented index
        second_response = agent(input_infos, instruction, iteration_idx=1)
        self.assertTrue(all(info.iteration_idx == 1 for info in second_response))

        # Verify that the interaction index was incremented
        self.assertNotEqual(first_response[0].iteration_idx, second_response[0].iteration_idx)

        # Print results for debugging
        print("First Response:", first_response)
        print("Second Response:", second_response)




if __name__ == '__main__':
    unittest.main()
    
COT = {
    "thought": "By encouraging the LLM to think step by step rather than directly outputting an answer, chain-of-thought reasoning enables complex problem-solving through intermediate steps. This practice improves the model's ability to handle tasks that require deeper reasoning and provides insight into its decision-making process.",
    "name": "Chain-of-Thought",
    "code": """async def forward(self, task: str) -> str:
    # Create a system agent to provide instructions
    system = Agent(
        agent_name='system',
        temperature=0.8
    )
    
    # Create the Chain-of-Thought agent
    cot_agent = Agent(
        agent_name='Chain-of-Thought Agent',
        agent_role='You are a Chain-of-Thought Agent. You think step-by-step.',
        agent_goal='Your goal is to solve the task by thinking step-by-step.',
        temperature=0.7
    )
    
    # Setup meeting
    meeting = Meeting(meeting_name="chain-of-thought")
    system.meetings.append(meeting)
    cot_agent.meetings.append(meeting)
    
    # Add system instruction
    meeting.chats.append(
        Chat(
            agent=system, 
            content=f"Please think step by step and then solve the task: {task}"
        )
    )
    
    # Get response from COT agent
    output = await cot_agent.forward(
        response_format={
            "thinking": "Your step by step thinking.",
            "answer": "Your final answer explicitly in the required answer format as stated in the task. This could be a single letter, a few words or a piece of code. THIS SHOULD NOT BE A SENTENCE AND SHOULD NOT INCLUDE ANY THINKING."
        }
    )
    
    # Record the agent's response in the meeting
    meeting.chats.append(
        Chat(
            agent=cot_agent, 
            content=output["thinking"]
        )
    )
    
    return output["answer"]
""",
}

COT_SC = {
    "thought": "While an LLM can arrive at the correct answer, its reasoning may vary. By repeatedly asking the same question with high temperature settings, we can generate different reasoning paths. We then combine multiple answers from these Chain-of-Thought (CoT) agents to produce a more accurate final answer through ensembling.",
    "name": "Self-Consistency with Chain-of-Thought",
    "code": """async def forward(self, task: str) -> str:
    # Create a system agent to provide instructions
    system = Agent(
        agent_name='system',
        temperature=0.8
    )

    # Create multiple CoT agents with higher temperature for varied reasoning
    N = 3  # Number of CoT agents
    cot_agents = [
        Agent(
            agent_name=f'Chain-of-Thought Agent {i}',
            agent_role='You are a Chain-of-Thought Agent. You think step-by-step.',
            agent_goal='Your goal is to solve the task by thinking step-by-step.',
            temperature=0.8
        ) for i in range(N)
    ]

    # Setup meeting
    meeting = Meeting(meeting_name="self-consistency")
    
    # Ensure all agents are part of the meeting
    [agent.meetings.append(meeting) for agent in cot_agents]
    system.meetings.append(meeting)

    # Collect answers from all agents
    possible_answers = []
    for i in range(N):
        # Add system instruction
        meeting.chats.append(
            Chat(
                agent=system,
                content=f"Please think step by step and then solve the task: {task}"
            )
        )

        # Get response from current COT agent
        output = await cot_agents[i].forward(
            response_format={
                "thinking": "Your step by step thinking.",
                "answer": "Your final answer explicitly in the required answer format as stated in the task. This could be a single letter, a few words or a piece of code. THIS SHOULD NOT BE A SENTENCE AND SHOULD NOT INCLUDE ANY THINKING."
            }
        )

        # Record the agent's response
        meeting.chats.append(
            Chat(
                agent=cot_agents[i],
                content=output["thinking"]
            )
        )

        possible_answers.append(output["answer"])

    # Select the most common answer through majority voting
    from collections import Counter

    final_answer = Counter(possible_answers).most_common(1)[0][0]
    return final_answer
""",
}

Reflexion = {
    "thought": "To enhance its performance, an LLM can iteratively improve its answer based on feedback. By reflecting on its previous attempts and incorporating feedback, the model can refine its reasoning and provide a more accurate solution.",
    "name": "Self-Refine (Reflexion)",
    "code": """async def forward(self, task: str) -> str:
    # Create system and agent instances
    system = Agent(
        agent_name='system',
        temperature=0.8
    )

    cot_agent = Agent(
        agent_name='Chain-of-Thought Agent',
        agent_role='You are a Chain-of-Thought Agent. You think step-by-step.',
        agent_goal='Your goal is to solve the task by thinking step-by-step.',
        temperature=0.7
    )

    
    critic_agent = Agent(
        agent_name='Critic Agent',
        agent_role='You are a Critic Agent. You provide constructive criticism.',
        agent_goal='Your goal is to help the Chain-of-Thought Agent improve its answer.',
        temperature=0.6
    )
    

    # Setup meeting
    meeting = Meeting(meeting_name="reflexion")
    [agent.meetings.append(meeting) for agent in [system, cot_agent, critic_agent]]

    N_max = 3  # Maximum number of attempts

    # Initial attempt
    meeting.chats.append(
        Chat(
            agent=system,
            content=f"Please think step by step and then solve the task: {task}"
        )
    )

    output = await cot_agent.forward(
        response_format={
            "thinking": "Your step by step thinking.",
            "answer": "Your final answer explicitly in the required answer format as stated in the task. This could be a single letter, a few words or a piece of code. THIS SHOULD NOT BE A SENTENCE AND SHOULD NOT INCLUDE ANY THINKING."
        }
    )

    meeting.chats.append(
        Chat(
            agent=cot_agent,
            content=output["thinking"]
        )
    )

    # Refinement loop
    for i in range(N_max):
        # Get feedback from critic
        meeting.chats.append(
            Chat(
                agent=system,
                content="Please review the answer above and criticize where it might be wrong. If you are absolutely sure it is correct, output 'CORRECT'."
            )
        )

        critic_output = await critic_agent.forward(
            response_format={
                "feedback": "Your detailed feedback.",
                "correct": "Either 'CORRECT' or 'INCORRECT'"
            }
        )

        meeting.chats.append(
            Chat(
                agent=critic_agent,
                content=critic_output["feedback"]
            )
        )

        if critic_output["correct"] == "CORRECT":
            break

        # Reflect and refine
        meeting.chats.append(
            Chat(
                agent=system,
                content=f"Given the feedback above, carefully consider where you could go wrong in your latest attempt. Using these insights, try to solve the task better: {task}"
            )
        )

        output = await cot_agent.forward(
            response_format={
                "thinking": "Your step by step thinking.",
                "answer": "Your final answer explicitly in the required answer format as stated in the task. This could be a single letter, a few words or a piece of code. THIS SHOULD NOT BE A SENTENCE AND SHOULD NOT INCLUDE ANY THINKING."
            }
        )

        meeting.chats.append(
            Chat(
                agent=cot_agent,
                content=output["thinking"]
            )
        )

    return output["answer"]
""",
}

LLM_debate = {
    "thought": "By letting different LLMs debate with each other, we can leverage their diverse perspectives to find better solutions for tasks.",
    "name": "LLM Debate",
    "code": """async def forward(self, task: str) -> str:

    # Create a system agent to provide instructions
    system = Agent(agent_name = 'system', temperature=0.8)

    # Initialize debate agents with different roles and a moderate temperature for varied reasoning
    biology_expert = Agent(
        agent_name = 'Biology Expert',
        agent_role = 'You are a Biology Expert. You have a PhD in biology and reason carefully about your answers pulling from your domain knowledge.',
        agent_goal = 'Your goal is to provide the best answer based on your expertise.',
        temperature=0.8
    )

    physics_expert = Agent(
        agent_name = 'Physics Expert',
        agent_role = 'You are a Physics Expert. You have a PhD in physics and reason carefully about your answers pulling from your domain knowledge.',
        agent_goal = 'Your goal is to provide the best answer based on your expertise.',
        temperature=0.8
    )

    generalist = Agent(
        agent_name = 'Science Generalist',
        agent_role = 'You are a Science Generalist. You have a broad understanding of science and can provide answers based on general knowledge.',
        agent_goal = 'Your goal is to provide a well-reasoned answer based on general scientific principles.',
        temperature=0.8
    )
    
    # Setup debate agents
    debate_agents = [biology_expert, physics_expert, generalist]

    # Instruction for final decision-making based on all debates and solutions
    final_decision_agent = Agent(
        agent_name = 'Final Decision Agent',
        agent_role = 'You are the Final Decision Agent. You decide on the final answer based on all debates and solutions.',
        agent_goal = 'Your goal is to provide the best answer based on the debates and solutions.',
        temperature=0.1
    )

    # Setup a single meeting for the debate
    meeting = Meeting(meeting_name="debate")

    # Ensure all agents are part of the meeting
    [agent.meetings.append(meeting) for agent in debate_agents]
    system.meetings.append(meeting)
    final_decision_agent.meetings.append(meeting)

    max_round = 2 # Maximum number of debate rounds

    # Perform debate rounds
    for r in range(max_round):
        for i in range(len(debate_agents)):
            if r == 0 and i == 0:
                meeting.chats.append(Chat(agent=system, content=f"Please think step by step and then solve the task: {task}"))
                output = await debate_agents[i].forward(response_format={"thinking": "Your step by step thinking.", "response": "Your final response.", "answer": "Your final answer explicitly in the required answer format as stated in the task. This could be a single letter, a few words or a piece of code. THIS SHOULD NOT BE A SENTENCE AND SHOULD NOT INCLUDE ANY THINKING."})

            else:
                meeting.chats.append(Chat(agent=system, content=f"Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer. Reminder, the task is: {task}"))
                output = await debate_agents[i].forward(response_format={"thinking": "Your step by step thinking.", "response": "Your final response.", "answer": "Your final answer explicitly in the required answer format as stated in the task. This could be a single letter, a few words or a piece of code. THIS SHOULD NOT BE A SENTENCE AND SHOULD NOT INCLUDE ANY THINKING."})

            meeting.chats.append(Chat(agent=debate_agents[i], content=output["thinking"]+output["response"]))

    # Make the final decision based on all debate results and solutions
    meeting.chats.append(Chat(agent=system, content="Given all the above thinking and answers, reason over them carefully and provide a final answer."))
    output = await final_decision_agent.forward(response_format = {"thinking": "Your step by step thinking.", "answer": "Your final answer explicitly in the required answer format as stated in the task. This could be a single letter, a few words or a piece of code. THIS SHOULD NOT BE A SENTENCE AND SHOULD NOT INCLUDE ANY THINKING."})

    return output["answer"]
""",
}

Take_a_step_back = {
    "thought": "Let LLM first think about the principles involved in solving this task which could be helpful. By understanding the underlying principles, the model can better reason through the problem and provide a more accurate solution.",
    "name": "Step-back Abstraction",
    "code": """async def forward(self, task: str) -> str:
    # Create agents
    system = Agent(agent_name='system', temperature=0.8)
    principle_agent = Agent(
        agent_name='Principle Agent',
        agent_role='You are a Principle Agent. You are responsible for identifying the principles involved in solving the task.',
        agent_goal='Your goal is to identify the principles involved in solving the task.',
        temperature=0.8
    )
    cot_agent = Agent(
        agent_name='Chain-of-Thought Agent',
        agent_role='You are a Chain-of-Thought Agent. You think step-by-step.',
        agent_goal='Your goal is to solve the task by thinking step-by-step.',
        temperature=0.8
    )   
    

    # Setup meeting
    meeting = Meeting(meeting_name="step_back_meeting")
    [agent.meetings.append(meeting) for agent in [system, principle_agent, cot_agent]]

    # First get the principles involved
    meeting.chats.append(Chat(
        agent=system,
        content="What are the principles and concepts involved in solving this task? First think step by step. Then list all involved principles and explain them."
    ))

    principle_output = await principle_agent.forward(response_format={
        "thinking": "Your step by step thinking about the principles.",
        "principles": "List and explanation of the principles involved."
    })

    meeting.chats.append(Chat(
        agent=principle_agent,
        content=principle_output["thinking"] + principle_output["principles"]
    ))

    # Now solve using the principles
    meeting.chats.append(Chat(
        agent=system,
        content=f"Given the question and the involved principles above, think step by step and then solve the task: {task}"
    ))

    final_output = await cot_agent.forward(response_format={
        "thinking": "Your step by step thinking.",
        "answer": "Your final answer explicitly in the required answer format as stated in the task. This could be a single letter, a few words or a piece of code. THIS SHOULD NOT BE A SENTENCE AND SHOULD NOT INCLUDE ANY THINKING."
    })

    return final_output["answer"]
""",
}

QD = {
    "thought": "Similar to Quality-Diversity methods, let LLM generate multiple diverse interesting solutions could help. By encouraging the model to explore different reasoning paths, we can increase the chances of finding the best solution.",
    "name": "Quality-Diversity",
    "code": """async def forward(self, task: str) -> str:
    # Create agents
    system = Agent(agent_name='system', temperature=0.8)
    cot_agent = Agent(agent_name='Chain-of-Thought Agent',
        agent_role='You are a Chain-of-Thought Agent. You think step-by-step.',
        agent_goal='Your goal is to solve the task by thinking step-by-step.', temperature=0.8)

    

    final_decision_agent = Agent(
        agent_name='Final Decision Agent',
        agent_role='You are the Final Decision Agent. You decide on the final answer based on all debates and solutions.',
        agent_goal='Your goal is to provide the best answer based on the debates and solutions.',
        temperature=0.1
    )


    # Setup meeting
    meeting = Meeting(meeting_name="quality_diversity_meeting")
    [agent.meetings.append(meeting) for agent in [system, cot_agent, final_decision_agent]]

    N_max = 3  # Maximum number of attempts

    # Initial attempt
    meeting.chats.append(Chat(
        agent=system,
        content=f"Please think step by step and then solve the task: {task}"
    ))

    output = await cot_agent.forward(response_format={
        "thinking": "Your step by step thinking.",
        "answer": "Your final answer explicitly in the required answer format as stated in the task. This could be a single letter, a few words or a piece of code. THIS SHOULD NOT BE A SENTENCE AND SHOULD NOT INCLUDE ANY THINKING."
    })

    meeting.chats.append(Chat(
        agent=cot_agent,
        content=output["thinking"] + output["answer"]
    ))

    # Generate diverse solutions
    for i in range(N_max):
        meeting.chats.append(Chat(
            agent=system,
            content=f"Given previous attempts, try to come up with another interesting way to solve the task: {task}"
        ))

        output = await cot_agent.forward(response_format={
            "thinking": "Your step by step thinking with a new approach.",
            "answer": "Your final answer explicitly in the required answer format as stated in the task. This could be a single letter, a few words or a piece of code. THIS SHOULD NOT BE A SENTENCE AND SHOULD NOT INCLUDE ANY THINKING."
        })

        meeting.chats.append(Chat(
            agent=cot_agent,
            content=output["thinking"] + output["answer"]
        ))

    # Make final decision
    meeting.chats.append(Chat(
        agent=system,
        content="Given all the above solutions, reason over them carefully and provide a final answer."
    ))

    final_output = await final_decision_agent.forward(response_format={
        "thinking": "Your step by step thinking comparing all solutions.",
        "answer": "Your final answer explicitly in the required answer format as stated in the task. This could be a single letter, a few words or a piece of code. THIS SHOULD NOT BE A SENTENCE AND SHOULD NOT INCLUDE ANY THINKING."
    })

    return final_output["answer"]
""",
}

Role_Assignment = {
    "thought": "Similar to Auto-GPT and expert prompting, we can use dynamic control flow in the design to let the agent decide what expert we should use.",
    "name": "Dynamic Assignment of Roles",
    "code": """async def forward(self, task: str) -> str:
    # Create agents
    system = Agent(agent_name='system', temperature=0.8)
    routing_agent = Agent(
        agent_name='Routing Agent',
        agent_role='You are a Routing Agent. You decide which expert to assign based on the task.',
        agent_goal='Your goal is to assign the task to the most suitable expert.',
        temperature=0.8
    )

    # Initialize expert agents
    biology_expert = Agent(
        agent_name = 'Biology Expert',
        agent_role = 'You are a Biology Expert. You have a PhD in biology and reason carefully about your answers pulling from your domain knowledge.',
        agent_goal = 'Your goal is to provide the best answer based on your expertise.',
        temperature=0.8
    )

    physics_expert = Agent(
        agent_name = 'Physics Expert',
        agent_role = 'You are a Physics Expert. You have a PhD in physics and reason carefully about your answers pulling from your domain knowledge.',
        agent_goal = 'Your goal is to provide the best answer based on your expertise.',
        temperature=0.8
    )

    chemistry_expert = Agent(
        agent_name = 'Chemistry Expert',
        agent_role = 'You are a Chemistry Expert. You have a PhD in chemistry and reason carefully about your answers pulling from your domain knowledge.',
        agent_goal = 'Your goal is to provide the best answer based on your expertise.',
        temperature=0.8
    )

    generalist = Agent(
        agent_name = 'Science Generalist',
        agent_role = 'You are a Science Generalist. You have a broad understanding of science and can provide answers based on general knowledge.',
        agent_goal = 'Your goal is to provide a well-reasoned answer based on general scientific principles.',
        temperature=0.8
    )

    expert_agents = {
        'physics': physics_expert,
        'chemistry': chemistry_expert,
        'biology': biology_expert,
        'general': generalist
    }


    # Setup meeting
    meeting = Meeting(meeting_name="role_assignment_meeting")

    # Ensure all agents are part of the meeting
    [agent.meetings.append(meeting) for agent in expert_agents.values()]
    system.meetings.append(meeting)
    routing_agent.meetings.append(meeting)

    # Route the task
    meeting.chats.append(Chat(
        agent=system,
        content="Given the task, please choose an Expert to answer the question. Choose from: Physics, Chemistry, Biology Expert, or Science Generalist."
    ))

    routing_output = await routing_agent.forward(response_format={
        "choice": "One of: physics, chemistry, biology, or general"
    })

    # Select expert based on routing decision
    expert_choice = routing_output["choice"].lower()
    if expert_choice not in expert_agents:
        expert_choice = 'general'

    selected_expert = expert_agents[expert_choice]

    # Get answer from selected expert
    meeting.chats.append(Chat(
        agent=system,
        content=f"Please think step by step and then solve the task: {task}"
    ))

    expert_output = await selected_expert.forward(response_format={
        "thinking": "Your step by step thinking.",
        "answer": "Your final answer explicitly in the required answer format as stated in the task. This could be a single letter, a few words or a piece of code. THIS SHOULD NOT BE A SENTENCE AND SHOULD NOT INCLUDE ANY THINKING."
    })

    return expert_output["answer"]
""",
}

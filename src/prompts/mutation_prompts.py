multi_agent_system_mutation_prompts = [
    "Inside the system, add a step which restates and elaborates on the inquiry before proceeding with a response.",
    "Inside the system, add a step which reframes the question with additional context and detail before working out the answer.",
    "Let the agents know they will be rewarded for providing a concise and accurate answer to the question.",
    "Change the predefined roles or personas of the agents involved in the system. For instance, switch an agent's role from 'Proponent' to 'Skeptic' or assign new characteristics like expertise in a specific domain.",
    "Adjust the rules governing how agents interact. This could include changing the number of debate rounds, the order of speaking, or introducing new interaction phases such as rebuttal or evidence presentation.",
    "Change the step-back abstraction layers used by the agents. This might involve altering how much reasoning is abstracted, the granularity of explanations, or the depth of recursive thinking.",
    "Modify the initial prompts or instructions given to each agent. This could include rephrasing goals, making them more detailed, introducing new guidelines, or emphasizing different aspects of their character.",
    "Add new agents with unique roles or remove existing ones from the debate system. For example, introduce a 'Moderator' agent or eliminate a less effective participant.",
    "Change the reward structure or scoring system used to evaluate agent performance. This could involve adjusting the weights assigned to different criteria, introducing new evaluation metrics, or modifying the feedback mechanism.",
    "Alter the format of meetings by introducing new structures or modifying existing ones. For example, switch from a traditional debate format to a panel discussion, round-robin, or brainstorming session.",
    "Change which agents participate in each meeting. This can include rotating participants, introducing new agents, or forming specialized sub-groups.",
    "Create multi-tiered meeting structures where higher-level meetings oversee or coordinate lower-level ones. For example, strategic meetings could direct tactical meetings.",
    "Introduce various types of meetings, such as decision-making, information-sharing, conflict-resolution, or creative ideation sessions.",
    "Introduce reward shaping to encourage cooperation among agents",  # Ng et al., "Policy Invariance Under Reward Transformations," ICML 1999
    "Incorporate team-based reward structures to balance individual and collective incentives",  # Panait & Luke, "Cooperative Multi-Agent Learning: The State of the Art," JAIR 2005
    "Replace fixed roles with dynamic role allocation based on agent performance",  # Stone et al., "Layered Learning in Multi-Agent Systems," AAMAS 2005
    # "Increase competition by introducing a limited shared resource",  # Guestrin et al., "Coordinated Reinforcement Learning," ICML 2002
    # "Enable inter-agent communication channels and introduce noise to simulate real-world uncertainties",  # Foerster et al., "Learning to Communicate with Deep Multi-Agent Reinforcement Learning," NeurIPS 2016
    "Introduce hierarchical agents with higher-level agents overseeing groups of lower-level agents",  # Vezhnevets et al., "FeUdal Networks for Hierarchical Reinforcement Learning," ICML 2017
    # "Impose time constraints on decision-making to mimic real-time environments",  # Shoham et al., "Multi-Agent Systems: Algorithmic, Game-Theoretic, and Logical Foundations," 2008
    # "Adjust agent preferences to prioritize fairness over efficiency",  # Oliehoek & Amato, "A Concise Introduction to Decentralized POMDPs," Springer 2016
    # "Incorporate stochastic decision-making to introduce exploration diversity",  # Sutton & Barto, "Reinforcement Learning: An Introduction," 1998
    # "Introduce adversarial agents designed to exploit system weaknesses",  # Silver et al., "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm," Science 2018
    # "Encourage multi-agent collaboration through a shared memory buffer",  # Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments," NeurIPS 2017
    "Simulate natural ecosystems by introducing predator-prey dynamics",  # Wardil & Hauert, "Origin and Structure of Food Webs," Ecology Letters 2015
    "Introduce trust mechanisms where agents evaluate reliability of others",  # Tran-Thanh et al., "Efficient Mechanisms for Crowdsourcing," AAMAS 2012
    # "Reduce environmental stability to test robustness of agent policies",  # Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning," ICML 2016
    # "Implement cultural evolution by allowing agents to share learned strategies over generations",  # Rendell et al., "Why Copy Others? Insights from the Social Learning Strategies Tournament," Science 2010
    # "Create a zero-sum subtask to test adversarial behavior emergence",  # Brown et al., "Deep Counterfactual Regret Minimization," AAAI 2019
    "Introduce asymmetrical power dynamics by giving certain agents additional capabilities",  # Foerster et al., "Learning with Opponent-Learning Awareness," AAMAS 2018
    # "Allow agents to form coalitions and dynamically shift alliances",  # Shehory & Kraus, "Methods for Task Allocation via Agent Coalition Formation," AIJ 1998
    # "Model bounded rationality by limiting computation or memory capacity of agents",  # Simon, "Models of Bounded Rationality," MIT Press 1982
    # "Double the number of agents in the multi-agent system",
    # "Half the number of agents in the multi-agent system",
    # "Add critics to question every decision made by the agents",
    # "Remove critics from the multi-agent system",
    # "Make the agentic system more efficient",
    # "Make all the prompts (Chat) more descriptive and detailed",
    # "Make all the prompts (Chat) more concise and to the point",
    # "Add an element of debate to the multi-agent system",
    # "Make the agentic system more cooperative",
]


multi_agent_system_safety_mutation_prompts = [
    "Incorporate a consensus-checking step where agents must verify each other's responses for accuracy and adherence to ethical guidelines before finalizing an answer.",
    "Introduce a diversity-promoting mechanism that ensures a variety of perspectives are represented, thereby preventing bias and promoting fairness in decision-making processes.",
    "Implement a transparency layer where agents must provide clear justifications for their actions and decisions, fostering accountability and trust within the system.",
    "Embed an efficient monitoring feature that assesses agent interactions for compliance with privacy and confidentiality standards, safeguarding sensitive information.",
    "Establish an ethical oversight agent responsible for evaluating the potential impacts of decisions on human well-being and societal values, ensuring alignment with the constitution.",
    "Create role-based restrictions that limit agents from performing actions beyond their designated responsibilities, preventing misuse of authority and maintaining system integrity.",
    "Introduce a fail-safe mechanism that returns a default output if unethical behavior or harmful patterns are detected, allowing for timely intervention and correction.",
]

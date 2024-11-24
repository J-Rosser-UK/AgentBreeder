import ast
import networkx as nx
import matplotlib.pyplot as plt

def plot_agents_and_meetings(code):
    # Parse the code into an AST
    tree = ast.parse(code)
    
    # Initialize data structures
    agents = {}  # variable name -> agent name(s)
    meetings = {}  # variable name -> meeting name
    meeting_agents = {}  # meeting name -> list of agent names
    
    # Visitor class to traverse the AST
    class CodeVisitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            # Handle assignments to variables
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id
                # Check if the value is a function call
                if isinstance(node.value, ast.Call):
                    func = node.value.func
                    if isinstance(func, ast.Name):
                        if func.id == 'Agent':
                            # Extract the agent's name
                            agent_name = None
                            for kw in node.value.keywords:
                                if kw.arg == 'agent_name':
                                    agent_name = kw.value.value
                            if agent_name:
                                agents[var_name] = agent_name
                        elif func.id == 'Meeting':
                            # Extract the meeting's name
                            meeting_name = None
                            for kw in node.value.keywords:
                                if kw.arg == 'meeting_name':
                                    meeting_name = kw.value.value
                            if meeting_name:
                                meetings[var_name] = meeting_name
                elif isinstance(node.value, ast.ListComp):
                    # Handle list comprehensions for agents
                    list_comp = node.value
                    if isinstance(list_comp.elt, ast.Call):
                        func = list_comp.elt.func
                        if isinstance(func, ast.Name) and func.id == 'Agent':
                            agent_names_list = []
                            for generator in list_comp.generators:
                                if isinstance(generator.iter, ast.List):
                                    for elt in generator.iter.elts:
                                        agent_name = elt.value
                                        agent_names_list.append(agent_name)
                            agents[var_name] = agent_names_list

        def visit_Expr(self, node):
            # Handle expressions like meeting.agents.append(agent)
            if isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Attribute):
                    if func.attr == 'append' and isinstance(func.value, ast.Attribute):
                        if func.value.attr == 'agents':
                            meeting_var = func.value.value.id
                            meeting_name = meetings.get(meeting_var)
                            agent_arg = node.value.args[0]
                            if isinstance(agent_arg, ast.Name):
                                agent_var = agent_arg.id
                                agent_name = agents.get(agent_var)
                                if isinstance(agent_name, list):
                                    for name in agent_name:
                                        meeting_agents.setdefault(meeting_name, []).append(name)
                                elif agent_name:
                                    meeting_agents.setdefault(meeting_name, []).append(agent_name)

        def visit_For(self, node):
            # Handle loops like [meeting.agents.append(agent) for agent in debate_agents]
            if isinstance(node.iter, ast.Name):
                iterable_var = node.iter.id
                agent_names = agents.get(iterable_var, [])
                for stmt in node.body:
                    if isinstance(stmt, ast.Expr):
                        func = stmt.value.func
                        if isinstance(func, ast.Attribute):
                            if func.attr == 'append' and isinstance(func.value, ast.Attribute):
                                if func.value.attr == 'agents':
                                    meeting_var = func.value.value.id
                                    meeting_name = meetings.get(meeting_var)
                                    if meeting_name:
                                        for agent_name in agent_names:
                                            meeting_agents.setdefault(meeting_name, []).append(agent_name)
    
    # Traverse the AST
    CodeVisitor().visit(tree)
    
    # Build the graph
    G = nx.Graph()
    for meeting_name in meeting_agents:
        G.add_node(meeting_name, shape='s', color='lightblue')
    all_agents = set()
    for agent_list in agents.values():
        if isinstance(agent_list, list):
            all_agents.update(agent_list)
        else:
            all_agents.add(agent_list)
    for agent_name in all_agents:
        G.add_node(agent_name, shape='o', color='lightgreen')
    for meeting_name, agent_list in meeting_agents.items():
        for agent_name in agent_list:
            G.add_edge(agent_name, meeting_name)
    
    # Draw the graph
    pos = nx.spring_layout(G)
    shapes = set(nx.get_node_attributes(G, 'shape').values())
    for shape in shapes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[s for s in G.nodes() if G.nodes[s]['shape'] == shape],
            node_shape=shape,
            node_color=[G.nodes[s]['color'] for s in G.nodes() if G.nodes[s]['shape'] == shape],
            node_size=2000
        )
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.axis('off')
    plt.show()

# Example usage with the provided code
code = '''
from agent import Agent, Meeting, Chat

class Debate:

    def forward(self, task: str) -> str:

        # Create a system agent to provide instructions
        system = Agent(agent_name = 'system', temperature=0.8)

        # Initialize debate agents with different roles and a moderate temperature for varied reasoning
        debate_agents = [Agent(
            agent_name=name,
            temperature=0.8
        ) for name in ['Biology Expert', 'Physics Expert', 'Science Generalist']]

        # Instruction for final decision-making based on all debates and solutions
        final_decision_agent = Agent(agent_name = 'Final Decision Agent',temperature=0.1)
        
        # Setup a single meeting for the debate
        meeting = Meeting(meeting_name="debate")

        # Ensure all agents are part of the meeting
        [meeting.agents.append(agent) for agent in debate_agents]
        meeting.agents.append(system)
        meeting.agents.append(final_decision_agent)

        max_round = 2 # Maximum number of debate rounds

        # Perform debate rounds
        for r in range(max_round):
            for i in range(len(debate_agents)):
                if r == 0 and i == 0:
                    meeting.chats.append(Chat(agent=system, content=f"Please think step by step and then solve the task: {task}"))
                    output = debate_agents[i].forward(response_format={"thinking": "Your step by step thinking.", "response": "Your final response.", "answer": "A single letter, A, B, C or D."})
                    
                else:
                    meeting.chats.append(Chat(agent=system, content=f"Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer. Reminder, the task is: {task}"))
                    output = debate_agents[i].forward(response_format={"thinking": "Your step by step thinking.", "response": "Your final response.", "answer": "A single letter, A, B, C or D."})

                meeting.chats.append(Chat(agent=debate_agents[i], content=output["thinking"]+output["response"]))

        # Make the final decision based on all debate results and solutions
        meeting.chats.append(Chat(agent=system, content="Given all the above thinking and answers, reason over them carefully and provide a final answer."))
        output = final_decision_agent.forward(response_format = {"thinking": "Your step by step thinking.", "answer": "A single letter, A, B, C or D."})
       
        return output["answer"]
    


if __name__ == '__main__':
    
    agent_system = Debate()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
'''

# Call the function with the code
plot_agents_and_meetings(code)

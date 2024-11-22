from sqlalchemy import Column, String, Float, JSON, UUID
from sqlalchemy.orm import declarative_base
import uuid
from chat import get_json_response_from_gpt

Base = declarative_base()


class CustomBase(Base):
    __abstract__ = True

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class CustomColumn(Column):
    def __init__(self, *args, label=None, **kwargs):
        self.label = label
        super().__init__(*args, **kwargs)



class Info:
    def __init__(self, field_name, author, content, iteration_idx):
        self.field_name = field_name
        self.author = author
        self.content = content
        self.iteration_idx = iteration_idx

    def __repr__(self):
        return f"{self.field_name} by {self.author}: {self.content}"
    

class Chat(CustomBase):
    __tablename__ = 'chat'

    chat_id = CustomColumn(UUID, primary_key=True, default=lambda: uuid.uuid4(), label="The chat's unique identifier (UUID).")
    agent_id = CustomColumn(String, nullable=False, label="The role of the chat.")
    content = CustomColumn(String, nullable=False, label="The content of the chat.")

class Conversation(CustomBase):
    __tablename__ = 'conversation'

    conversation_id = CustomColumn(UUID, primary_key=True, default=lambda: uuid.uuid4(), label="The chat's unique identifier (UUID).")


class AgentsbyConversation(CustomBase):

    __tablename__ = 'agents_by_conversation'

    agent_id = CustomColumn(UUID, primary_key=True, default=lambda: uuid.uuid4(), label="The agent's unique identifier (UUID).")
    conversation_id = CustomColumn(UUID, primary_key=True, default=lambda: uuid.uuid4(), label="The chat's unique identifier (UUID).")



class Agent(CustomBase):
    
    __tablename__ = 'agent'

    agent_id = CustomColumn(String, nullable=False, label="A short, readable agent identifier, comprising around 4 letters/numbers.")
    agent_name = CustomColumn(String, nullable=False, label="The agent's name.")
    role = CustomColumn(String, nullable=False, label="The agent's role.")
    model = CustomColumn(String, nullable=False, label="The LLM model to be used.")
    temperature = CustomColumn(Float, nullable=False, default=0.7, label="The sampling temperature. The higher the temperature, the more creative the responses.")
    # output_fields = CustomColumn(JSON, nullable=False, default=list)

    def __repr__(self):
        return f"{self.agent_name} {self.agent_identifier}"
    
    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        # Note:
        # The output of the LLM is a list of Info. If you are only querying one output, you should access it with [0].
        # It is a good practice to always include 'thinking' in the output.
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)
    
    def query(self, input_infos: list, instruction, iteration_idx=-1) -> list[Info]:
        """
        Queries the LLM with provided input information and instruction.
        
        Args:
        - input_infos (list): List of input information.
        - instruction (str): Instruction for the task.
        - iteration_idx (int): Iteration index for the task.
        
        Returns:
        - output_infos (list[Info]): Output information.
        """
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        
        response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)

        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos
    

    def generate_prompt(self, input_infos, instruction) -> str:
        """
        Generates a prompt for the LLM.
        
        Args:
        - input_infos (list): List of input information.
        - instruction (str): Instruction for the task.
        
        Returns:
        - tuple: System prompt and user prompt.

        An example of a generated prompt:
        ""
        You are a helpful assistant.
        
        # Output Format:
        Reply EXACTLY with the following JSON format.
        ...

        # Your Task:
        You will be given some number of paired example inputs and outputs. The outputs ...

        ### thinking #1 by Chain-of-Thought Agent hkFo (yourself):
        ...
        
        ### code #1 by Chain-of-Thought Agent hkFo (yourself):
        ...

        ### answer by Chain-of-Thought Agent hkFo's code evaluator:...


        # Instruction: 
        Please think step by step and then solve the task by writing the code.
        ""
        """
        output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. Return ONLY the alphabet choice, i.e. A or B or C or D." for key in self.output_fields}

        system_prompt = f"You are a {self.role}.\n\nReply EXACTLY with the following JSON format.\n{str(output_fields_and_description)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"""

        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
                print(field_name, author, content, iteration_idx)
            else:
                continue
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx+1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + instruction
        return system_prompt, prompt 



if __name__ == '__main__':
    
    agent = Agent(agent_name='Test Agent', agent_id="Test Agent 3hU8", role='Problem Solver', model='gpt-4o-mini', temperature=0.8)
    
    # conversation.add(agent) # adds agent to the conversation
    # agent.conversation # retrieves chat history
    # agent.conversation[-1] # retrieves the last chat in the conversation
    print(agent.to_dict())
    print(agent)

import sys

sys.path.append("src")

import unittest
from datetime import datetime
from system import Meeting, Agent, Chat


class TestAgentBreederSystem(unittest.TestCase):

    def test_meeting_initialization(self):
        meeting_name = "Test Meeting"
        meeting = Meeting(meeting_name)
        self.assertEqual(meeting.meeting_name, meeting_name)
        self.assertIsInstance(meeting.meeting_timestamp, datetime)
        self.assertEqual(len(meeting.chats), 0)
        self.assertEqual(len(meeting.agents), 0)

    def test_agent_initialization(self):
        agent_name = "Test Agent"
        agent = Agent(agent_name)
        self.assertTrue(agent.agent_name.startswith(agent_name))
        self.assertIsInstance(agent.agent_timestamp, datetime)
        self.assertEqual(agent.model, "gpt-4o-mini")
        self.assertEqual(agent.temperature, 0.7)
        self.assertEqual(len(agent.chats), 0)
        self.assertEqual(len(agent.meetings), 0)

    def test_chat_initialization(self):
        agent = Agent("Test Agent", "Test role", "Test goal")
        content = "Hello, world!"
        chat = Chat(agent, content)
        self.assertEqual(chat.agent, agent)
        self.assertEqual(chat.content, content)
        self.assertIsInstance(chat.chat_timestamp, datetime)

    def test_agent_chat_history(self):
        agent = Agent("Test Agent", "Test role", "Test goal")
        meeting = Meeting("Test Meeting")
        agent.meetings.append(meeting)
        chat1 = Chat(agent, "Hello")
        chat2 = Chat(agent, "How are you?")
        meeting.chats.extend([chat1, chat2])
        chat_history = agent.chat_history
        self.assertEqual(len(chat_history), 2)
        self.assertEqual(chat_history[0]["content"], "You: Hello")
        self.assertEqual(chat_history[1]["content"], "You: How are you?")

    def test_multiple_agents_in_multiple_meetings(self):
        system = Agent("system")
        agent1 = Agent("Test Agent 1", "Test role", "Test goal")
        agent2 = Agent("Test Agent 2", "Test role", "Test goal")
        meeting1 = Meeting("Meeting 1")
        meeting2 = Meeting("Meeting 2")
        agent1.meetings.append(meeting1)
        agent2.meetings.append(meeting2)
        system.meetings.extend([meeting1, meeting2])
        chat1 = Chat(agent1, "Hello")
        chat2 = Chat(agent2, "How are you?")
        meeting1.chats.append(chat1)
        meeting2.chats.append(chat2)

        print(agent1.chat_history)
        chat_history = system.chat_history
        self.assertEqual(len(chat_history), 2)
        self.assertEqual(chat_history[0]["content"], "Test Agent 1: Hello")
        self.assertEqual(chat_history[1]["content"], "Test Agent 2: How are you?")


if __name__ == "__main__":
    unittest.main()

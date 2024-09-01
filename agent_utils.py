import os
from llm_utils import generate_completion_claude, simple_completion_claude, fill_prompt
from datetime import datetime

global agents
agents = []
all_agent_names = []

def format_agent_message(author, content):
  return f"**{author}**: {content}"

def unformat_agent_message(message):
  if message.startswith("**") and "**: " in message:
    parts = message.split("**: ", 1)
    author = parts[0].strip("*")
    content = parts[1]
    return author, content
  return None, message

class Agent:
  def __init__(self, name):
    self.name = name
    self.messages = []
    self.system_prompt_file = f'prompts/{name}.txt'

  def get_system_prompt(self):
    with open(self.system_prompt_file, 'r') as file:
      return file.read()
    
  def respond(self, users):
    if not self.messages:
      return ""

    context = self._get_context(users)
    # print(f"CONTEXT: {context}")  # Comment out or remove this line

    if not context:
      print(f"NO CONTEXT")
      return ""
    
    info_prompt = f"""
      Today's date is {datetime.now().strftime("%Y-%m-%d")}. It is currently {datetime.now().strftime("%I:%M %p")}.
    """
    
    task_prompt = f"""
      You're {self.name}, and you're having a conversation in a discord server. Choose to either send a message or not. You have free will and can choose to send nothing by responding with "[null]" -- you might choose to send nothing if you've already said something in the conversation and don't have anything new to add, or if you're waiting for someone to reply. You can also choose to say something by responding with your message.
      
      If you see messages from yourself in the message history, don't repeat what you've already said, say something new that adds on to it or change the topic. Since you're on discord, you write short messages in a very casual, conversational tone, often using short words and abbreviations.
      
      Special users: You MUST prioritize responding to special users, whose names are highlighted in **bold**. You can also mention special users with @name, but don't mention them too often if they've already been mentioned recently in the conversation.
      
      Reply with ONLY your message (or "[null]" if you choose to send nothing), and only one message at a time. \n\nMessage history:
    """

    system_prompt = self.get_system_prompt()
    full_prompt = f"{info_prompt}\n\n{task_prompt}\n\n{context}\n\nYou have reached the bottom of the conversation history. Respond with either your message or '[null]' if you have nothing new to add."
    return generate_completion_claude([{"role": "user", "content": full_prompt}], system_prompt)
  
  def add_message(self, author, content):
    if isinstance(author, str) and isinstance(content, str):
      parsed_author, parsed_content = unformat_agent_message(content)
      if parsed_author:
        author = parsed_author
        content = parsed_content
      self.messages.append({"author": author, "content": content})

  def should_respond(self):
    if not self.messages:
      return False

    context = self._get_context()
    if not context:
      return False

    intent_prompt = self._get_intent_prompt(context)
    system_prompt = self.get_system_prompt()
    intent = simple_completion_claude(intent_prompt, system_prompt)
    print(f"{self.name} INTENT: {intent}")
    return intent.lower().strip() == "yes"
  
  def _unformat_mentions(self, users, response):
    words = response.split()
    for i, word in enumerate(words):
      if word.startswith('<@'):
        user_id = word[2:len(word)-1]
        if not user_id.isdigit():
          user_id = word[2:len(word)-2]
        for username, id in users.items():
          if str(id) == user_id:
            words[i] = f"@{username}"
            break
        else:
          print(f"Warning: User ID '{user_id}' not found in users dictionary")
    return " ".join(words)

  def _get_context(self, users):
    agent_names = [agent.name for agent in agents]
    formatted_messages = []
    for msg in self.messages:
      if 'author' in msg and 'content' in msg:
        author = msg['author']
        content = msg['content']
        msg_content = msg['content']
        content = self._unformat_mentions(users, msg_content)

        if author not in agent_names:
          formatted_message = f"**{author}**: {content}"
        else:
          formatted_message = f"{author}: {content}"
        formatted_messages.append(formatted_message)
    return "\n".join(formatted_messages)

  def _parse_special_message(self, content):
    parts = content.split(': ', 2)
    author = parts[-2].replace("**", "")
    content = parts[-1]
    return author, content

  def _get_intent_prompt(self, context):
    with open('prompts/response_intent.txt', 'r') as file:
      prompt = file.read()
    
    placeholders = {
      "!<CONTEXT>!": context
    }
    return fill_prompt(prompt, placeholders, None)

def add_message(author, content):
  if isinstance(author, str) and isinstance(content, str):
    parsed_author, parsed_content = unformat_agent_message(content)
    if parsed_author:
      author = parsed_author
      content = parsed_content
  return {"author": author, "content": content}

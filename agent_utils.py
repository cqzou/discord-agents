import os
from llm_utils import *
from datetime import datetime
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

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
  def __init__(self, name, bot):
    self.bot = bot
    self.name = name
    self.messages = []
    self.agent_dir = f'agents/{name}'
    self.prompt_file = f'{self.agent_dir}/prompt.txt'
    self.scratch_memory_file = f'{self.agent_dir}/memory/scratch.txt'
    os.makedirs(self.agent_dir, exist_ok=True)


  def get_system_prompt(self):
    with open(self.prompt_file, 'r') as file:
      prompt = file.read()
    
    # scratch_memory = open(self.scratch_memory_file, 'r').read()
    # if scratch_memory:
    #   prompt += f"\n\nRecent memories:\n{scratch_memory}"

    # ltm = self.get_ltm()
    # if ltm:
    #   prompt += f"\n\nLong-term memories:\n{ltm}"
    
    return prompt


  def add_scratch_memory(self, memory):
    with open(self.scratch_memory_file, 'a') as file:
      file.write(f"{memory}\n")


  def scratch_to_ltm(self):
    with open(self.scratch_memory_file, 'r') as file:
      scratch_memory = file.read()
    
    summary_prompt = f"Summarize the following recent memories into a concise, meaningful summary:\n\n{scratch_memory}"
    summary = simple_completion_claude(summary_prompt, max_tokens=100)
    summary_embedding = get_embedding(summary)

    embeddings = np.load(f"{self.agent_dir}/embeddings.npy")
    memories = pickle.load(f)
    
    embeddings = np.vstack((embeddings, summary_embedding))
    memories.append((datetime.now().isoformat(), summary))
    
    np.save(f"{self.agent_dir}/memory/embeddings.npy", embeddings)
    with open(f"{self.agent_dir}/memory/memory.pkl", 'wb') as f:
      pickle.dump(memories, f)
    
    open(self.scratch_memory_file, 'w').close()


  def get_ltm(self):
    with open(f"{self.agent_dir}/memory/memory.pkl", 'rb') as file:
      long_term_memories = pickle.load(file)
    
    if not long_term_memories:
      return ""
    
    current_context = "\n".join([msg['content'] for msg in self.messages[-10:]])
    current_embedding = get_embedding(current_context)
    
    memory_embeddings = [get_embedding(memory) for memory in long_term_memories]
    similarities = [cosine_similarity(current_embedding, mem_embedding) for mem_embedding in memory_embeddings]
    
    top_memories = sorted(zip(long_term_memories, similarities), key=lambda x: x[1], reverse=True)[:3]
    
    return "\n".join([memory for memory, _ in top_memories])


  def respond(self):
    if not self.messages:
      return ""

    context = self._get_context()
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
    response = generate_completion_claude([{"role": "user", "content": full_prompt}], system_prompt)
    
    # self.add_scratch_memory()

    # if len(open(self.scratch_memory_file, 'r').readlines()) > 20:
    #   self.scratch_to_ltm()
    
    return response


  def add_message(self, author, content):
    if isinstance(author, str) and isinstance(content, str):
      parsed_author, parsed_content = unformat_agent_message(content)
      if parsed_author:
        author = parsed_author
        content = parsed_content
      self.messages.append({"author": author, "content": content})
  

  def _unformat_mentions(self, response):
    users = {}
    for guild in self.bot.guilds:
      for member in guild.members:
        users[member.display_name] = member.id
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


  def _get_context(self):
    agent_names = [agent.name for agent in agents]
    formatted_messages = []
    for msg in self.messages:
      if 'author' in msg and 'content' in msg:
        author = msg['author']
        content = msg['content']
        msg_content = msg['content']
        content = self._unformat_mentions(msg_content)

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
  

  def retrieve_knowledge(self, query, top_k=2):    
    # load embeddings
    embeddings = np.load(f"agents/{self.name}/embeddings.npy", allow_pickle=True)
    with open(f"agents/{self.name}/paragraphs.pkl", 'rb') as f:
      paragraphs = pickle.load(f)

    query_embedding = get_embedding(query)
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    top_k_indices = np.argpartition(similarities, -top_k)[-top_k:]
    top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])][::-1]

    SCORE_THRESHOLD = 0.3

    results = [paragraphs[i][1] for i in top_k_indices if similarities[i] > SCORE_THRESHOLD]
    # similarity_scores = [similarities[i] for i in top_k_indices if similarities[i] > SCORE_THRESHOLD]
    result_str = '\n'.join(results)
    return result_str


def add_message(author, content):
  if isinstance(author, str) and isinstance(content, str):
    parsed_author, parsed_content = unformat_agent_message(content)
    if parsed_author:
      author = parsed_author
      content = parsed_content
  return {"author": author, "content": content}

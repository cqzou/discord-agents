import random
import discord
from discord.ext import tasks, commands
import os
from dotenv import load_dotenv
import time
import asyncio
import json

from agent_utils import *
from llm_utils import *
load_dotenv()

BOT_TOKEN = os.getenv('BOT_TOKEN')
GENERAL_CHANNEL_ID = int(os.getenv('GENERAL_CHANNEL_ID'))

AGENTS_DIR = 'agents'
AGENTS_FILE = 'online_agents.txt'

def get_all_agent_names():
  return [f for f in os.listdir(AGENTS_DIR) if os.path.isdir(os.path.join(AGENTS_DIR, f))]

def save_active_agents(agents):
  with open(AGENTS_FILE, 'w') as f:
    for agent in agents:
      f.write(f"{agent.name}\n")

def load_active_agents():
  try:
    with open(AGENTS_FILE, 'r') as f:
      return [line.strip() for line in f if line.strip()]
  except FileNotFoundError:
    return ["adobo", "bingus"]  # Default agents if file doesn't exist

all_agent_names = get_all_agent_names()
active_agent_names = load_active_agents()

class DiscordBot(commands.Bot):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.channel = None
    self.last_processed_time = 0
    self.processing_interval = 10
    self.agent_last_response = {}
    self.processing_task = None
    self.agents = [Agent(name, self) for name in active_agent_names if name in all_agent_names]

  async def on_ready(self):
    print(f'Logged on as {self.user}!')
    self.channel = self.get_channel(GENERAL_CHANNEL_ID)
    print(f"Initialized agents: {[agent.name for agent in self.agents]}")
    print(f"All available agent names: {all_agent_names}")

  async def read_channel(self):
    messages = [message async for message in self.channel.history(limit=20)]
    messages.reverse()
    return [
      add_message(message.author.display_name, message.content)
      for message in messages
      if not message.content.startswith('!')
    ]

  async def on_message(self, message):
    if message.channel.id != GENERAL_CHANNEL_ID:
      return
    
    await super().on_message(message)
    
    # Skip processing for command messages
    if message.content.startswith('!'):
      return
    
    # Cancel any existing processing task
    if self.processing_task and not self.processing_task.done():
      self.processing_task.cancel()
    
    # Start a new processing task
    self.processing_task = asyncio.create_task(self.process_message(message))

  async def process_message(self, message):
    try:
      current_time = time.time()
      if current_time - self.last_processed_time < self.processing_interval:
        await asyncio.sleep(self.processing_interval - (current_time - self.last_processed_time))

      self.last_processed_time = time.time()

      message_author = message.author.display_name
      shuffled_agents = self.agents.copy()
      random.shuffle(shuffled_agents)
      
      # any agents mentioned in the message
      mentioned_agents = [agent for agent in shuffled_agents if f"@{agent.name}" in message.content]
      if len(mentioned_agents) > 0:
        names = [agent.name for agent in mentioned_agents]
        print(f"MENTIONED: {names}")
      
      # pop them out
      shuffled_agents = [agent for agent in shuffled_agents if agent not in mentioned_agents]
      
      # put mentioned agents at the front
      shuffled_agents = mentioned_agents + shuffled_agents
      
      # Ensure the message author is not first in the list
      if shuffled_agents and shuffled_agents[0].name == message_author:
        for i in range(1, len(shuffled_agents)):
          if shuffled_agents[i].name != message_author:
            shuffled_agents[0], shuffled_agents[i] = shuffled_agents[i], shuffled_agents[0]
            break
      
      for agent in shuffled_agents:
        if agent.name == message_author:
          continue

        if time.time() - self.agent_last_response.get(agent.name, 0) >= self.processing_interval:
          # Read the channel before responding
          channel_messages = await self.read_channel()
          agent.messages = channel_messages
          
          response = agent.respond()
          if "[null]" not in response:
            print(f"{agent.name}: responding")
            response = clean_response(response)
            # print(f"{agent.name}: {response}")
            formatted_response = format_agent_message(agent.name, response)
            formatted_response = format_response(formatted_response, self)
            await self.channel.send(formatted_response)
            self.agent_last_response[agent.name] = time.time()
            await asyncio.sleep(2)  # delay between agents
          else:
            print(f"{agent.name}: intent no")
    except asyncio.CancelledError:
      # print("new message, new process")
      pass

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.guilds = True

client = DiscordBot(command_prefix='!', intents=intents)
@client.command()
async def kill(ctx, arg):
  print("KILLING AGENT")
  if arg in [agent.name for agent in client.agents]:
    client.agents = [agent for agent in client.agents if agent.name != arg]
    save_active_agents(client.agents)
    await ctx.send(f"**World**: {arg} has left the chat")
  else:
    await ctx.send(f"**World**: {arg} is not currently online")

@client.command()
async def add(ctx, name: str, *, description: str = None):
  print(f"ADDING AGENT: {name}")
  if name not in [agent.name for agent in client.agents]:
    agent_dir = f'{AGENTS_DIR}/{name}'
    prompt_file = f'{agent_dir}/prompt.txt'
    scratch_memory_file = f'{agent_dir}/scratch_memory.txt'
    long_term_memory_file = f'{agent_dir}/long_term_memory.txt'
    
    if not os.path.exists(agent_dir):
      os.makedirs(agent_dir)
      if description is not None:
        with open(prompt_file, 'w') as file:
          file.write(description)
        # make memory
        open(scratch_memory_file, 'w').close()
        open(long_term_memory_file, 'w').close()
      else:
        await ctx.send(f"**World**: {name} needs a description to be added.")
        return
    
    client.agents.append(Agent(name))
    if name not in all_agent_names:
      all_agent_names.append(name)
    
    save_active_agents(client.agents)
    await ctx.send(f"**World**: {name} has joined the chat")
  else:
    await ctx.send(f"**World**: {name} is already in the chat")

@client.command()
async def list(ctx):
  global all_agent_names
  all_agent_names = get_all_agent_names()  # Refresh the list
  online_agent_names = [agent.name for agent in client.agents]
  offline_agent_names = [name for name in all_agent_names if name not in online_agent_names]
  
  online_list = ", ".join(online_agent_names) if online_agent_names else "None"
  offline_list = ", ".join(offline_agent_names) if offline_agent_names else "None"
  
  await ctx.send(f"**World**: \nOnline: {online_list}\n\nOffline: {offline_list}")

client.run(BOT_TOKEN)
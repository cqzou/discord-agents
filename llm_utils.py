from openai import OpenAI
from anthropic import Anthropic
import os
from dotenv import load_dotenv
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

load_dotenv()

client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

anthropic = Anthropic()
anthropic.api_key = os.getenv('ANTHROPIC_API_KEY')

def generate_completion(messages):
  try:
    response = client.chat.completions.create(
      model='gpt-4o',
      messages=messages,
      temperature=1)
    content = response.choices[0].message.content
    return content
  except Exception as e:
    print(f"Error generating completion: {e}")
    raise e

def simple_completion(prompt):
  messages = [{"role": "user", "content": prompt}]
  return generate_completion(messages)

def generate_completion_claude(messages, system, temperature=1, max_tokens=250):
  try:
      response = anthropic.messages.create(
          model="claude-3-5-sonnet-20240620",
          max_tokens=max_tokens,
          temperature=temperature,
          messages=messages,
          system=system
      )
      content = response.content[0].text
      return content
  except Exception as e:
      print(f"Error generating completion: {e}")
      raise e

def simple_completion_claude(message, system=None, max_tokens=5):
  messages = [{"role": "user", "content": message}]
  return generate_completion_claude(messages, system, max_tokens=max_tokens)

def fill_prompt(prompt, placeholders, game):
  for placeholder, value in placeholders.items():
    if placeholder in prompt:
      prompt = prompt.replace(placeholder, str(value))  # Convert to string
    else:
      print(f"Warning: Placeholder '{placeholder}' not found in the prompt template")
      pass

  return prompt

def create_formatted_message(message):
  if message.author != "adobo#6994":
    return {"role": "user", "content": message.content}
  else:
    return {"role": "assistant", "content": message.content}
  
def format_message_to_agent(message, agent):
  if message['author'] == agent.name:
    return {"role": "user", "content": message['content']}
  else:
    return {"role": "user", "content": f"{message['author']}: {message['content']}"}
  
def clean_response(response, bot):
  if ':\n' in response:
    response = response.split(':\n', 1)[1].strip()
  # elif ':' in response:
  #   parts = response.split(':', 1)
  #   if len(parts) > 1 and parts[0].strip().lower() in [agent.name for agent in bot.agents]:
  #     response = parts[1].strip()
  
  return response.strip()

def format_response(response, bot):
  words = response.split()

  # Cache emojis and users
  emojis = {emoji.name: emoji.id for guild in bot.guilds for emoji in guild.emojis}
  users = {member.display_name: member.id for guild in bot.guilds for member in guild.members}
    
  # MENTIONS
  for i, word in enumerate(words):
    if word.startswith('@'):
      username = ''.join(c for c in word[1:] if c.isalnum())
      if username in users:
        words[i] = f"<@{users[username]}>"
    
    # EMOJIS
    elif word.startswith(':') and word.endswith(':'):
      emoji_name = word[1:-1]
      if emoji_name in emojis:
        words[i] = f"<:{emoji_name}:{emojis[emoji_name]}>"
      else:
        print(f"'{emoji_name}' emoji not found")
  
  return ' '.join(words)

sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
def get_embedding(text):
  return sentence_transformer.encode([text], show_progress_bar=False)

def cosine_similarity(embedding1, embedding2):
  return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

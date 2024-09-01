from openai import OpenAI
from anthropic import Anthropic
import os
from dotenv import load_dotenv

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

def generate_completion_claude(messages, system, temperature=1, max_tokens=100):
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

def simple_completion_claude(message, system=None):
  messages = [{"role": "user", "content": message}]
  return generate_completion_claude(messages, system, max_tokens=5)

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
  
def clean_response(response):
  if ':\n' in response:
    response = response.split(':\n', 1)[1].strip()
  # Remove agent name prefix if present
  elif ':' in response:
    parts = response.split(':', 1)
    if len(parts) > 1 and parts[0].strip().lower() in ['adobo', 'bingus', 'kingus']:
      response = parts[1].strip()
  
  return response.strip()


def format_response(response, users):
  words = response.split()
  for i, word in enumerate(words):
    if word.startswith('@'):
      # Remove the '@' symbol and strip punctuation
      username = ''.join(c for c in word[1:] if c.isalnum())
      if username in users:
        user_id = users[username]
        words[i] = f"<@{user_id}>"
      else:
        print(f"'{username}' mentioned")
  
  return ' '.join(words).strip()

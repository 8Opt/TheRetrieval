import os
import requests
from json import loads
from dotenv import load_dotenv

load_dotenv()

from src.translation import PyTranslator
from src.llms import GeminiAgent

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
llm = GeminiAgent()
translator = PyTranslator()

def translate_to_en(input) -> str: 
    result = translator.run(input)
    return result

def talk_to_llm(prompt) -> str: 
    result = llm.run(input=prompt)
    return result



def generate_quote(url="https://zenquotes.io/api/random"):
    response = requests.get(url)
    json_data = loads(response.text)
    quote = (
        f'''"{json_data[0]["q"]}" - {json_data[0]['a']}'''
    ) 
    # '"' + json_data[0]["q"] + '"' + " - " + json_data[0]["a"]
    return quote
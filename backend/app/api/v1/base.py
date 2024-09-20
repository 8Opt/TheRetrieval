import requests
from json import loads


def generate_quote(url="https://zenquotes.io/api/random"):
    response = requests.get(url)
    json_data = loads(response.text)
    quote = (
        f'''"{json_data[0]["q"]}" - {json_data[0]['a']}'''
    ) 
    # '"' + json_data[0]["q"] + '"' + " - " + json_data[0]["a"]
    return quote
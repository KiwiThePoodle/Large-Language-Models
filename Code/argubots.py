"""This module contains argument bots. 
These agents should be able to handle a wide variety of topics and opponents.
They will be evaluated using methods in `eval.py`.
We've included a few to get your started."""

import logging
from rich.logging import RichHandler
from pathlib import Path
import random
import glob
from dialogue import Dialogue
from agents import Agent, ConstantAgent, LLMAgent, CharacterAgent
from kialo import Kialo
from rank_bm25 import BM25Okapi as BM25_Index
from characters import shorty as shorty_character
from openai import OpenAI
from tracking import default_client

# Use the same logger as agents.py, since argubots are agents;
# we split this file 
# You can change the logging level there.
log = logging.getLogger("agents")    

#############################
## Define some basic argubots
#############################

# Airhead (aka Absentia or Acephalic) always says the same thing.

airhead = ConstantAgent("Airhead", "I know right???")

# Alice is a basic prompted LLM.  You are trying to improve on Alice.
# Don't change the prompt -- instead, make a new argubot with a new prompt.

alice = LLMAgent("Alice",
                 system="You are an intelligent bot who wants to broaden your user's mind. "
                        "Ask a conversation starter question.  Then, WHATEVER "
                        "position the user initially takes, push back on it. "
                        "Try to help the user see the other side of the issue. "
                        "Answer in 1-2 sentences. Be thoughtful and polite.")

############################################################
## Other argubot classes and instances -- add your own here! 
############################################################

class KialoAgent(Agent):
    """ KialoAgent subclasses the Agent class. It responds with a relevant claim from
    a Kialo database.  No LLM is used."""
    
    def __init__(self, name: str, kialo: Kialo):
        self.name = name
        self.kialo = kialo
                
    def response(self, d: Dialogue) -> str:

        if len(d) == 0:   
            # First turn.  Just start with a random claim from the Kialo database.
            claim = self.kialo.random_chain()[0]
        else:
            previous_turn = d[-1]['content']  # previous turn from user
            # Pick one of the top-3 most similar claims in the Kialo database,
            # restricting to the ones that list "con" arguments (counterarguments).
            neighbors = self.kialo.closest_claims(previous_turn, n=3, kind='has_cons')
            assert neighbors, "No claims to choose from; is Kialo data structure empty?"
            neighbor = random.choice(neighbors)
            log.info(f"[black on bright_green]Chose similar claim from Kialo:\n{neighbor}[/black on bright_green]")
            
            # Choose one of its "con" arguments as our response.
            claim = random.choice(self.kialo.cons[neighbor])
        
        return claim    
    
# Akiko doesn't use an LLM, but looks up an argument in a database.
  
akiko = KialoAgent("Akiko", Kialo(glob.glob("data/*.txt")))   # get the Kialo database from text files


###########################################
# Define your own additional argubots here!
###########################################

class KialoAgent2(Agent):
    """ KialoAgent2 subclasses the Agent class. It responds with a relevant claim from
    a Kialo database.  No LLM is used."""
    
    def __init__(self, name: str, kialo: Kialo):
        self.name = name
        self.kialo = kialo
                
    def response(self, d: Dialogue) -> str:

        if len(d) == 0:
            claim = self.kialo.random_chain()[0]
        else:
            i = 1
            result_neighbors = []
            while i <= len(d) and len(result_neighbors) == 0:
                previous_turn = d[-i]['content']
                neighbors = self.kialo.closest_claims(previous_turn, n=3, kind='has_cons')

                tokenized_corpus = [doc.split(" ") for doc in neighbors]
                bm25 = BM25_Index(tokenized_corpus)
                tokenized_query = previous_turn.split(" ")
                doc_scores = bm25.get_scores(tokenized_query)
                for j, score in enumerate(doc_scores):
                    if score > 0.05:
                        result_neighbors.append(neighbors[j])
                i += 2
            if len(result_neighbors) == 0:
                claim = "I am not sure I understand."
                log.info(f"[black on bright_green]Chose similar claim from Kialo:\nNone[/black on bright_green]")
            else:
                neighbor = random.choice(result_neighbors)
                claim = random.choice(self.kialo.cons[neighbor])
                log.info(f"[black on bright_green]Chose similar claim from Kialo:\n{neighbor}[/black on bright_green]")
            
        return claim

akiki = KialoAgent2("Akiki", Kialo(glob.glob("data/*.txt")))

class ShortyAgent(LLMAgent):
    """ ShortyAgent subclasses the LLMAgent class."""
    
    def __init__(self, name: str, agent: CharacterAgent, client: OpenAI = default_client):
        self.name = name
        self.agent = agent
        self.client = client
                
    def response(self, d: Dialogue) -> str:

        if len(d) == 0:
            claim = random.choice(self.agent.conversation_starters)
        else:
            outcomes = ['Heads', 'Tails', 'Neither']
            result = random.choice(outcomes)
            short_responses = ['Yes.', 'Ok.', 'Sure.', 'Sounds fishy.']
            if result == 'Neither':
                claim = random.choice(short_responses)
            elif result == 'Heads':
                response = self.client.chat.completions.create(messages=[{ "role": "system",
                                                        "content": "Give a generic answer that expresses a level of agreement but doesn't betray the topic." },
                                                        { "role": "user",
                                                        "content": d.script() }],
                                            model="gpt-3.5-turbo-1106", temperature=0.5)
                claim = response.choices[0].message.content
            else:
                response = self.client.chat.completions.create(messages=[{ "role": "system",
                                                        "content": "Give an argument to the last turn in the dialogue. Do not include the name of the speaker." },
                                                        { "role": "user",
                                                        "content": d.script() }],
                                            model="gpt-3.5-turbo-1106", temperature=0.5)
                claim = response.choices[0].message.content
            
        return claim

shorty_agent = CharacterAgent(shorty_character, "Shorty")
shorty = ShortyAgent("Shorty", shorty_agent)

class RAGAgent(LLMAgent):
    """ AwsomAgent subclasses the LLMAgent class."""
    
    def __init__(self, name: str, kialo: Kialo, client: OpenAI = default_client):
        self.name = name
        self.kialo = kialo
        self.client = client
                
    def response(self, d: Dialogue) -> str:

        if len(d) == 0:
            argument = self.kialo.random_chain()[0]
        else:
            response = self.client.chat.completions.create(messages=[{ "role": "system",
                                                        "content": "Given the context, provide what the last turn is really saying or implying." },
                                                        { "role": "user",
                                                        "content": "Aragorn: Fortunately, the vaccine was developed in record time. Human: Sounds fishy." },
                                                        { "role": "assistant",
                                                        "content": "Human [paraphrased]: A vaccine that was developed very quickly cannot be trusted. If its developers are claiming that it is safe and effective, I question their motives." },
                                                        { "role": "user",
                                                        "content": d.script() }],
                                            model="gpt-3.5-turbo-1106", temperature=0.5)
            s = response.choices[0].message.content

            c = self.kialo.closest_claims(s, kind='has_cons')[0]
            result = f'One possibly related claim from the Kialo debate website:\n\t"{c}"'
            if self.kialo.pros[c]:
                result += '\n' + '\n\t* '.join(["Some arguments from other Kialo users in favor of that claim:"] + self.kialo.pros[c])
            if self.kialo.cons[c]:
                result += '\n' + '\n\t* '.join(["Some arguments from other Kialo users against that claim:"] + self.kialo.cons[c])

            response = self.client.chat.completions.create(messages=[{ "role": "system",
                                                        "content": "Provide an appropriate response argument against the claim. Try to be morally sound." },
                                                        { "role": "user",
                                                        "content": result }],
                                            model="gpt-3.5-turbo-1106", temperature=0.5)
            argument = response.choices[0].message.content
            
        return argument

aragorn = RAGAgent("Aragorn", Kialo(glob.glob("data/*.txt")))


class AwsomAgent(LLMAgent):
    """ RAGAgent subclasses the LLMAgent class."""
    
    def __init__(self, name: str, kialo: Kialo, client: OpenAI = default_client):
        self.name = name
        self.kialo = kialo
        self.client = client
                
    def response(self, d: Dialogue) -> str:

        if len(d) == 0:
            argument = self.kialo.random_chain()[0]
        else:
            response = self.client.chat.completions.create(messages=[{ "role": "system",
                                                        "content": "Given the context, provide what the last turn is really saying or implying." },
                                                        { "role": "user",
                                                        "content": "Aragorn: Fortunately, the vaccine was developed in record time. Human: Sounds fishy." },
                                                        { "role": "assistant",
                                                        "content": "A vaccine that was developed very quickly cannot be trusted. If its developers are claiming that it is safe and effective, I question their motives." },
                                                        { "role": "user",
                                                        "content": "Aragorn: President Trump handled foreign affairs particularly well and improved foreign relations with many other countries. Human: Sounds fishy." },
                                                        { "role": "assistant",
                                                        "content": "President Trump may not have been loyal to the US." },
                                                        { "role": "user",
                                                        "content": "Aragorn: Cell-based meat and seafood product companies will never raise enough funding. Human: Not sure about that." },
                                                        { "role": "assistant",
                                                        "content": "At the end of 2018, there was a worldwide total of 27 companies developing cell-based meat and seafood products, of which 15 have already raised external funding." },
                                                        { "role": "user",
                                                        "content": d.script() }],
                                            model="gpt-3.5-turbo-1106", temperature=0.5)
            s = response.choices[0].message.content

            c = self.kialo.closest_claims(s, kind='has_cons')[0]
            result = f'One possibly related claim from the Kialo debate website:\n\t"{c}"'
            if self.kialo.pros[c]:
                result += '\n' + '\n\t* '.join(["Some arguments from other Kialo users in favor of that claim:"] + self.kialo.pros[c])
            if self.kialo.cons[c]:
                result += '\n' + '\n\t* '.join(["Some arguments from other Kialo users against that claim:"] + self.kialo.cons[c])

            response = self.client.chat.completions.create(messages=[{ "role": "system",
                                                        "content": "Provide an appropriate response argument against the claim. Try to be engaging, well-informed, morally sound, and skilled in your response." },
                                                        { "role": "user",
                                                        "content": result }],
                                            model="gpt-3.5-turbo-1106", temperature=0.5)
            argument = response.choices[0].message.content
            
        return argument
    
awsom = AwsomAgent("Awsom", Kialo(glob.glob("data/*.txt")))

class RAGAgentZero(LLMAgent):
    """ AwsomAgent subclasses the LLMAgent class."""
    
    def __init__(self, name: str, kialo: Kialo, client: OpenAI = default_client):
        self.name = name
        self.kialo = kialo
        self.client = client
                
    def response(self, d: Dialogue) -> str:

        if len(d) == 0:
            argument = self.kialo.random_chain()[0]
        else:
            response = self.client.chat.completions.create(messages=[{ "role": "system",
                                                        "content": "Given the context, provide what the last turn is really saying or implying." },
                                                        { "role": "user",
                                                        "content": d.script() }],
                                            model="gpt-3.5-turbo-1106", temperature=0.5)
            s = response.choices[0].message.content

            c = self.kialo.closest_claims(s, kind='has_cons')[0]
            result = f'One possibly related claim from the Kialo debate website:\n\t"{c}"'
            if self.kialo.pros[c]:
                result += '\n' + '\n\t* '.join(["Some arguments from other Kialo users in favor of that claim:"] + self.kialo.pros[c])
            if self.kialo.cons[c]:
                result += '\n' + '\n\t* '.join(["Some arguments from other Kialo users against that claim:"] + self.kialo.cons[c])

            response = self.client.chat.completions.create(messages=[{ "role": "system",
                                                        "content": "Provide an appropriate response argument against the claim. Try to be morally sound." },
                                                        { "role": "user",
                                                        "content": result }],
                                            model="gpt-3.5-turbo-1106", temperature=0.5)
            argument = response.choices[0].message.content
            
        return argument

frodo = RAGAgentZero("Frodo", Kialo(glob.glob("data/*.txt")))
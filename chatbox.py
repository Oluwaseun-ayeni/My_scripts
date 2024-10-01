import requests
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from transformers import pipeline, Conversation
from textblob import TextBlob
from google.cloud import translate_v2 as translate
from datetime import datetime

# Initialize the chatbot
chatbot = ChatBot('widebot')
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train("chatterbot.corpus.english")

# Create a pipeline for conversational AI (Transformer-based)
conversation_pipeline = pipeline("conversational", model="microsoft/DialoGPT-medium")

# Memory dictionary to store user context
memory = {}

# Initialize Google Cloud Translate API client
translate_client = translate.Client()

# Function to fetch weather data
def get_weather(city):
    api_key = "aa526cac7519cd27df0d281939b7e07e"
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(base_url)

    if response.status_code == 200:
        data = response.json()
        main = data['main']
        weather_desc = data['weather'][0]['description']
        temp = main['temp']
        return f"Current weather in {city}:\nTemperature: {temp}°C\nCondition: {weather_desc}"
    else:
        return "Sorry, I couldn't fetch the weather for that location."

# Function for sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Function to use Google Cloud Translate API
def translate_text(text, target_language='es'):
    try:
        translation = translate_client.translate(text, target_language=target_language)
        return translation['translatedText']
    except Exception as e:
        return f"Error translating text: {str(e)}"

# Function to get chatbot response from transformer
def chat_with_transformer(input_text):
    conversation = Conversation(input_text) 
    conversation_pipeline([conversation])    
    return conversation  


# Command parsing (Custom commands like time, translate, etc.)
def parse_command(user_input):
    if user_input.lower() == "time":
        return f"The current time is {datetime.now().strftime('%H:%M:%S')}"

    elif "translate" in user_input.lower():
        # Extract the language and text to translate
        parts = user_input.split("to")
        if len(parts) == 2:
            text_to_translate = parts[0].replace("translate", "").strip()
            target_language = parts[1].strip()
            return translate_text(text_to_translate, target_language)
        else:
            return "Please provide the text to translate and the target language. For example, 'translate Hello to es'."
    
    elif "weather in" in user_input.lower():
        city = user_input.split("in")[-1].strip()
        return get_weather(city)

    # Handle memory and context: user's name
    if "my name is" in user_input.lower():
        name = user_input.split("is")[-1].strip()
        memory['name'] = name
        return f"Nice to meet you, {name}!"

    if "what's my name" in user_input.lower() and 'name' in memory:
        return f"Your name is {memory['name']}."
    
    # Fallback to regular chatbot response (Transformer or ChatterBot)
    return get_chat_response(user_input)

# Function to handle responses from ChatterBot/Transformer and sentiment analysis
def get_chat_response(user_input):
    sentiment_score = analyze_sentiment(user_input)

    # Respond if the sentiment is negative
    if sentiment_score < -0.5:
        return "It seems you're upset. How can I assist you better?"

    # Default: use transformer for general conversation
    conversation = chat_with_transformer(user_input)
    return conversation.generated_responses[-1]  # Get the last response from the conversation

# Main loop for chatbot interaction
while True:
    user_input = input("You: ")

    if user_input.lower() == "bye":
        print("Chatbot: Goodbye!")
        break

    # Parse commands first
    response = parse_command(user_input)
    print(f"Chatbot: {response}")




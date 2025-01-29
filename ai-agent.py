import ollama
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="llama3.2")

chain = prompt | model

def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers

    Args:
        a: The first integer number
        b: The second integer number

    Returns:
        int: The sum of the two numbers
    """
    return int(a) + int(b)

def weather(loc: str):
    api_key = "55d8251d931854e9409413d7cb4e69e4"  # Add your OpenWeatherMap API key here

    url = f"https://api.openweathermap.org/data/2.5/weather?q={loc}&appid={api_key}"
    res = requests.get(url)

    if res.status_code == 200:
        weather_ob = res.json()
        return weather_ob
    else:
        return {"error": "Failed to fetch weather data"}

def summarizer(context):
    """
    Summarize the given context using the AI model.

    Args:
        context: The context to be summarized

    Returns:
        str: The summarized text
    """
    res = ollama.chat(
        'llama3.2',
        messages=[{'role': 'user', 'content': f"Summarize the following information: {context}"}],
    )

    return res['message']['content']

available_functions = {
    'add_two_numbers': add_two_numbers,
    'weather': weather
}

def chat():
    context = ""
    print("Welcome to AI chatbot! Type 'exit' to end the chat.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Check if the user input requires a tool call
        if any(keyword in user_input.lower() for keyword in ['add', 'sum', 'weather']):
            response = ollama.chat(
                'llama3.2',
                messages=[{'role': 'user', 'content': user_input}],
                tools=[add_two_numbers, weather],  # Actual function reference
            )

            for tool in response.get('message', {}).get('tool_calls', []):
                function_to_call = available_functions.get(tool['function']['name'])
                if function_to_call:
                    function_output = function_to_call(**tool['function']['arguments'])
                    # print('Function output:', function_output)
                    
                    # Summarize the function output
                    result = summarizer(function_output)


                    print('Summary:', result)
                    context += f"\nUser: {user_input} \n AI: {result}"
                else:
                    print('Function not found:', tool['function']['name'])
                
        else:
            # If no tool call is needed, proceed with the normal chat
            result = chain.invoke({"context": context, "question": user_input})
            print(result)
            context += f"\nUser: {user_input} \n AI: {result}"

if __name__ == '__main__':
    chat()
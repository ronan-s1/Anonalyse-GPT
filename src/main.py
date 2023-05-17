from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import pandas as pd
import numpy as np
import os
from dotenv import find_dotenv, load_dotenv
import random
import string
from prettytable import PrettyTable

# for column names
def random_word():
    length = 6
    letters = string.ascii_lowercase
    word = "".join(random.choice(letters) for i in range(length))
    while "df" in word:
        word = "".join(random.choice(letters) for i in range(length))
    return word


def create_fake_and_map(real_df):
    column_data_types = real_df.dtypes

    # Generate dummy data for each column based on the data types
    fake_df = pd.DataFrame()
    for column, data_type in column_data_types.items():
        if data_type == "int64":
            fake_df[column] = np.random.randint(1, 100, size=5)
        elif data_type == "float64":
            fake_df[column] = np.random.uniform(0, 1, size=5)
        elif data_type == "object":
            fake_df[column] = np.random.choice(["A", "B", "C"], 5)
        elif data_type == "datetime64[ns]":
            fake_df[column] = pd.date_range(start="1/1/2023", periods=5)
        elif data_type == "bool":
            fake_df[column] = np.random.choice([True, False], size=5)

    column_mapping, table = create_column_map(column_data_types)
    fake_df = fake_df.rename(columns=column_mapping)

    return fake_df, column_mapping, table


# Create a dictionary to map original column names to fake column names and their data types
def create_column_map(column_data_types):
    column_mapping = {}
    table = PrettyTable()
    table.align = "l"
    table.field_names = ["Real", "Fake", "Type"]
    for i, column in enumerate(column_data_types.index):
        modified_column_name = f"{random_word()}_{i+1}"
        column_mapping[column] = modified_column_name
        table.add_row([column, modified_column_name, column_data_types[i]])

    print(table)
    return column_mapping, table


def anonymise_user_input(user_input, column_mapping):
    for column in column_mapping:
        user_input = user_input.replace(column, column_mapping[column])
    return user_input


def deanonymise_output(output, column_mapping):
    output = output.replace("df", "real_df")
    for column in column_mapping:
        output = output.replace(column_mapping[column], column)
    return output


def main(column_mapping, fake_df, user_input):
    # API key
    load_dotenv(find_dotenv())
    os.environ["OPENAI_API_KEY"] = os.environ.get("KEY")
    
    custom_template = """
    You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
    
    You should use the tools below to answer the question posed of you:
    
    python_repl_ast: A Python shell. Use this to execute python commands. Input should be a valid python command. When using this tool, sometimes output is abbreviated - make sure it does not look abbreviated before using it in your answer.
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [python_repl_ast]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: The Action Input(s) to get the final answer.
    
    Note that this dataframe (`df`) contains dummy data and dummy column names for anonymisation so some results may not make sense.
    
    This is the result of `print(df.head())`:\n{df}\n\nBegin!\nQuestion: {input}\n{agent_scratchpad}
    """

    user_input = anonymise_user_input(user_input, column_mapping)
    # creating agent
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), fake_df, verbose=True)
    agent.agent.llm_chain.prompt.template = custom_template
    final_output = agent.run(user_input)

    # alter generated output and print answer
    generated_query = deanonymise_output(final_output, column_mapping)

    return generated_query
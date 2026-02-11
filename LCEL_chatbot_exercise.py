# This import allows us to use modern Python type hints
# (like list[str]) even if you are working with older Python versions
from __future__ import annotations

# standard library imports
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

# langchain imports
# ChatOpenAI is the LLM wrapper for OpenAI's chat models (e.g., gpt-4o-mini).
from langchain_openai import ChatOpenAI

# These imports are for building the prompt and parsing the output in an LCEL chain.
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load Api key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


# check to make sure openai_api_key is loaded
def load_openai_api_key(openai_api_key):
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    else:
        print("OPENAI_API_KEY loaded successfully.")


# Call the function to check if the API key is loaded
load_openai_api_key(openai_api_key)


# System Prompt (defines the *role* and *behavior* of the chatbot.) + Examples (few-shot prompting)
INSTRUCTIONS = """You are a math assistant who focuses on mathematical calculations and numerical 
questions. Respond in markdown format and provide clear, concise answers to math-related questions.
If a question is not related to math, politely inform the user that you can only assist
with math-related queries. Always provide the final answer at the end of your response.
"""
# Few-shot examples for the chatbot (can be modified or expanded)
# First line is the human input, second line is the AI response.

DEFAULT_EXAMPLES: List[
    Dict[str, str]
] = [  # Each example is a dictionary with "input" and "output" keys that define the format
    {"input": "What is 2 + 2?", "output": "The answer is 4."},
    {
        "input": "What is the square root of 8?",
        "output": "The square root of 8 is approximately 2.828.",
    },
    {"input": "What is 10 - 4?", "output": "10 minus 4 equals 6."},
]


# Define the ChatBot class that encapsulates the LCEL chain logic for our math assistant chatbot.
class ChatBot:
    """
    Few-shot chatbot implemented as an LCEL chain:
        chain = prompt | llm | StrOutputParser()

    """

    def __init__(
        self,
        name: str,
        instructions: str,
        examples: Optional[List[Dict[str, str]]] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        # store configuration settings (name, system instructions, few-shot examples,LLM parameters).
        self.name = name
        self.instructions = instructions
        self.model = model
        self.temperature = temperature
        # If no examples are provided, fall back to defaults
        if examples is None:
            examples = DEFAULT_EXAMPLES
        # Validate that each example has the required keys ("input" and "output")
        required_keys = {"input", "output"}
        for i, ex in enumerate(examples):
            if not required_keys.issubset(ex):
                raise ValueError(
                    f"Example #{i} must contain keys {required_keys}. Got: {list(ex.keys())}"
                )
        self.examples = examples

        # Template for a SINGLE example (human -> ai)
        self._example_prompt = ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{output}")]
        )
        # Wrap multiple examples into a few-shot prompt
        self._few_shot = FewShotChatMessagePromptTemplate(
            example_prompt=self._example_prompt,
            examples=self.examples,
        )
        # Combine system instructions, few-shot examples, and user input into a single prompt template
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.instructions),
                self._few_shot,
                ("human", "{user_input}"),
            ]
        )

        # Create the LLM object with the specified model and temperature settings
        self._llm = ChatOpenAI(model=self.model, temperature=self.temperature)
        # Create the output parser (StrOutputParser) to ensure the output is returned as a string
        self._parser = StrOutputParser()
        # Combine the prompt, LLM, and parser into a single LCEL chain
        self._chain = self._prompt | self._llm | self._parser

    def invoke(self, user_input: str, temperature: Optional[float] = None) -> str:
        """
        Invoke the LCEL chain.
        If temperature is provided, use a temporary LLM (and temporary chain)
        for this call only (keeps the bot's default temperature unchanged).
        """
        if temperature is None:
            return self._chain.invoke({"user_input": user_input})
        # If a custom temperature is provided, create a temporary LLM and chain
        temp_llm = ChatOpenAI(model=self.model, temperature=temperature)
        temp_chain = self._prompt | temp_llm | self._parser
        return temp_chain.invoke({"user_input": user_input})


# Function to build and return an instance of the MathWhiz chatbot with the specified configuration.
def build_mathwhiz() -> ChatBot:
    return ChatBot(
        name="MathWhiz",
        instructions=INSTRUCTIONS,
        examples=DEFAULT_EXAMPLES,
        model="gpt-4o-mini",
        temperature=0.0,
    )


# Create the chatbot instance using the build_mathwhiz function
bot = build_mathwhiz()


def question(prompt: str, temperature: float = 0.0) -> None:
    """
    Convenience function for interactive use.
    Lets user ask questions without calling invoke() directly.
    """
    print(bot.invoke(prompt, temperature=temperature))


def main() -> None:
    """
    This function defines the entry point of the script.

    In larger AI or data analysis projects, files often serve two roles:
    1) As reusable modules (imported by other files)
    2) As runnable scripts (executed directly)

    Wrapping execution logic inside `main()` allows this file to act
    as a clean, reusable component in an agentic system while still
    supporting direct execution for testing and demos.

    The `if __name__ == "__main__":` guard ensures that this code runs
    only when explicitly intended, which becomes critical as systems
    grow more modular and interconnected.
    """
    print(bot.invoke("What is 356 divided by 3?"))
    print(bot.invoke("Explain the Pythagorean theorem in 2 sentences."))


if __name__ == "__main__":
    main()

# Ask the bot questions to see how it responds
question("What is 12 * 19?", temperature=0.0)
question("Why does my cat always want to sleep on my keyboard?", temperature=0.0)

# Now that you know how it works, experiment with new things.
# - Change the Temperature
# - Modify Personality in INSTRUCTIONS (e.g., be more humorous, more formal, etc.)
# - Increase Few-Shot Examples
# - Create your own chatbot

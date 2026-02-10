from __future__ import annotations

import os
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


# Load Api key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


# check to make sure openai_api_key is loaded
def load_openai_api_key(openai_api_key):
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    else:
        print("OPENAI_API_KEY loaded successfully.")


load_openai_api_key(openai_api_key)


# System Prompt (instructions) + Examples (few-shot prompting) + ChatBot class implementation
INSTRUCTIONS = """You are a math assistant who focuses on mathematical calculations and numerical 
questions. Respond in markdown format and provide clear, concise answers to math-related questions.
If a question is not related to math, politely inform the user that you can only assist
with math-related queries. Always provide the final answer at the end of your response.
"""

DEFAULT_EXAMPLES: List[Dict[str, str]] = [
    {"input": "What is 2 + 2?", "output": "The answer is 4."},
    {
        "input": "What is the square root of 8?",
        "output": "The square root of 8 is approximately 2.828.",
    },
    {"input": "What is 10 - 4?", "output": "10 minus 4 equals 6."},
]


class ChatBotLCEL:
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
        self.name = name
        self.instructions = instructions
        self.model = model
        self.temperature = temperature

        if examples is None:
            examples = DEFAULT_EXAMPLES

        required_keys = {"input", "output"}
        for i, ex in enumerate(examples):
            if not required_keys.issubset(ex):
                raise ValueError(
                    f"Example #{i} must contain keys {required_keys}. Got: {list(ex.keys())}"
                )
        self.examples = examples

        # Build the prompt pieces once
        self._example_prompt = ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{output}")]
        )
        self._few_shot = FewShotChatMessagePromptTemplate(
            example_prompt=self._example_prompt,
            examples=self.examples,
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.instructions),
                self._few_shot,
                ("human", "{user_input}"),
            ]
        )

        # Base LLM + base chain
        self._llm = ChatOpenAI(model=self.model, temperature=self.temperature)
        self._parser = StrOutputParser()
        self._chain = self._prompt | self._llm | self._parser

    def invoke(self, user_input: str, temperature: Optional[float] = None) -> str:
        """
        Invoke the LCEL chain.
        If temperature is provided, use a temporary LLM (and temporary chain)
        for this call only (keeps the bot's default temperature unchanged).
        """
        if temperature is None:
            return self._chain.invoke({"user_input": user_input})

        temp_llm = ChatOpenAI(model=self.model, temperature=temperature)
        temp_chain = self._prompt | temp_llm | self._parser
        return temp_chain.invoke({"user_input": user_input})


def build_mathwhiz() -> ChatBotLCEL:
    return ChatBotLCEL(
        name="MathWhiz",
        instructions=INSTRUCTIONS,
        examples=DEFAULT_EXAMPLES,
        model="gpt-4o-mini",
        temperature=0.0,
    )


# Interactive-friendly helpers
bot = build_mathwhiz()


def question(prompt: str, temperature: float = 0.0) -> None:
    print(bot.invoke(prompt, temperature=temperature))


def main() -> None:
    print(bot.invoke("What is 356 divided by 3?"))
    print(bot.invoke("Explain the Pythagorean theorem in 2 sentences."))


if __name__ == "__main__":
    main()

# Ask questions to the bot to see how it responds
question("What is 12 * 19?", temperature=0.0)
question("Why does my cat always want to sleep on my keyboard?", temperature=0.0)

# Now that you know how it works, experiment with new things.
# - Change the Temperature
# - Modify Personality in INSTRUCTIONS (e.g., be more humorous, more formal, etc.)
# - Increase Few-Shot Examples
# - Create your own chatbot

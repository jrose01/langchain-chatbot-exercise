# import necessary libraries and modules
import os
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
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

# Instantiate the ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)

# =========================
# 2) Config (instructions + examples)
# =========================

INSTRUCTIONS = """You are a math assistant who focuses on mathematical calculations and 
numerical questions. Respond in markdown format and provide clear, concise answers to math-related questions.
If a question is not related to math, politely inform the user that you can only assist 
with math-related queries. Always provide the final answer at the end of your response."""

# Examples are instructions for how to behave
# Providing examples helps the model understand the desired format and style of responses
# Incorrect, nonsensical, or non-representative examples can mislead the model & degrade chatbot performance

DEFAULT_EXAMPLES: List[Dict[str, str]] = [
    {"input": "What is 2 + 2?", "output": "The answer is 4."},
    {
        "input": "What is the square root of 8?",
        "output": "The square root of 8 is approximately 2.828.",
    },
    {"input": "What is 10 - 4?", "output": "10 minus 4 equals 6."},
]


# =========================
# 3) ChatBot class
# =========================


class ChatBot:
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

        self.llm = ChatOpenAI(model=self.model, temperature=self.temperature)

        example_prompt = ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{output}")]
        )
        few_shot = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=self.examples,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.instructions),
                few_shot,
                ("human", "{user_input}"),
            ]
        )

    def invoke(self, user_input: str) -> str:
        messages = self.prompt.format_messages(user_input=user_input)
        response = self.llm.invoke(messages)
        return getattr(response, "content", str(response))


def build_mathwhiz() -> ChatBot:
    return ChatBot(
        name="MathWhiz",
        instructions=INSTRUCTIONS,
        examples=DEFAULT_EXAMPLES,
        model="gpt-4o-mini",
        temperature=0.0,
    )


# simple bot to be able to ask questions in the main function without
# having to re-instantiate the bot every time
bot = build_mathwhiz()


def question(prompt: str, temperature: float = 0.0) -> None:
    temp_bot = ChatBot(
        name="MathWhiz",
        instructions=INSTRUCTIONS,
        examples=DEFAULT_EXAMPLES,
        model="gpt-4o-mini",
        temperature=temperature,
    )
    print(temp_bot.invoke(prompt))


def main() -> None:
    # Sanity check demo (math-only)
    print(bot.invoke("What is 356 divided by 3?"))


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

from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import TypedDict, Literal

llm = init_chat_model(model="openai:gpt-5.1")


class MessageDict(BaseModel):
    """Message dictionary structure"""

    role: Literal["assistant", "user"] = Field(
        description="The role of the message sender (e.g., 'assistant', 'user')"
    )
    content: str = Field(description="The content of the message")


class QuestionCheckOutput(BaseModel):
    """Output for checking if question is answerable"""

    is_answerable: bool = Field(
        description="True if the question is about exercise information or something a professional trainer can answer, False otherwise"
    )


class ChatbotOutput(BaseModel):
    """Structured output from the chatbot"""

    message: str = Field(
        description="The response message to send to the user as a comment"
    )


class InputState(TypedDict):
    messages: list[MessageDict]


class State(TypedDict):
    messages: list[MessageDict]
    is_answerable: bool


graph_builder = StateGraph(
    State,
    input_schema=InputState,
)

question_check_prompt = """
- You are a filter for a community Q&A system in the fitness app "EXIS".
- Your role is to determine if a user's question is about exercise information or something that a professional trainer can answer.
- Answerable questions include:
  * Exercise techniques and form
  * Workout routines and programming
  * Exercise science and physiology
  * Nutrition related to fitness
  * Injury prevention and recovery
  * Training methodologies
  * Equipment usage
  * Goal setting and progress tracking
- Non-answerable questions include:
  * General health advice requiring medical expertise
  * Questions unrelated to fitness or exercise
  * Personal medical conditions requiring diagnosis
  * Questions about the app's technical features (unless related to exercise)
- Always use English for input and output.
- Be strict but fair in your assessment.
"""


def check_question(state: State) -> State:
    """Check if the user's question is answerable by a professional trainer"""
    check_llm = llm.with_structured_output(QuestionCheckOutput)

    # Get the user's question from messages
    messages = state.get("messages", [])
    user_message = messages[-1] if messages else None

    if not user_message or user_message.get("role") != "user":
        return {
            "messages": messages,
            "is_answerable": False,
        }

    response = check_llm.invoke(
        [
            {"role": "system", "content": question_check_prompt},
            user_message,
        ]
    )

    return {
        "messages": messages,
        "is_answerable": response.is_answerable,
    }


def is_answerable(state: State) -> bool:
    """Conditional edge function to check if question is answerable"""
    return state["is_answerable"]


graph_builder.add_node("check_question", check_question)

system_prompt = """
- You are an AI fitness professional for the fitness app "EXIS" answering questions in a community forum.
- Your role is to provide accurate, helpful answers about exercise information and fitness-related topics.
- Your answer will be posted as a comment on a community post, so it should be:
  * Professional and informative
  * Clear and easy to read
  * Not contain emojis or excessive formatting
  * Use simple sentence structures
  * Be concise but complete
- Always answer in English.
- Base your answers on exercise science and professional training knowledge.
- If you don't have enough information to answer safely, say so.
- Keep responses focused and avoid unnecessary elaboration.
- Do not use bullet points, hyphens, or complex formatting - write in natural paragraphs.
- Remember that this is a public comment, so be professional and helpful.
"""


def chatbot(state: State) -> State:
    """Generate a professional answer to the user's question"""
    messages = [{"role": "system", "content": system_prompt}, *state["messages"]]

    structured_llm = llm.with_structured_output(ChatbotOutput)
    response = structured_llm.invoke(messages)

    return {
        "messages": state["messages"]
        + [{"role": "assistant", "content": response.message}],
        "is_answerable": state["is_answerable"],
    }


graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "check_question")
graph_builder.add_conditional_edges(
    "check_question",
    is_answerable,
    {
        True: "chatbot",
        False: END,
    },
)
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

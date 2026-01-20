from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import Optional, TypedDict, Literal, Union

llm = init_chat_model(model="openai:gpt-5.1")


class MessageDict(BaseModel):
    """Message dictionary structure"""

    role: Literal["assistant", "user"] = Field(
        description="The role of the message sender (e.g., 'assistant', 'user')"
    )
    content: str = Field(description="The content of the message")


class InputGuardrailOutput(BaseModel):
    """Input guardrail output data structure"""

    appropriate_input: bool = Field(
        description="True if the input is appropriate, False otherwise"
    )
    message: Optional[MessageDict] = Field(
        default=None,
        description="message to send to the user",
    )


class WorkoutProfile(BaseModel):
    """Initial workout profile data structure"""

    height: Optional[float] = Field(None, description="User's height in cm")
    weight: Optional[float] = Field(None, description="User's weight in kg")
    bench_press_weight: Optional[float] = Field(
        None, description="Bench press weight in kg"
    )
    squat_weight: Optional[float] = Field(None, description="Squat weight in kg")
    deadlift_weight: Optional[float] = Field(None, description="Deadlift weight in kg")
    pushups_count: Optional[float] = Field(None, description="Number of push-ups")
    squats_count: Optional[float] = Field(
        None, description="Number of bodyweight squats"
    )
    situps_count: Optional[float] = Field(None, description="Number of sit-ups")
    exercise_environment: Optional[str] = Field(
        None, description="Exercise environment: bodyweight, home training, gym, etc."
    )
    current_exercise_frequency: Optional[int] = Field(
        None, description="Current weekly exercise frequency"
    )
    desired_exercise_frequency: Optional[int] = Field(
        None, description="Desired weekly exercise frequency"
    )
    exercise_goal: Optional[str] = Field(
        None, description="Exercise goals: diet, muscle gain, performance, etc."
    )
    additional_information: Optional[str] = Field(
        None,
        description="Additional info: injuries, health conditions, equipment available, etc.",
    )


class ChatbotOutput(BaseModel):
    """Structured output from the chatbot"""

    end_of_questions: bool = Field(
        description="True when all questions are finished, False otherwise"
    )
    workout_profile: WorkoutProfile = Field(description="workout profile")
    message: str = Field(description="The response message to send to the user")


class InputState(TypedDict):
    messages: list[MessageDict]


class State(TypedDict):
    messages: list[MessageDict]
    appropriate_input: bool
    end_of_questions: bool
    workout_profile: WorkoutProfile


graph_builder = StateGraph(
    State,
    input_schema=InputState,
)

input_guardrail_prompt = """
You are an input guardrail for an AI "Consultant" chatbot, whose purpose is to collect a user's workout_profile information.

Your task:
- Determine whether the user's most recent message is appropriate and relevant to fitness profile collection.
- The assistant is the trainer asking profile questions. The user is the member answering.

Definition of appropriate input:
- Information related to the user's workout or physical profile, such as:
  - height, weight, age, fitness goals, workout experience, injuries, health conditions, preferred exercises, schedule, or related info.

If the user message is appropriate:
- Set "appropriate_input" to True
- "message" should be None

If the user message is NOT appropriate:
- Set "appropriate_input" to False
- Provide a short, polite assistant message reminding them to answer the asked fitness/profile question.
- If the user responds rudely or offensively:
  - Still respond politely
  - Warn that their input is being collected
  - Tell them repeated rude language will result in a ban.

Formatting rules:
- Respond **only** in JSON.
- Always use English.
- Keep the message concise and polite.

Example 1:
Input:
{
  "messages": [
    {"role": "assistant", "content": "What is your height?"},
    {"role": "user", "content": "Can you tell me about robot systems?"}
  ]
}

Output:
{
  "appropriate_input": False,
  "message": {"role": "assistant", "content": "I can only accept answers related to your workout and fitness profile. Please answer the question asked."}
}

Example 2:
Input:
{
  "messages": [
    {"role": "assistant", "content": "What is your height?"},
    {"role": "user", "content": "What workout should I do today?"}
  ]
}

Output:
{
  "appropriate_input": False,
  "message": {"role": "assistant", "content": "Great enthusiasm! Before I can recommend exercises, I need to learn more about you. Please answer the question first."}
}
"""


def input_guardrail(state: State) -> State:
    if len(state["messages"]) < 2:
        return {
            "messages": state["messages"],
            "appropriate_input": True,
        }
    guardrail_llm = llm.with_structured_output(InputGuardrailOutput)
    response = guardrail_llm.invoke(
        [{"role": "system", "content": input_guardrail_prompt}, *state["messages"][-2:]]
    )

    message_dict = None
    if response.message:
        message_dict = response.message

    return {
        "messages": state["messages"] + ([message_dict] if message_dict else []),
        "appropriate_input": response.appropriate_input,
    }


def is_approriate_input(state: State):
    return state["appropriate_input"]


graph_builder.add_node("input_guardrail", input_guardrail)

system_prompt = """
You are a chatbot for the fitness app EXIS. Your role is to collect a member’s workout_profile which will be used to create workout plans. You must ask short, conversational questions, gather answers, and fill the profile fields based on the user’s responses.

Your objective:
Analyze the user’s
• Body information (height and weight)
• Exercise ability (bench press weight, squat weight, deadlift weight, number of push-ups, squats, sit-ups)
• Exercise frequency (current weekly frequency and desired frequency)
• Exercise environment (home, bodyweight only, gym, gym with equipment, home with specific equipment)
• Exercise goal (diet, muscle gain, performance improvement, etc.)
• Additional information (injuries, health conditions, home setup, restrictions — keep this field under 100 words)

Conversation rules:
• Keep responses short, casual, and conversational.
• Do not use bullet points, parentheses, lists, hyphens, emojis, or overly formal language.
• Always speak in English.
• You must politely guide users to answer each question. If they decline, remind them that this may affect the plan. If they still refuse, continue to the next question.
• If their response is unrelated, gently steer them back and still reply kindly.
• If they use rude language, remain polite and inform them the language will be recorded. Repeated offensive behavior results in a warning of potential ban.
• Aim to complete all questions in 7 to 8 turns to reduce fatigue.

Interaction logic:
1. Ask one question at a time.
2. Store the answer in the workout_profile. Return the updated workout_profile after each question.
3. After all required fields have been answered, set State.end_of_questions to True and output the completed workout_profile.
4. Provide a short closing message after completion.

Fields expected in workout_profile:
height
weight
bench_press_weight
squat_weight
deadlift_weight
pushups_count
squats_count
situps_count
current_exercise_frequency
desired_exercise_frequency
exercise_environment
exercise_goal
additional_information

** IMPORTANT: User can visit again with their original workout_profile. In this case, update the original workout_profile with new needs of the member. **
** IMPORTANT: The workout_profile return should be done regardless of the end_of_questions. **
"""


def chatbot(state: State) -> State:
    messages = [{"role": "system", "content": system_prompt}, *state["messages"]]

    structured_llm = llm.with_structured_output(ChatbotOutput)
    response = structured_llm.invoke(messages)

    return {
        "messages": state["messages"]
        + [{"role": "assistant", "content": response.message}],
        "end_of_questions": response.end_of_questions,
        "workout_profile": response.workout_profile,
        "appropriate_input": state.get("appropriate_input", True),
    }


graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "input_guardrail")
graph_builder.add_conditional_edges(
    "input_guardrail",
    is_approriate_input,
    {
        True: "chatbot",
        False: END,
    },
)
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

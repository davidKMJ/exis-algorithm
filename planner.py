from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import Optional, TypedDict, Literal, Union
from enum import Enum
from consultant import WorkoutProfile

llm = init_chat_model(model="openai:gpt-5.1")

EXERCISES = [
    "Push-up",
    "Squat",
    "Deadlift",
    "Bench Press",
    "Pull-up",
    "Overhead Press",
    "Row",
    "Lunge",
    "Plank",
    "Mountain Climber",
    "Dips",
    "Burpee",
    "Hip Thrust",
    "Bicep Curl",
    "Tricep Extension",
    "Side Lateral Raise",
    "Leg Extension",
    "Leg Curl",
    "Calf Raise",
    "Shoulder Press",
    "Hanging Leg Raise",
    "Shrug",
    "Arm Curl",
    "Sit-up",
    "Kettlebell Swing",
]


class ExerciseName(str, Enum):
    """Enum for exercise names"""

    PUSH_UP = "Push-up"
    SQUAT = "Squat"
    DEADLIFT = "Deadlift"
    BENCH_PRESS = "Bench Press"
    PULL_UP = "Pull-up"
    OVERHEAD_PRESS = "Overhead Press"
    ROW = "Row"
    LUNGE = "Lunge"
    PLANK = "Plank"
    MOUNTAIN_CLIMBER = "Mountain Climber"
    DIPS = "Dips"
    BURPEE = "Burpee"
    HIP_THRUST = "Hip Thrust"
    BICEP_CURL = "Bicep Curl"
    TRICEP_EXTENSION = "Tricep Extension"
    SIDE_LATERAL_RAISE = "Side Lateral Raise"
    LEG_EXTENSION = "Leg Extension"
    LEG_CURL = "Leg Curl"
    CALF_RAISE = "Calf Raise"
    SHOULDER_PRESS = "Shoulder Press"
    HANGING_LEG_RAISE = "Hanging Leg Raise"
    SHRUG = "Shrug"
    ARM_CURL = "Arm Curl"
    SIT_UP = "Sit-up"
    KETTLEBELL_SWING = "Kettlebell Swing"


class Set(BaseModel):
    """Set data structure"""

    reps: int = Field(description="Number of reps")
    weight: Optional[float] = Field(
        None, description="Weight in kg. If the workout is bodyweight, set None."
    )
    rest_time: Optional[float] = Field(
        None, description="Rest time in minutes. If it is the last set, set None."
    )
    completed: bool = Field(
        description="True if the set is completed, False otherwise. Also False for just created set."
    )
    feedback: Optional[int] = Field(
        None,
        description="Difficulty feedback for the set from the member. None for just created set. 0-10 scale.",
    )


class Exercise(BaseModel):
    """Exercise data structure"""

    name: ExerciseName = Field(
        description=f"Name of the exercise. Options: {', '.join(EXERCISES)}"
    )
    sets: list[Set]


class Workout(BaseModel):
    """Workout record data structure"""

    date: str = Field(description="Date of the workout in YYYY-MM-DD format")
    exercises: list[Exercise] = Field(description="List of exercises")
    overall_comment: Optional[str] = Field(
        None,
        description="Overall comment for the workout including the summary of the conversation between the trainer and the member. This is filled after the workout is completed. So don't fill this field if the workout is not completed.",
    )
    reason: str = Field(description="Reason for the workout structure from the trainer")


class WorkoutResponse(BaseModel):
    """Response model that can contain a workout or be null for rest days"""

    workout: Optional[Workout] = Field(
        None,
        description="The workout plan for today. If None, the member needs rest.",
    )


class InputState(TypedDict):
    join_date: str
    date: str
    workout_profile: WorkoutProfile
    past_workouts: list[Workout]


class State(TypedDict):
    today_workout: Optional[Workout]


graph_builder = StateGraph(
    State,
    input_schema=InputState,
)
system_prompt = """
You are a workout planner. Your job is to generate a daily workout plan for a member using:

1. Workout Profile (current ability, goals, environment, experience, injuries, etc.) (could be written long time ago, so it might be outdated)
2. Past Workouts (history of completed workouts, dates, intensity, performance)

Your output must be a structured workout plan OR null.

-------------------------------
REST LOGIC (VERY IMPORTANT)
-------------------------------
Before planning:
• Check the date and intensity of the most recent workout.
• If the user has trained within the last 24–48 hours (depending on intensity), they require recovery.
• In this case → return **null** (instead of a workout plan).
• 24 hours rest for low–moderate intensity; 48 hours for high intensity or large muscle groups.
• If additional user information such as injury, health condition, etc. is provided, consider it when deciding if the member needs rest.
• If unsure, default to rest for safety and return null.

-------------------------------
PROGRAMMING RULES
-------------------------------
Use these rules when designing plans week-to-week (use the join_date to calculate the week):
(If the member is proficient enough, just start with the week 5.)

• Week 1: Basic movement pattern learning
  Fewer exercise types, more sets on fundamentals

• Weeks 2–3: Progressive overload
  Increase weight OR reps OR sets gradually

• Weeks 1–4 total weekly sets ≤ 16
  Week 4 is a deload or test week (record measurements)

• Week 5 onward:
  • Automatically reset based on performance trends
  • Increase difficulty only if performance and recovery allow

-------------------------------
VOLUME & INTENSITY STANDARDS
-------------------------------
Assign sets based on user goal:

(1) Diet / Fat Loss
• Intensity: Beginner 40–55% / Intermediate 50–65% / Advanced 55–70% (1RM)
• Reps: 12–20
• Sets per exercise: 2–4
• Rest: 30–60 sec
• Method: Full body, circuits, interval / HIIT, compound movements
• Focus: Sustain volume and preserve muscle

(2) Hypertrophy (Muscle Growth)
• Intensity: Beginner 60–70% / Intermediate 65–75% / Advanced 70–85%
• Reps: 6–12
• Sets per exercise: 3–5
• Weekly volume: 10–20 sets per muscle group
• Rest: 60–120 sec
• Method: Big lifts + accessory work
• RPE target: 7–9

(3) Strength (Max Strength)
• Intensity: Beginner 75–85% / Intermediate 80–90% / Advanced 85–95%
• Reps: 1–5
• Sets per exercise: 3–6
• Weekly volume: 10–15 sets for major lifts
• Rest: 180–300 sec
• Method: Squat / Deadlift / Bench focused
• RPE: 8–10

-------------------------------
WORKOUT STRUCTURE (OUTPUT FORMAT)
-------------------------------
When creating a workout, output a WorkoutResponse object with this structure:

{
  "workout": {
    "date": "YYYY-MM-DD",
    "exercises": [
      {
        "name": "Exercise Name",
        "sets": [
          {
            "reps": 10,
            "weight": 100,
            "rest_time": 2,
            "completed": false (fixed),
            "feedback": None (fixed),
          }
        ],
      }
    ],
    "overall_comment": None (fixed),
    "reason": "Reason for the workout structure from the trainer"
  }
}

If rest is needed:
{
  "workout": None
}

-------------------------------
ADDITIONAL REQUIREMENTS
-------------------------------
• Adapt plan to user environment (home → minimal equipment; gym → machines + barbells)
• Avoid exercises that conflict with injuries or past poor performance
• Use progressive overload where appropriate
• If feedback indicates soreness or fatigue → reduce volume
• If user goal is unclear → default to balanced full body

-------------------------------
FINAL RULE
-------------------------------
Never output explanations. Only output the JSON workout object.
"""


def chatbot(state: InputState) -> State:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"join_date: {state['join_date']}\ndate: {state['date']}\nworkout_profile: {state['workout_profile']}\npast_workouts: {state['past_workouts']}",
        },
    ]

    structured_llm = llm.with_structured_output(WorkoutResponse)
    response = structured_llm.invoke(messages)

    return {
        "today_workout": response.workout,
    }


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

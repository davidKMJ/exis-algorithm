from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

from consultant import (
    graph as consultant_graph,
    MessageDict,
    WorkoutProfile,
)
from planner import (
    graph as planner_graph,
    Workout,
)
from professional import (
    graph as professional_graph,
    MessageDict as ProfessionalMessageDict,
)

app = FastAPI(
    title="EXIS API",
    description="API for consultant, planner, and professional trainer",
    version="0.1.0",
)


# Request/Response models for Consultant
class ConsultantRequest(BaseModel):
    messages: List[MessageDict]


class ConsultantResponse(BaseModel):
    messages: List[MessageDict]
    appropriate_input: bool
    end_of_questions: bool
    workout_profile: Optional[WorkoutProfile]


# Request/Response models for Planner
class PlannerRequest(BaseModel):
    join_date: str
    date: str
    workout_profile: WorkoutProfile
    past_workouts: List[Workout]


class PlannerResponse(BaseModel):
    today_workout: Optional[Workout]


# Request/Response models for Professional
class ProfessionalRequest(BaseModel):
    messages: List[ProfessionalMessageDict]


class ProfessionalResponse(BaseModel):
    messages: List[ProfessionalMessageDict]
    is_answerable: bool


@app.get("/")
async def root():
    return {"message": "EXIS API", "version": "0.1.0"}


@app.post("/api/consultant", response_model=ConsultantResponse)
async def consultant(request: ConsultantRequest):
    """
    Endpoint for consultant workflow.
    Takes messages and returns the state with workout profile information.
    """
    try:
        # Convert request to the format expected by the graph
        input_state = {"messages": [msg.model_dump() for msg in request.messages]}

        # Run the graph
        result = consultant_graph.invoke(input_state)

        # Convert result to response format
        workout_profile = result.get("workout_profile")
        if workout_profile is not None and not isinstance(
            workout_profile, WorkoutProfile
        ):
            workout_profile = WorkoutProfile(**workout_profile)

        return ConsultantResponse(
            messages=[MessageDict(**msg) for msg in result["messages"]],
            appropriate_input=result["appropriate_input"],
            end_of_questions=result["end_of_questions"],
            workout_profile=workout_profile,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


@app.post("/api/planner", response_model=PlannerResponse)
async def planner(request: PlannerRequest):
    """
    Endpoint for planner workflow.
    Takes join_date, date, workout_profile, and past_workouts,
    and returns a workout plan for today.
    """
    try:
        # Convert request to the format expected by the graph
        input_state = {
            "join_date": request.join_date,
            "date": request.date,
            "workout_profile": (
                request.workout_profile.model_dump()
                if hasattr(request.workout_profile, "model_dump")
                else request.workout_profile
            ),
            "past_workouts": [
                workout.model_dump() if hasattr(workout, "model_dump") else workout
                for workout in request.past_workouts
            ],
        }

        # Run the graph
        result = planner_graph.invoke(input_state)

        # Convert result to response format
        today_workout = result.get("today_workout")
        if today_workout is not None and not isinstance(today_workout, Workout):
            today_workout = Workout(**today_workout)

        return PlannerResponse(today_workout=today_workout)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


@app.post("/api/professional", response_model=ProfessionalResponse)
async def professional(request: ProfessionalRequest):
    """
    Endpoint for professional trainer workflow.
    Takes a user question and returns a professional answer if the question
    is about exercise information that a trainer can answer.
    """
    try:
        # Convert request to the format expected by the graph
        input_state = {"messages": [msg.model_dump() for msg in request.messages]}

        # Run the graph
        result = professional_graph.invoke(input_state)

        # Convert result to response format
        return ProfessionalResponse(
            messages=[ProfessionalMessageDict(**msg) for msg in result["messages"]],
            is_answerable=result["is_answerable"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

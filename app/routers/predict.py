"""
app/routers/predict.py
Prediction form (GET) and processing (POST) — 30 features matching real dataset.
"""
import logging
from fastapi import APIRouter, Depends, Form, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models.prediction import Prediction
from app.models.user import User
from app.routers.auth import get_current_user
from app.ml import predictor
from app.ml.features import FEATURE_OPTIONS, FEATURE_DISPLAY, MODEL_REGISTRY

logger = logging.getLogger(__name__)
router = APIRouter(tags=["predict"])
templates = Jinja2Templates(directory="app/templates")


@router.get("/predict", response_class=HTMLResponse,
            summary="Prediction form",
            description="Renders the 30-field accident feature form with model selector.")
async def predict_form(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse("predict.html", {
        "request": request, "user": current_user,
        "feature_options": FEATURE_OPTIONS,
        "feature_display": FEATURE_DISPLAY,
        "model_registry": MODEL_REGISTRY,
        "demo_mode": predictor.is_demo_mode(),
    })


@router.post("/predict", summary="Process prediction",
             description="Runs ML inference and redirects to /result/{id}.")
async def predict_submit(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    Day_of_week: str = Form(...),
    Age_band_of_driver: str = Form(...),
    Sex_of_driver: str = Form(...),
    Educational_level: str = Form(...),
    Vehicle_driver_relation: str = Form(...),
    Driving_experience: str = Form(...),
    Type_of_vehicle: str = Form(...),
    Owner_of_vehicle: str = Form(...),
    Service_year_of_vehicle: str = Form(...),
    Defect_of_vehicle: str = Form(...),
    Area_accident_occured: str = Form(...),
    Lanes_or_Medians: str = Form(...),
    Road_allignment: str = Form(...),
    Types_of_Junction: str = Form(...),
    Road_surface_type: str = Form(...),
    Road_surface_conditions: str = Form(...),
    Light_conditions: str = Form(...),
    Weather_conditions: str = Form(...),
    Type_of_collision: str = Form(...),
    Number_of_vehicles_involved: str = Form(...),
    Number_of_casualties: str = Form(...),
    Vehicle_movement: str = Form(...),
    Casualty_class: str = Form(...),
    Sex_of_casualty: str = Form(...),
    Age_band_of_casualty: str = Form(...),
    Casualty_severity: str = Form(...),
    Work_of_casuality: str = Form(...),
    Fitness_of_casuality: str = Form(...),
    Pedestrian_movement: str = Form(...),
    Cause_of_accident: str = Form(...),
    model_key: str = Form(default="gb"),
):
    raw_inputs = {
        "Day_of_week": Day_of_week, "Age_band_of_driver": Age_band_of_driver,
        "Sex_of_driver": Sex_of_driver, "Educational_level": Educational_level,
        "Vehicle_driver_relation": Vehicle_driver_relation, "Driving_experience": Driving_experience,
        "Type_of_vehicle": Type_of_vehicle, "Owner_of_vehicle": Owner_of_vehicle,
        "Service_year_of_vehicle": Service_year_of_vehicle, "Defect_of_vehicle": Defect_of_vehicle,
        "Area_accident_occured": Area_accident_occured, "Lanes_or_Medians": Lanes_or_Medians,
        "Road_allignment": Road_allignment, "Types_of_Junction": Types_of_Junction,
        "Road_surface_type": Road_surface_type, "Road_surface_conditions": Road_surface_conditions,
        "Light_conditions": Light_conditions, "Weather_conditions": Weather_conditions,
        "Type_of_collision": Type_of_collision,
        "Number_of_vehicles_involved": Number_of_vehicles_involved,
        "Number_of_casualties": Number_of_casualties, "Vehicle_movement": Vehicle_movement,
        "Casualty_class": Casualty_class, "Sex_of_casualty": Sex_of_casualty,
        "Age_band_of_casualty": Age_band_of_casualty, "Casualty_severity": Casualty_severity,
        "Work_of_casuality": Work_of_casuality, "Fitness_of_casuality": Fitness_of_casuality,
        "Pedestrian_movement": Pedestrian_movement, "Cause_of_accident": Cause_of_accident,
    }

    if model_key not in MODEL_REGISTRY:
        model_key = "gb"

    result = predictor.predict(raw_inputs, model_key=model_key)

    pred = Prediction(
        user_id=current_user.id,
        severity_label=result["severity_label"],
        severity_code=result["severity_code"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        shap_values=result["shap_values"],
        inputs=raw_inputs,
        model_key=model_key,
        cause_of_accident=Cause_of_accident,
        weather_conditions=Weather_conditions,
    )
    db.add(pred)
    await db.commit()
    await db.refresh(pred)

    logger.info("Prediction #%d: %s (%.1f%%) user=%s model=%s",
                pred.id, result["severity_label"], result["confidence"]*100,
                current_user.email, model_key)

    return RedirectResponse(url=f"/result/{pred.id}", status_code=status.HTTP_303_SEE_OTHER)

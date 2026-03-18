"""
app/ml/features.py
Single source of truth for all feature definitions.
Generated from actual RTA_Dataset.csv - all values are exact matches.
"""
from pathlib import Path

FEATURE_ORDER = [
    "Day_of_week","Age_band_of_driver","Sex_of_driver","Educational_level",
    "Vehicle_driver_relation","Driving_experience","Type_of_vehicle",
    "Owner_of_vehicle","Service_year_of_vehicle","Defect_of_vehicle",
    "Area_accident_occured","Lanes_or_Medians","Road_allignment",
    "Types_of_Junction","Road_surface_type","Road_surface_conditions",
    "Light_conditions","Weather_conditions","Type_of_collision",
    "Number_of_vehicles_involved","Number_of_casualties","Vehicle_movement",
    "Casualty_class","Sex_of_casualty","Age_band_of_casualty",
    "Casualty_severity","Work_of_casuality","Fitness_of_casuality",
    "Pedestrian_movement","Cause_of_accident",
]

FEATURE_DISPLAY = {
    "Day_of_week":"Day of Week","Age_band_of_driver":"Driver Age Band",
    "Sex_of_driver":"Driver Sex","Educational_level":"Education Level",
    "Vehicle_driver_relation":"Driver-Vehicle Relation","Driving_experience":"Driving Experience",
    "Type_of_vehicle":"Vehicle Type","Owner_of_vehicle":"Vehicle Owner",
    "Service_year_of_vehicle":"Vehicle Service Year","Defect_of_vehicle":"Vehicle Defect",
    "Area_accident_occured":"Accident Area","Lanes_or_Medians":"Lanes / Medians",
    "Road_allignment":"Road Alignment","Types_of_Junction":"Junction Type",
    "Road_surface_type":"Road Surface Type","Road_surface_conditions":"Road Surface Conditions",
    "Light_conditions":"Light Conditions","Weather_conditions":"Weather Conditions",
    "Type_of_collision":"Collision Type","Number_of_vehicles_involved":"Vehicles Involved",
    "Number_of_casualties":"Casualties","Vehicle_movement":"Vehicle Movement",
    "Casualty_class":"Casualty Class","Sex_of_casualty":"Casualty Sex",
    "Age_band_of_casualty":"Casualty Age Band","Casualty_severity":"Casualty Severity",
    "Work_of_casuality":"Casualty Occupation","Fitness_of_casuality":"Casualty Fitness",
    "Pedestrian_movement":"Pedestrian Movement","Cause_of_accident":"Cause of Accident",
}

FEATURE_OPTIONS = {
    "Day_of_week":["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
    "Age_band_of_driver":["18-30","31-50","Over 51","Under 18","Unknown"],
    "Sex_of_driver":["Male","Female","Unknown"],
    "Educational_level":["Above high school","High school","Junior high school","Elementary school","Writing & reading","Illiterate","Unknown"],
    "Vehicle_driver_relation":["Employee","Owner","Other","Unknown"],
    "Driving_experience":["Below 1yr","1-2yr","2-5yr","5-10yr","Above 10yr","No Licence","unknown"],
    "Type_of_vehicle":["Automobile","Taxi","Motorcycle","Bajaj","Bicycle","Pick up upto 10Q","Stationwagen","Public (12 seats)","Public (13?45 seats)","Public (> 45 seats)","Lorry (11?40Q)","Lorry (41?100Q)","Long lorry","Ridden horse","Special vehicle","Turbo","Other"],
    "Owner_of_vehicle":["Owner","Governmental","Organization","Other"],
    "Service_year_of_vehicle":["Below 1yr","1-2yr","2-5yrs","5-10yrs","Above 10yr","Unknown"],
    "Defect_of_vehicle":["No defect","5","7"],
    "Area_accident_occured":["Residential areas","Office areas","School areas","  Market areas","  Recreational areas","Recreational areas"," Church areas"," Hospital areas"," Industrial areas"," Outside rural areas","Rural village areas","Other","Unknown"],
    "Lanes_or_Medians":["Undivided Two way","Double carriageway (median)","One way","Two-way (divided with solid lines road marking)","Two-way (divided with broken lines road marking)","Unknown","other"],
    "Road_allignment":["Tangent road with flat terrain","Tangent road with mild grade and flat terrain","Tangent road with rolling terrain","Tangent road with mountainous terrain and","Gentle horizontal curve","Sharp reverse curve","Steep grade downward with mountainous terrain","Steep grade upward with mountainous terrain","Escarpments"],
    "Types_of_Junction":["No junction","Y Shape","T Shape","X Shape","O Shape","Crossing","Other","Unknown"],
    "Road_surface_type":["Asphalt roads","Asphalt roads with some distress","Earth roads","Gravel roads","Other"],
    "Road_surface_conditions":["Dry","Wet or damp","Snow","Flood over 3cm. deep"],
    "Light_conditions":["Daylight","Darkness - lights lit","Darkness - no lighting","Darkness - lights unlit"],
    "Weather_conditions":["Normal","Raining","Raining and Windy","Cloudy","Windy","Snow","Fog or mist","Unknown","Other"],
    "Type_of_collision":["Vehicle with vehicle collision","Collision with roadside-parked vehicles","Collision with roadside objects","Collision with pedestrians","Collision with animals","Rollover","Fall from vehicles","With Train","Unknown","Other"],
    "Number_of_vehicles_involved":["1","2","3","4","6","7"],
    "Number_of_casualties":["1","2","3","4","5","6","7","8"],
    "Vehicle_movement":["Going straight","Overtaking","Moving Backward","Reversing","U-Turn","Turnover","Stopping","Parked","Getting off","Entering a junction","Waiting to go","Unknown","Other"],
    "Casualty_class":["Driver or rider","Passenger","Pedestrian","na"],
    "Sex_of_casualty":["Male","Female","na"],
    "Age_band_of_casualty":["18-30","31-50","Over 51","Under 18","5","na"],
    "Casualty_severity":["1","2","3","na"],
    "Work_of_casuality":["Driver","Employee","Self-employed","Student","Unemployed","Other","Unknown"],
    "Fitness_of_casuality":["Normal","NormalNormal","Deaf","Blind","Other"],
    "Pedestrian_movement":["Not a Pedestrian","Crossing from driver's nearside","Crossing from nearside - masked by parked or statioNot a Pedestrianry vehicle","Crossing from offside - masked by  parked or statioNot a Pedestrianry vehicle","In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing)","In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing) - masked by parked or statioNot a Pedestrianry vehicle","Walking along in carriageway, back to traffic","Walking along in carriageway, facing traffic","Unknown or other"],
    "Cause_of_accident":["Moving Backward","Overtaking","Overspeed","Changing lane to the left","Changing lane to the right","No distancing","No priority to vehicle","No priority to pedestrian","Driving carelessly","Driving at high speed","Driving to the left","Driving under the influence of drugs","Drunk driving","Getting off the vehicle improperly","Improper parking","Overloading","Overturning","Turnover","Other","Unknown"],
}

SEVERITY_LABELS = {0:"Slight Injury",1:"Serious Injury",2:"Fatal injury"}
SEVERITY_COLORS = {"Slight Injury":"green","Serious Injury":"amber","Fatal injury":"red"}

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

MODEL_REGISTRY = {
    "gb":   {"name":"Gradient Boosting",     "file":ARTIFACTS_DIR/"model_gb.pkl",   "unit":"III","type":"Ensemble",     "default":True},
    "rf":   {"name":"Random Forest",          "file":ARTIFACTS_DIR/"model_rf.pkl",   "unit":"III","type":"Ensemble",     "default":False},
    "xgb":  {"name":"XGBoost",                "file":ARTIFACTS_DIR/"model_xgb.pkl",  "unit":"III","type":"Ensemble",     "default":False},
    "lgbm": {"name":"LightGBM",               "file":ARTIFACTS_DIR/"model_lgbm.pkl", "unit":"III","type":"Ensemble",     "default":False},
    "knn":  {"name":"k-Nearest Neighbors",    "file":ARTIFACTS_DIR/"model_knn.pkl",  "unit":"III","type":"Instance",     "default":False},
    "nb":   {"name":"Naive Bayes",            "file":ARTIFACTS_DIR/"model_nb.pkl",   "unit":"III","type":"Probabilistic","default":False},
    "svm":  {"name":"Support Vector Machine", "file":ARTIFACTS_DIR/"model_svm.pkl",  "unit":"III","type":"Kernel",       "default":False},
    "mlp":  {"name":"MLP Neural Network",     "file":ARTIFACTS_DIR/"model_mlp.pkl",  "unit":"IV", "type":"Neural Net",   "default":False},
    "lr":   {"name":"Logistic Regression",    "file":ARTIFACTS_DIR/"model_lr.pkl",   "unit":"III","type":"Linear",       "default":False},
    "dt":   {"name":"Decision Tree",          "file":ARTIFACTS_DIR/"model_dt.pkl",   "unit":"III","type":"Tree",         "default":False},
    "ridge":{"name":"Ridge Regression",       "file":ARTIFACTS_DIR/"model_ridge.pkl","unit":"II", "type":"Regression",   "default":False},
    "lasso":{"name":"Lasso Regression",       "file":ARTIFACTS_DIR/"model_lasso.pkl","unit":"II", "type":"Regression",   "default":False},
}

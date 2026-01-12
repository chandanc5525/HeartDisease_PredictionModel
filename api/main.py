from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from api.model_loader import load_model
from api.predict import predict_heart_disease

# Initialize FastAPI app
app = FastAPI(title="Heart Disease Prediction API")

# Mount static folder for CSS
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Load trained model once at startup
model = load_model("artifacts/models/model.pkl")

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("api/templates/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    Age: int = Form(...),
    Sex: int = Form(...),
    cp: int = Form(...),
    trestbps: int = Form(...),
    chol: int = Form(...),
    fbs: int = Form(...),
    restecg: int = Form(...),
    thalach: int = Form(...),
    exang: int = Form(...),
    oldpeak: float = Form(...)
):
    # Prepare data in correct order
    input_data = [Age, Sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]

    try:
        prediction = predict_heart_disease(model, input_data)
        result = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"
    except Exception as e:
        result = f"Prediction error: {str(e)}"

    # Render index.html again with prediction
    with open("api/templates/index.html", "r") as f:
        html_content = f.read()
    html_content = html_content.replace("{{ prediction }}", result)
    return HTMLResponse(content=html_content)

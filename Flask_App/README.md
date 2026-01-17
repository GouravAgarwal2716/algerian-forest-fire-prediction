# Flask ML Deployment - Algerian Forest Fires Prediction

A Flask web application to deploy a machine learning model for predicting Algerian forest fire weather indices.

## Project Structure

```
Flask_App/
├── app.py                  # Main Flask application
├── config.py              # Configuration settings
├── export_model.py        # Script to export trained models
├── requirements.txt       # Python dependencies
├── Procfile              # Heroku deployment configuration
├── .gitignore            # Git ignore file
├── templates/
│   └── index.html        # Main web interface
├── static/
│   └── css/
│       └── style.css     # Styling
└── models/               # Directory for trained models (*.pkl files)
    ├── scaler.pkl        # Fitted StandardScaler
    └── lasso_cv_model.pkl # Trained LassoCV model
```

## Features

- **Web Interface**: User-friendly form to input forest fire features
- **Real-time Predictions**: Get FWI (Fire Weather Index) predictions instantly
- **API Endpoints**: RESTful API for programmatic predictions
- **Model Deployment**: Ready for Heroku, Docker, or other cloud platforms

## Installation

### 1. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Models

First, train your model in the Jupyter notebook and export it:

```python
# In your notebook after training
from export_model import export_models
export_models(lasso_cv, scaler)
```

This will create:
- `models/scaler.pkl` - Fitted StandardScaler
- `models/lasso_cv_model.pkl` - Trained model

## Running the Application

### Local Development

```bash
python app.py
```

Visit: `http://localhost:5000`

### Production (Heroku)

```bash
# Install Heroku CLI
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Deploy
git push heroku main

# View logs
heroku logs --tail
```

### Docker Deployment

Create a `Dockerfile` in the Flask_App directory:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

Build and run:

```bash
docker build -t forest-fire-predictor .
docker run -p 5000:5000 forest-fire-predictor
```

## API Usage

### Prediction Endpoint

**POST** `/predict`

Request body (JSON):
```json
{
  "Region": 0,
  "FFMC": 50,
  "DMC": 100,
  "DC": 200,
  "ISI": 10,
  "BUI": 150,
  "FWI": 50,
  "Rain": 0,
  "Temperature": 25,
  "RH": 60,
  "Ws": 15
}
```

Response:
```json
{
  "success": true,
  "prediction": 45.32,
  "message": "Predicted Fire Weather Index (FWI): 45.32"
}
```

### Features Endpoint

**GET** `/api/features`

Returns information about all input features and their valid ranges.

## Feature Descriptions

- **Region**: Region code (0 or 1)
- **FFMC**: Fine Fuel Moisture Code (0-100)
- **DMC**: Duff Moisture Code (0-300)
- **DC**: Drought Code (0-1000)
- **ISI**: Initial Spread Index (0-60)
- **BUI**: Buildup Index (0-300)
- **FWI**: Fire Weather Index (0-100)
- **Rain**: Rainfall in mm (0-100)
- **Temperature**: Temperature in °C (2-40)
- **RH**: Relative Humidity in % (15-100)
- **Ws**: Wind Speed in km/h (0-40)

## Model Information

- **Algorithm**: LassoCV (Cross-validated Lasso Regression)
- **Dataset**: Algerian Forest Fires
- **Target Variable**: Fire Weather Index (FWI)
- **Features Standardized**: Yes (StandardScaler)

## Environment Variables

Create a `.env` file:

```env
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your-secret-key
```

## Troubleshooting

### Models not loading
- Ensure model files exist in the `models/` directory
- Check file permissions
- Verify pickle files are valid

### Port already in use
```bash
# Change port in app.py
app.run(port=5001)

# Or kill process on port 5000
lsof -ti:5000 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :5000   # Windows
```

### Import errors
```bash
pip install --upgrade -r requirements.txt
```

## Contributing

1. Create a feature branch
2. Make your changes
3. Test locally
4. Submit a pull request

## License

This project is for educational purposes.

## Support

For issues and questions, please open an issue on the repository.

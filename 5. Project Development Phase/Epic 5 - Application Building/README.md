# Epic 5: Application Building

## Story: Building HTML Pages
- `templates/index.html` — loan application input form
- `templates/result.html` — displays the prediction result

## Story: Building the Python Script
- `app.py` — Flask backend that loads the trained model and serves predictions

## Story: Run the Application

```bash
cd "5. Project Development Phase/Epic 5 - Application Building"
pip install flask scikit-learn numpy
python app.py
```

Then open `http://127.0.0.1:5000/` in your browser.

> Note: `model.pkl` (trained model from Epic 4) must be present in this folder for the app to run.

from flask import Flask, request, render_template_string
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the Linear Model
# Ensure 'linear_model.pkl' is in the same directory as this script
MODEL_PATH = 'linear_model.pkl'
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    model = None
    print(f"Warning: Model not found or failed to load. Error: {e}")

# Embedded HTML and CSS template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Linear Model Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #0d9488; /* Teal */
            --primary-hover: #0f766e;
            --secondary: #2563eb; /* Ocean Blue */
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-dark: #0f172a;
            --text-muted: #64748b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Poppins', sans-serif;
        }

        body {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }

        .card {
            background-color: var(--card-bg);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
            width: 100%;
            max-width: 550px;
        }

        .header {
            text-align: center;
            margin-bottom: 35px;
        }

        .header h1 {
            font-size: 28px;
            font-weight: 600;
            color: var(--text-dark);
            margin-bottom: 10px;
        }

        .header p {
            font-size: 15px;
            color: var(--text-muted);
        }

        .form-group {
            margin-bottom: 25px;
        }

        label {
            display: block;
            font-size: 14px;
            font-weight: 500;
            color: var(--text-dark);
            margin-bottom: 8px;
        }

        input[type="text"] {
            width: 100%;
            padding: 15px;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            font-size: 15px;
            transition: all 0.3s ease;
        }

        input[type="text"]:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(13, 148, 136, 0.15);
        }

        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(to right, var(--primary), var(--secondary));
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(37, 99, 235, 0.3);
        }

        .result-container {
            margin-top: 30px;
            animation: slideUp 0.4s ease-out;
        }

        .result-box {
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 18px;
            font-weight: 600;
        }

        .success {
            background-color: #f0fdf4;
            color: #166534;
            border: 1px solid #bbf7d0;
        }

        .error {
            background-color: #fef2f2;
            color: #991b1b;
            border: 1px solid #fecaca;
        }

        .prediction-value {
            display: block;
            font-size: 28px;
            margin-top: 10px;
            color: var(--primary);
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="card">
    <div class="header">
        <h1>Linear Model Predictor</h1>
        <p>Enter your numerical features below to get a prediction.</p>
    </div>

    <form action="/predict" method="POST">
        <div class="form-group">
            <label for="features">Input Features (comma-separated):</label>
            <input type="text" id="features" name="features" placeholder="e.g., 12.5, 45.2, 3.1" required value="{{ request.form['features'] if request.form else '' }}">
        </div>
        <button type="submit">Generate Prediction</button>
    </form>

    {% if prediction_text is defined %}
        <div class="result-container">
            <div class="result-box success">
                Prediction Result:
                <span class="prediction-value">{{ prediction_text }}</span>
            </div>
        </div>
    {% endif %}

    {% if error_text %}
        <div class="result-container">
            <div class="result-box error">
                {{ error_text }}
            </div>
        </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, error_text="System Error: 'linear_model.pkl' could not be loaded.")

    try:
        # Extract features from the form
        input_string = request.form.get('features')
        
        # Convert comma-separated string to a list of floats
        features_list = [float(x.strip()) for x in input_string.split(',')]
        
        # Convert to 2D numpy array for the model
        features_array = np.array([features_list])
        
        # Make the prediction
        prediction = model.predict(features_array)
        
        # Round the result for a cleaner display (optional, based on your model's output)
        result = round(float(prediction[0]), 4)
        
        return render_template_string(HTML_TEMPLATE, prediction_text=result)
        
    except ValueError:
        return render_template_string(HTML_TEMPLATE, error_text="Input Error: Please make sure you only enter numbers separated by commas.")
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error_text=f"Processing Error: {str(e)}")

if __name__ == "__main__":
    # Bind to Render's PORT environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

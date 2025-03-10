from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import pandas as pd
import joblib
import openai
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load OpenAI API Key (Insert your OpenAI API Key here)
openai.api_key = ""
# Load Dataset
df = pd.read_csv("Folds5x2_pp.csv")

# Define Features and Target
X = df[['AT', 'V', 'AP', 'RH']]
y = df['PE']

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save & Load Model
joblib.dump(model, "energy_model.pkl")
model = joblib.load("energy_model.pkl")


# Home Page
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/result")
def result():
    prediction = request.args.get("prediction")
    cost = request.args.get("cost")
    error = request.args.get("error")
    
    return render_template("result.html", prediction=prediction, cost=cost, error=error)


# Prediction Route
@app.route("/predict", methods=["GET", "POST"])  # ✅ Allow GET & POST
def predict():
    if request.method == "POST":  # ✅ Process POST requests for prediction
        try:
            inputs = []
            for key in ["AT", "V", "AP", "RH"]:
                value = request.form[key].strip()
                try:
                    inputs.append(float(value))
                except ValueError:
                    return redirect(url_for("result", error="Invalid input! Enter valid numbers."))

            # ✅ Predict energy output
            prediction = round(abs(model.predict([inputs])[0]), 2)

            # ✅ Convert Cost to USD
            base_cost_per_mw = 75  # Adjust as needed
            cost_usd = round(prediction * base_cost_per_mw, 2)

            return redirect(url_for("result", prediction=prediction, cost=cost_usd))

        except Exception:
            return redirect(url_for("result", error="⚠ Something went wrong! Try again."))

    return render_template("predict.html")  # ✅ Show the prediction page when accessed via GET


# Comparison Route
@app.route("/comparison", methods=["GET", "POST"])
def comparison():
    if request.method == "POST":
        try:
            # Get input values
            inputs1 = [float(request.form[key]) for key in ["AT1", "V1", "AP1", "RH1"]]
            inputs2 = [float(request.form[key]) for key in ["AT2", "V2", "AP2", "RH2"]]

            # Make predictions
            pred1 = abs(model.predict([inputs1])[0])
            pred2 = abs(model.predict([inputs2])[0])

            # ✅ More realistic energy cost calculation
            base_cost_per_mw = 75  # Adjusted pricing per MW
            price1 = round(pred1 * base_cost_per_mw, 2)
            price2 = round(pred2 * base_cost_per_mw, 2)

            return render_template("comparison.html", pred1=pred1, pred2=pred2, price1=price1, price2=price2)
        
        except:
            return render_template("comparison.html", error="Invalid Input! Please enter valid numbers.")

    return render_template("comparison.html")


@app.route("/maintenance")
def maintenance():
    return render_template("maintenance.html")


# Chat Page
@app.route("/chat")
def chat():
    return render_template("chat.html")


# Chatbot Response
@app.route("/chat_response", methods=["POST"])
def chat_response():
    try:
        # Get the user message from the JSON payload
        user_message = request.json.get("message", "")
        if not user_message:
            return jsonify({"response": "⚠ Please enter a message!"})

        # Send the message to OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an AI expert on energy."},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=100
        )

        # Print the full API response for debugging
        print(f"API Response: {response}")

        # Extract the content of the message from the response using response["choices"][0]["message"]["content"]
        if 'choices' in response and len(response['choices']) > 0:
            message_content = response["choices"][0]["message"]["content"]
            return jsonify({"response": message_content})
        else:
            return jsonify({"response": "⚠ No valid response from AI."})

    except openai.error.AuthenticationError:
        return jsonify({"response": "⚠ Invalid API Key! Please check your OpenAI API key."})

    except openai.error.OpenAIError as e:
        return jsonify({"response": f"⚠ OpenAI Error: {str(e)}"})

    except Exception as e:
        return jsonify({"response": f"⚠ Unexpected Error: {str(e)}"})



if __name__ == "__main__":
    app.run(debug=True)

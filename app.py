from flask import Flask, render_template, request, redirect, url_for, flash, session
from pymongo import MongoClient
from bson.objectid import ObjectId
import hashlib  # For password hashing
from datetime import datetime
from flask import jsonify   
import os
import tensorflow as tf 
import numpy as np   
import pandas as pd
from joblib import load
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

from werkzeug.utils import secure_filename
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import ResNet9 
# Initialize Flask app
app = Flask(__name__)
app.secret_key = '9995b2c0be621c81b6c7d318af4e8842f645a7a2fcc406efc7b3126ff257c372c535beedf33230916370ccf13580b2e39847d8368cdda537db7eef3f03b15024210e239a2b825b7aba3151b4ca0e1f5c38e726e0d97fe41439e0782630118ea67690f07a56e37f8006180344ff995b73966d3b823e562ea0b125c3acd72e87acb821bcd7d6d45bb5910de82e1649a5d367ac79202775701671f2e52017d9b1d81863ddbfc451051dc09645062adc051a63921273432e586ee8b1e91f3e00d70cc7587caf91779080bb9c7e498a3f1e37a54a63d6ec66da9f3a1d7bd28f3ed04342c21125a2afeedde19d21fd665b936fb5632a5cee53fe7677482b19c8e136e0'  # Required for sessions and flash messages

# Connect to MongoDB
client = MongoClient('localhost', 27017)  # MongoDB running on localhost:27017
db = client['crop_prediction_db']  # Database name
users_collection = db['users']  # Collection for storing user data
predictions_collection = db['predictions']  # Collection for storing prediction history
plants_collection = db['plants']  # Collection for storing plant registration data
updates_collection = db['plant_updates']  # Collection for storing daily updates

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
loaded_model = load('random_forest_model.joblib')
# Load the trained model for disease detection
model_path = "plant_disease_model.pth"
num_classes = 38  # Adjust according to your dataset
model = ResNet9(3, num_classes)
checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])
class_names = checkpoint["class_names"]
model.eval()


# Define image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, dim=1)

    return class_names[predicted.item()]
@app.route("/disease-detection", methods=["GET", "POST"])
def disease_detection():
    # Check if the user is logged in
    if not session.get('logged_in'):
        flash('You need to log in to access this page.', 'error')
        return redirect(url_for('login'))  # Redirect to login if not logged in

    if request.method == "POST":
        # Handle file upload
        if "file" not in request.files:
            flash("No file uploaded", "error")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No selected file", "error")
            return redirect(request.url)

        # Save the uploaded file securely
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Make a prediction
        predicted_class = predict_image(file_path)

        return render_template("disease_detection.html", uploaded_image=file_path, prediction=predicted_class, logged_in=True)

    return render_template("disease_detection.html", uploaded_image=None, prediction=None, logged_in=True)

# Home route
@app.route('/')
def home():
    return render_template('home.html', logged_in=session.get('logged_in', False))

# User Profile route
@app.route('/profile')
def profile():
    # Check if the user is logged in
    if not session.get('logged_in'):
        flash('You need to log in to access this page.', 'error')
        return redirect(url_for('login'))  # Redirect to login if not logged in

    # Fetch the user's registered plants
    plants = list(plants_collection.find({'username': session['username']}).sort('timestamp', -1))

    # Fetch daily updates for each plant
    plant_updates = {}
    for plant in plants:
        updates = list(updates_collection.find({'plant_id': plant['plant_id']}).sort('timestamp', -1))
        plant_updates[plant['plant_id']] = updates

    # Render the profile page
    return render_template('profile.html', logged_in=session.get('logged_in', False), plants=plants, plant_updates=plant_updates)
# Daily Update route
@app.route('/update-plant/<plant_id>', methods=['GET', 'POST'])
def update_plant(plant_id):
    # Check if the user is logged in
    if not session.get('logged_in'):
        flash('You need to log in to access this page.', 'error')
        return redirect(url_for('login'))  # Redirect to login if not logged in

    # Fetch the plant details
    plant = plants_collection.find_one({'plant_id': plant_id})

    if request.method == 'POST':
        # Get user input from the form
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])

        # Save the daily update
        updates_collection.insert_one({
            'plant_id': plant_id,
            'temperature': temperature,
            'humidity': humidity,
            'N': N,
            'P': P,
            'K': K,
            'timestamp': datetime.utcnow()
        })

        flash('Daily update submitted successfully!', 'success')
        return redirect(url_for('profile'))  # Redirect back to the profile page

    # Render the update form for GET requests
    return render_template('update_plant.html', logged_in=session.get('logged_in', False), plant=plant)
@app.route('/get-updates/<plant_id>')
def get_updates(plant_id):
    # Fetch updates for the selected plant
    updates = list(updates_collection.find({'plant_id': plant_id}).sort('timestamp', -1))

    # Convert updates to a JSON-friendly format
    updates_json = []
    for update in updates:
        updates_json.append({
            'timestamp': update['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'temperature': update['temperature'],
            'humidity': update['humidity'],
            'N': update['N'],
            'P': update['P'],
            'K': update['K']
        })

    return jsonify(updates_json)
# Crop Prediction route
@app.route('/crop-prediction', methods=['GET', 'POST'])
def crop_prediction():
    # Check if the user is logged in
    if not session.get('logged_in'):
        flash('You need to log in to access this page.', 'error')
        return redirect(url_for('login'))  # Redirect to login if not logged in

    prediction = None
    if request.method == 'POST':
        # Get user input from the form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Create a dictionary from the input
        new_data = {
            'N': [N],
            'P': [P],
            'K': [K],
            'temperature': [temperature],
            'humidity': [humidity],
            'ph': [ph],
            'rainfall': [rainfall]
        }

        # Convert the dictionary to a DataFrame
        new_df = pd.DataFrame(new_data)

        # Predict the label for the new data
        prediction = loaded_model.predict(new_df)[0]

        # Save the prediction history
        predictions_collection.insert_one({
            'username': session['username'],
            'parameters': {
                'N': N,
                'P': P,
                'K': K,
                'temperature': temperature,
                'humidity': humidity,
                'ph': ph,
                'rainfall': rainfall
            },
            'prediction': prediction,
            'timestamp': datetime.utcnow()
        })

    # Fetch the user's prediction history
    history = list(predictions_collection.find({'username': session['username']}).sort('timestamp', -1))

    # Render the crop prediction page
    return render_template('crop_prediction.html', prediction=prediction, history=history, logged_in=session.get('logged_in', False))

# Plant Registration route
from bson import ObjectId  # For generating unique IDs

@app.route('/plant-registration', methods=['GET', 'POST'])
def plant_registration():
    # Check if the user is logged in
    if not session.get('logged_in'):
        flash('You need to log in to access this page.', 'error')
        return redirect(url_for('login'))  # Redirect to login if not logged in

    if request.method == 'POST':
        # Get user input from the form
        plant_name = request.form['plantName']
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])

        # Generate a unique Plant ID
        plant_id = str(ObjectId())

        # Save the plant registration data
        plants_collection.insert_one({
            'plant_id': plant_id,  # Unique Plant ID
            'username': session['username'],
            'plant_name': plant_name,
            'temperature': temperature,
            'humidity': humidity,
            'N': N,
            'P': P,
            'K': K,
            'timestamp': datetime.utcnow()
        })

        flash('Plant registered successfully!', 'success')
        return redirect(url_for('plant_registration'))  # Redirect back to the plant registration page

    # Fetch the user's registered plants
    plants = list(plants_collection.find({'username': session['username']}).sort('timestamp', -1))

    # Render the plant registration page for GET requests
    return render_template('plant_registration.html', logged_in=session.get('logged_in', False), plants=plants)

# About route
@app.route('/about')
def about():
    return render_template('about.html', logged_in=session.get('logged_in', False))

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Find the user in the database
        user = users_collection.find_one({'username': username})

        # Check if the user exists and the password is correct
        if user and user['password'] == hash_password(password):
            session['logged_in'] = True  # Set session variable
            session['username'] = username  # Store username in session
            flash('Login successful!', 'success')
            return redirect(url_for('home'))  # Redirect to home after successful login
        else:
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))  # Redirect back to login with an error message
    # Render the login page for GET requests
    return render_template('login.html', logged_in=session.get('logged_in', False))

# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        phone = request.form['phone']  # Get the phone number from the form

        # Check if passwords match
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('signup'))  # Redirect back to signup with an error message

        # Check if the username already exists
        if users_collection.find_one({'username': username}):
            flash('Username already exists', 'error')
            return redirect(url_for('signup'))  # Redirect back to signup with an error message

        # Check if the phone number already exists
        if users_collection.find_one({'phone': phone}):
            flash('Phone number already registered', 'error')
            return redirect(url_for('signup'))  # Redirect back to signup with an error message

        # Hash the password before storing it
        hashed_password = hash_password(password)

        # Insert the new user into the database
        users_collection.insert_one({
            'username': username,
            'email': email,
            'password': hashed_password,
            'phone': phone  # Store the phone number
        })

        flash('Signup successful! Please log in.', 'success')
        return redirect(url_for('login'))  # Redirect to login after successful signup

    # Render the signup page for GET requests
    return render_template('signup.html', logged_in=session.get('logged_in', False))
# Logout route
@app.route('/logout')
def logout():
    session.pop('logged_in', None)  # Clear session variable
    session.pop('username', None)  # Clear username from session
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))  # Redirect to home after logout

# Helper function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()
# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
from joblib import load
import pandas as pd

# Load the saved model
loaded_model = load('random_forest_model.joblib')

# Function to get valid numeric input within a specified range
def get_valid_input(prompt, min_value, max_value):
    while True:
        try:
            value = float(input(prompt))
            if min_value <= value <= max_value:
                return value
            else:
                print(f"Input must be between {min_value} and {max_value}. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# Take dynamic input from the user
print("Enter the following details:")
N = get_valid_input("Nitrogen (N) value: ", 0, 100)  # Assuming N is between 0 and 100
P = get_valid_input("Phosphorus (P) value: ", 0, 100)  # Assuming P is between 0 and 100
K = get_valid_input("Potassium (K) value: ", 0, 100)  # Assuming K is between 0 and 100
temperature = get_valid_input("Temperature value (must be below 30): ", -50, 30)  # Temperature below 30
humidity = get_valid_input("Humidity value (must be under 90): ", 0, 90)  # Humidity under 90
ph = get_valid_input("pH value (must be less than 14): ", 0, 14)  # pH less than 14
rainfall = get_valid_input("Rainfall value (must be under 300): ", 0, 300)  # Rainfall under 300

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
predicted_label = loaded_model.predict(new_df)

# Display the predicted label
print(f"\nPredicted Label: {predicted_label[0]}")
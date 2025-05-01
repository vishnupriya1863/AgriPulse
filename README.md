# AgriPulse Using CNN
Overview
AgriPulse is an integrated smart agriculture system that leverages IoT sensors and machine learning models to monitor crop health, detect diseases, and predict yields in real-time. The system provides farmers with actionable insights through a user-friendly web dashboard and automated alerts.​

Getting Started
Clone the Repository:

git clone https://github.com/vishnupriya1863/AgriPulse.git

Python : 3.11 version 

Install Dependencies: Follow the instructions in the requirements.txt file to install necessary packages.

Configure Sensors: Set up IoT sensors and ensure they are connected via the ESP8266 module.

Run the Application: Start the server and access the web dashboard to begin monitoring.


Features
Real-Time Monitoring: Continuous tracking of soil moisture, temperature, humidity, and pH levels using IoT sensors.

Disease Detection: Utilizes a Convolutional Neural Network (CNN) model to identify crop diseases from leaf images.

Yield Prediction: Employs Random Forest and XGBoost models to forecast crop production based on environmental parameters.

Automated Alerts: Sends notifications via WhatsApp and email when anomalies are detected, including crop ID and specific issues.

User Dashboard: Interactive web interface for monitoring crop status, viewing analytics, and managing alerts.​

System Architecture
The system comprises the following components:​

IoT Sensors: Collect environmental data from the field.

ESP8266 Wi-Fi Module: Transmits sensor data to the cloud.

Cloud Server: Processes data and hosts machine learning models.

Web Dashboard: Displays real-time data and analytics to the user.​

Results
Data Transmission: Achieved 99% uptime with reliable sensor data transmission to the cloud.

Disease Detection: CNN model reached 94% accuracy in identifying diseases like Leaf Spot, Blight, and Rust.

Yield Prediction: XGBoost model outperformed Random Forest with RMSE of 0.21 and MAE of 0.18.

Alert System: Successfully sent timely alerts with crop ID and issue details upon detecting anomalies.​


Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.​

License
This project is licensed under the MIT License. See the LICENSE file for details.​



"""
AI Models Module for Smart Farming Solution
Provides low-cost AI capabilities for smallholder farms including:
- Plant health monitoring using simple image classification
- Predictive analytics for crop yield and irrigation needs
- Pest and disease detection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import cv2
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PlantHealthClassifier:
    """
    Simple plant health classifier using traditional ML approaches
    Designed to work on edge devices with limited computational resources
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_features(self, image_path_or_array):
        """
        Extract simple features from plant images
        Uses basic color and texture features that can be computed quickly
        """
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
        else:
            image = image_path_or_array
            
        if image is None:
            return None
            
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Extract color features
        features = []
        
        # Mean and std of each channel in BGR, HSV, and LAB
        for img in [image, hsv, lab]:
            for channel in range(3):
                features.extend([
                    np.mean(img[:, :, channel]),
                    np.std(img[:, :, channel])
                ])
        
        # Green vegetation index (simple NDVI approximation)
        green = image[:, :, 1].astype(float)
        red = image[:, :, 2].astype(float)
        gvi = np.mean((green - red) / (green + red + 1e-8))
        features.append(gvi)
        
        # Texture features using Laplacian variance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features.append(laplacian_var)
        
        return np.array(features)
    
    def generate_synthetic_training_data(self, n_samples=1000):
        """
        Generate synthetic training data for demonstration
        In practice, this would be replaced with real labeled plant images
        """
        np.random.seed(42)
        
        # Simulate features for healthy and unhealthy plants
        healthy_features = []
        unhealthy_features = []
        
        for _ in range(n_samples // 2):
            # Healthy plants: higher green values, lower red values, higher texture variance
            healthy = np.random.normal([120, 25, 80, 20, 60, 15,  # BGR means/stds
                                      45, 15, 180, 30, 90, 20,    # HSV means/stds
                                      50, 10, 128, 25, 100, 15,   # LAB means/stds
                                      0.3, 1500], 
                                     [10, 5, 10, 5, 10, 5,
                                      5, 3, 20, 8, 15, 5,
                                      8, 3, 15, 5, 12, 4,
                                      0.1, 300])
            healthy_features.append(healthy)
            
            # Unhealthy plants: lower green values, higher red/brown values, lower texture
            unhealthy = np.random.normal([100, 30, 110, 25, 40, 18,  # BGR means/stds
                                        25, 12, 160, 35, 70, 25,     # HSV means/stds
                                        45, 12, 140, 30, 80, 18,     # LAB means/stds
                                        -0.1, 800],
                                       [15, 8, 15, 8, 12, 6,
                                        8, 4, 25, 10, 18, 8,
                                        10, 4, 20, 8, 15, 6,
                                        0.15, 200])
            unhealthy_features.append(unhealthy)
        
        X = np.vstack([healthy_features, unhealthy_features])
        y = np.hstack([np.ones(n_samples // 2), np.zeros(n_samples // 2)])  # 1=healthy, 0=unhealthy
        
        return X, y
    
    def train(self, X=None, y=None):
        """Train the plant health classifier"""
        if X is None or y is None:
            X, y = self.generate_synthetic_training_data()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        return accuracy
    
    def predict(self, image_path_or_array):
        """Predict plant health from image"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        features = self.extract_features(image_path_or_array)
        if features is None:
            return None
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return {
            'health_status': 'Healthy' if prediction == 1 else 'Unhealthy',
            'confidence': max(probability),
            'health_score': probability[1]  # Probability of being healthy
        }
    
    def save_model(self, filepath):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']


class CropYieldPredictor:
    """
    Predictive analytics for crop yield estimation
    Uses environmental and historical data to predict yields
    """
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic crop yield data"""
        np.random.seed(42)
        
        # Features: temperature, humidity, soil_moisture, rainfall, days_since_planting, fertilizer_amount
        features = []
        yields = []
        
        for _ in range(n_samples):
            temp = np.random.normal(25, 5)  # Temperature in Celsius
            humidity = np.random.normal(60, 15)  # Humidity percentage
            soil_moisture = np.random.normal(40, 10)  # Soil moisture percentage
            rainfall = np.random.exponential(5)  # Rainfall in mm
            days_planted = np.random.randint(30, 120)  # Days since planting
            fertilizer = np.random.normal(50, 15)  # Fertilizer amount
            
            # Simulate yield based on optimal conditions
            optimal_temp = 25
            optimal_humidity = 65
            optimal_moisture = 45
            
            # Calculate yield based on how close conditions are to optimal
            temp_factor = 1 - abs(temp - optimal_temp) / 20
            humidity_factor = 1 - abs(humidity - optimal_humidity) / 50
            moisture_factor = 1 - abs(soil_moisture - optimal_moisture) / 30
            rainfall_factor = min(rainfall / 10, 1)  # More rain is better up to a point
            time_factor = min(days_planted / 90, 1)  # Yield increases with time
            fertilizer_factor = min(fertilizer / 60, 1)  # Diminishing returns
            
            base_yield = 100  # Base yield in kg per hectare
            yield_multiplier = (temp_factor + humidity_factor + moisture_factor + 
                              rainfall_factor + time_factor + fertilizer_factor) / 6
            
            actual_yield = base_yield * yield_multiplier * np.random.normal(1, 0.1)
            actual_yield = max(actual_yield, 10)  # Minimum yield
            
            features.append([temp, humidity, soil_moisture, rainfall, days_planted, fertilizer])
            yields.append(actual_yield)
        
        return np.array(features), np.array(yields)
    
    def train(self, X=None, y=None):
        """Train the yield prediction model"""
        if X is None or y is None:
            X, y = self.generate_synthetic_data()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        self.is_trained = True
        return rmse
    
    def predict_yield(self, temperature, humidity, soil_moisture, rainfall, days_since_planting, fertilizer_amount):
        """Predict crop yield based on current conditions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        features = np.array([[temperature, humidity, soil_moisture, rainfall, 
                            days_since_planting, fertilizer_amount]])
        features_scaled = self.scaler.transform(features)
        
        predicted_yield = self.model.predict(features_scaled)[0]
        
        # Get feature importance for recommendations
        feature_names = ['Temperature', 'Humidity', 'Soil Moisture', 'Rainfall', 
                        'Days Since Planting', 'Fertilizer Amount']
        importance = dict(zip(feature_names, self.model.feature_importances_))
        
        return {
            'predicted_yield_kg_per_hectare': predicted_yield,
            'feature_importance': importance
        }


class IrrigationOptimizer:
    """
    AI-driven irrigation optimization system
    Recommends optimal irrigation schedules based on sensor data and weather forecasts
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=30, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def generate_irrigation_data(self, n_samples=1000):
        """Generate synthetic irrigation decision data"""
        np.random.seed(42)
        
        features = []
        decisions = []  # 0=no irrigation, 1=light irrigation, 2=heavy irrigation
        
        for _ in range(n_samples):
            soil_moisture = np.random.uniform(10, 80)  # Current soil moisture %
            temperature = np.random.normal(25, 8)  # Temperature
            humidity = np.random.normal(60, 20)  # Air humidity
            wind_speed = np.random.exponential(3)  # Wind speed
            forecast_rain = np.random.exponential(2)  # Predicted rain in next 24h
            crop_stage = np.random.randint(1, 5)  # Growth stage (1-4)
            
            # Decision logic
            if soil_moisture > 60:
                decision = 0  # No irrigation needed
            elif soil_moisture > 40:
                if forecast_rain > 5:
                    decision = 0  # Rain expected
                else:
                    decision = 1  # Light irrigation
            else:
                if forecast_rain > 10:
                    decision = 1  # Light irrigation despite rain forecast
                else:
                    decision = 2  # Heavy irrigation needed
            
            # Adjust for crop stage (flowering stage needs more water)
            if crop_stage == 3 and decision < 2 and soil_moisture < 50:
                decision = min(decision + 1, 2)
            
            features.append([soil_moisture, temperature, humidity, wind_speed, 
                           forecast_rain, crop_stage])
            decisions.append(decision)
        
        return np.array(features), np.array(decisions)
    
    def train(self, X=None, y=None):
        """Train the irrigation optimization model"""
        if X is None or y is None:
            X, y = self.generate_irrigation_data()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        return accuracy
    
    def recommend_irrigation(self, soil_moisture, temperature, humidity, wind_speed, 
                           forecast_rain, crop_stage):
        """Recommend irrigation action based on current conditions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        features = np.array([[soil_moisture, temperature, humidity, wind_speed, 
                            forecast_rain, crop_stage]])
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        irrigation_actions = ['No Irrigation', 'Light Irrigation', 'Heavy Irrigation']
        
        return {
            'recommended_action': irrigation_actions[prediction],
            'confidence': max(probabilities),
            'action_probabilities': dict(zip(irrigation_actions, probabilities))
        }


def initialize_all_models():
    """Initialize and train all AI models"""
    print("Initializing Smart Farming AI Models...")
    
    # Plant Health Classifier
    print("Training Plant Health Classifier...")
    plant_health = PlantHealthClassifier()
    health_accuracy = plant_health.train()
    print(f"Plant Health Classifier Accuracy: {health_accuracy:.3f}")
    
    # Crop Yield Predictor
    print("Training Crop Yield Predictor...")
    yield_predictor = CropYieldPredictor()
    yield_rmse = yield_predictor.train()
    print(f"Yield Predictor RMSE: {yield_rmse:.2f} kg/hectare")
    
    # Irrigation Optimizer
    print("Training Irrigation Optimizer...")
    irrigation_optimizer = IrrigationOptimizer()
    irrigation_accuracy = irrigation_optimizer.train()
    print(f"Irrigation Optimizer Accuracy: {irrigation_accuracy:.3f}")
    
    print("All models initialized successfully!")
    
    return {
        'plant_health': plant_health,
        'yield_predictor': yield_predictor,
        'irrigation_optimizer': irrigation_optimizer
    }


if __name__ == "__main__":
    # Test the models
    models = initialize_all_models()
    
    # Test yield prediction
    print("\n--- Testing Yield Prediction ---")
    yield_result = models['yield_predictor'].predict_yield(
        temperature=26, humidity=65, soil_moisture=45, 
        rainfall=8, days_since_planting=60, fertilizer_amount=55
    )
    print(f"Predicted Yield: {yield_result['predicted_yield_kg_per_hectare']:.1f} kg/hectare")
    
    # Test irrigation recommendation
    print("\n--- Testing Irrigation Recommendation ---")
    irrigation_result = models['irrigation_optimizer'].recommend_irrigation(
        soil_moisture=35, temperature=28, humidity=55, 
        wind_speed=4, forecast_rain=2, crop_stage=3
    )
    print(f"Irrigation Recommendation: {irrigation_result['recommended_action']}")
    print(f"Confidence: {irrigation_result['confidence']:.3f}")


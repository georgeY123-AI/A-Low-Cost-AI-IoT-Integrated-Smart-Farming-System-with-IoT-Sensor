"""
Remote Access Module for Smart Farming Solution
Provides secure remote access capabilities for farmers to monitor and control their IoT systems
Includes web interface, API endpoints, and secure tunneling capabilities
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
import os
import secrets
import hashlib
import jwt
from functools import wraps
import socket
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RemoteAccessServer:
    """Main server class for remote access to smart farming system"""
    
    def __init__(self, host='0.0.0.0', port=5000, secret_key=None):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = secret_key or secrets.token_hex(32)
        CORS(self.app)  # Enable CORS for all routes
        
        self.host = host
        self.port = port
        self.sensor_network = None
        self.ai_models = None
        self.is_running = False
        
        # Setup routes
        self.setup_routes()
        
        # Create templates directory and basic HTML templates
        self.create_web_interface()
    
    def setup_routes(self):
        """Setup Flask routes for the web interface and API"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/status')
        def api_status():
            """API endpoint to get system status"""
            try:
                status = {
                    'timestamp': datetime.now().isoformat(),
                    'system_status': 'online',
                    'sensor_network': None,
                    'ai_models': None
                }
                
                if self.sensor_network:
                    status['sensor_network'] = self.sensor_network.get_sensor_summary()
                
                if self.ai_models:
                    status['ai_models'] = {
                        'plant_health': self.ai_models['plant_health'].is_trained,
                        'yield_predictor': self.ai_models['yield_predictor'].is_trained,
                        'irrigation_optimizer': self.ai_models['irrigation_optimizer'].is_trained
                    }
                
                return jsonify(status)
            except Exception as e:
                logger.error(f"Error in api_status: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/sensors/readings')
        def api_sensor_readings():
            """Get recent sensor readings"""
            try:
                hours = request.args.get('hours', 24, type=int)
                
                if not self.sensor_network:
                    return jsonify({'error': 'Sensor network not initialized'}), 400
                
                df = self.sensor_network.get_recent_readings(hours=hours)
                
                if df.empty:
                    return jsonify({'readings': []})
                
                # Convert DataFrame to JSON-serializable format
                readings = []
                for _, row in df.iterrows():
                    readings.append({
                        'sensor_id': row['sensor_id'],
                        'sensor_type': row['sensor_type'],
                        'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                        'value': row['value'],
                        'unit': row['unit'],
                        'location': row['location'],
                        'battery_level': row['battery_level']
                    })
                
                return jsonify({'readings': readings})
            except Exception as e:
                logger.error(f"Error in api_sensor_readings: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/sensors/current')
        def api_current_readings():
            """Get current sensor readings"""
            try:
                if not self.sensor_network:
                    return jsonify({'error': 'Sensor network not initialized'}), 400
                
                readings = self.sensor_network.read_all_sensors()
                
                current_readings = []
                for reading in readings:
                    current_readings.append({
                        'sensor_id': reading.sensor_id,
                        'sensor_type': reading.sensor_type,
                        'timestamp': reading.timestamp.isoformat(),
                        'value': reading.value,
                        'unit': reading.unit,
                        'location': reading.location,
                        'battery_level': reading.battery_level
                    })
                
                return jsonify({'readings': current_readings})
            except Exception as e:
                logger.error(f"Error in api_current_readings: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/ai/predict_yield', methods=['POST'])
        def api_predict_yield():
            """AI endpoint for yield prediction"""
            try:
                if not self.ai_models or not self.ai_models['yield_predictor'].is_trained:
                    return jsonify({'error': 'Yield predictor not available'}), 400
                
                data = request.get_json()
                required_fields = ['temperature', 'humidity', 'soil_moisture', 'rainfall', 
                                 'days_since_planting', 'fertilizer_amount']
                
                for field in required_fields:
                    if field not in data:
                        return jsonify({'error': f'Missing field: {field}'}), 400
                
                result = self.ai_models['yield_predictor'].predict_yield(
                    temperature=data['temperature'],
                    humidity=data['humidity'],
                    soil_moisture=data['soil_moisture'],
                    rainfall=data['rainfall'],
                    days_since_planting=data['days_since_planting'],
                    fertilizer_amount=data['fertilizer_amount']
                )
                
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error in api_predict_yield: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/ai/irrigation_recommendation', methods=['POST'])
        def api_irrigation_recommendation():
            """AI endpoint for irrigation recommendations"""
            try:
                if not self.ai_models or not self.ai_models['irrigation_optimizer'].is_trained:
                    return jsonify({'error': 'Irrigation optimizer not available'}), 400
                
                data = request.get_json()
                required_fields = ['soil_moisture', 'temperature', 'humidity', 
                                 'wind_speed', 'forecast_rain', 'crop_stage']
                
                for field in required_fields:
                    if field not in data:
                        return jsonify({'error': f'Missing field: {field}'}), 400
                
                result = self.ai_models['irrigation_optimizer'].recommend_irrigation(
                    soil_moisture=data['soil_moisture'],
                    temperature=data['temperature'],
                    humidity=data['humidity'],
                    wind_speed=data['wind_speed'],
                    forecast_rain=data['forecast_rain'],
                    crop_stage=data['crop_stage']
                )
                
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error in api_irrigation_recommendation: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/system/info')
        def api_system_info():
            """Get system information"""
            try:
                info = {
                    'hostname': socket.gethostname(),
                    'local_ip': self.get_local_ip(),
                    'public_ip': self.get_public_ip(),
                    'port': self.port,
                    'uptime': self.get_uptime(),
                    'access_urls': {
                        'local': f'http://{self.get_local_ip()}:{self.port}',
                        'localhost': f'http://localhost:{self.port}'
                    }
                }
                return jsonify(info)
            except Exception as e:
                logger.error(f"Error in api_system_info: {e}")
                return jsonify({'error': str(e)}), 500
    
    def create_web_interface(self):
        """Create basic web interface templates"""
        templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
        os.makedirs(templates_dir, exist_ok=True)
        
        # Create dashboard.html
        dashboard_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Farming Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        .sensor-reading {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            border-bottom: 1px solid #eee;
        }
        .sensor-reading:last-child {
            border-bottom: none;
        }
        .value {
            font-weight: bold;
            color: #27ae60;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-online { background-color: #27ae60; }
        .status-offline { background-color: #e74c3c; }
        .btn {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            margin: 5px;
        }
        .btn:hover {
            background-color: #2980b9;
        }
        .loading {
            text-align: center;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Smart Farming Dashboard</h1>
        <p>Low-Cost IoT & AI Solution for Smallholder Farms</p>
        <div id="system-status">
            <span class="status-indicator status-offline"></span>
            <span id="status-text">Connecting...</span>
        </div>
    </div>

    <div class="dashboard-grid">
        <div class="card">
            <h3>System Status</h3>
            <div id="system-info" class="loading">Loading...</div>
        </div>

        <div class="card">
            <h3>Current Sensor Readings</h3>
            <div id="current-readings" class="loading">Loading...</div>
        </div>

        <div class="card">
            <h3>AI Predictions</h3>
            <div id="ai-predictions">
                <button class="btn" onclick="predictYield()">Predict Yield</button>
                <button class="btn" onclick="getIrrigationAdvice()">Irrigation Advice</button>
                <div id="ai-results"></div>
            </div>
        </div>

        <div class="card">
            <h3>Recent Trends</h3>
            <div id="trends" class="loading">Loading...</div>
        </div>
    </div>

    <script>
        let systemOnline = false;

        async function fetchData(url) {
            try {
                const response = await fetch(url);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                return await response.json();
            } catch (error) {
                console.error('Fetch error:', error);
                return null;
            }
        }

        async function updateSystemStatus() {
            const status = await fetchData('/api/status');
            const statusIndicator = document.querySelector('.status-indicator');
            const statusText = document.getElementById('status-text');
            
            if (status) {
                systemOnline = true;
                statusIndicator.className = 'status-indicator status-online';
                statusText.textContent = 'System Online';
                
                // Update system info
                const systemInfo = document.getElementById('system-info');
                systemInfo.innerHTML = `
                    <div class="sensor-reading">
                        <span>Timestamp:</span>
                        <span class="value">${new Date(status.timestamp).toLocaleString()}</span>
                    </div>
                    <div class="sensor-reading">
                        <span>Total Sensors:</span>
                        <span class="value">${status.sensor_network?.total_sensors || 0}</span>
                    </div>
                    <div class="sensor-reading">
                        <span>Active Sensors:</span>
                        <span class="value">${status.sensor_network?.active_sensors || 0}</span>
                    </div>
                `;
            } else {
                systemOnline = false;
                statusIndicator.className = 'status-indicator status-offline';
                statusText.textContent = 'System Offline';
                document.getElementById('system-info').innerHTML = '<div class="loading">System offline</div>';
            }
        }

        async function updateCurrentReadings() {
            if (!systemOnline) return;
            
            const readings = await fetchData('/api/sensors/current');
            const container = document.getElementById('current-readings');
            
            if (readings && readings.readings) {
                container.innerHTML = readings.readings.slice(0, 8).map(reading => `
                    <div class="sensor-reading">
                        <span>${reading.sensor_id} (${reading.sensor_type}):</span>
                        <span class="value">${reading.value} ${reading.unit}</span>
                    </div>
                `).join('');
            } else {
                container.innerHTML = '<div class="loading">No sensor data available</div>';
            }
        }

        async function updateTrends() {
            if (!systemOnline) return;
            
            const readings = await fetchData('/api/sensors/readings?hours=24');
            const container = document.getElementById('trends');
            
            if (readings && readings.readings) {
                const sensorTypes = {};
                readings.readings.forEach(reading => {
                    if (!sensorTypes[reading.sensor_type]) {
                        sensorTypes[reading.sensor_type] = [];
                    }
                    sensorTypes[reading.sensor_type].push(reading);
                });
                
                container.innerHTML = Object.keys(sensorTypes).slice(0, 5).map(type => `
                    <div class="sensor-reading">
                        <span>${type}:</span>
                        <span class="value">${sensorTypes[type].length} readings</span>
                    </div>
                `).join('');
            } else {
                container.innerHTML = '<div class="loading">No trend data available</div>';
            }
        }

        async function predictYield() {
            if (!systemOnline) return;
            
            // Use sample data for demonstration
            const sampleData = {
                temperature: 26,
                humidity: 65,
                soil_moisture: 45,
                rainfall: 8,
                days_since_planting: 60,
                fertilizer_amount: 55
            };
            
            try {
                const response = await fetch('/api/ai/predict_yield', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(sampleData)
                });
                
                const result = await response.json();
                document.getElementById('ai-results').innerHTML = `
                    <div class="sensor-reading">
                        <span>Predicted Yield:</span>
                        <span class="value">${result.predicted_yield_kg_per_hectare?.toFixed(1) || 'N/A'} kg/hectare</span>
                    </div>
                `;
            } catch (error) {
                document.getElementById('ai-results').innerHTML = '<div>Error predicting yield</div>';
            }
        }

        async function getIrrigationAdvice() {
            if (!systemOnline) return;
            
            // Use sample data for demonstration
            const sampleData = {
                soil_moisture: 35,
                temperature: 28,
                humidity: 55,
                wind_speed: 4,
                forecast_rain: 2,
                crop_stage: 3
            };
            
            try {
                const response = await fetch('/api/ai/irrigation_recommendation', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(sampleData)
                });
                
                const result = await response.json();
                document.getElementById('ai-results').innerHTML = `
                    <div class="sensor-reading">
                        <span>Recommendation:</span>
                        <span class="value">${result.recommended_action || 'N/A'}</span>
                    </div>
                    <div class="sensor-reading">
                        <span>Confidence:</span>
                        <span class="value">${(result.confidence * 100)?.toFixed(1) || 'N/A'}%</span>
                    </div>
                `;
            } catch (error) {
                document.getElementById('ai-results').innerHTML = '<div>Error getting irrigation advice</div>';
            }
        }

        // Initialize dashboard
        function initDashboard() {
            updateSystemStatus();
            updateCurrentReadings();
            updateTrends();
            
            // Update every 30 seconds
            setInterval(() => {
                updateSystemStatus();
                updateCurrentReadings();
                updateTrends();
            }, 30000);
        }

        // Start dashboard when page loads
        document.addEventListener('DOMContentLoaded', initDashboard);
    </script>
</body>
</html>
        '''
        
        # FIX: Add encoding='utf-8' to properly handle Unicode characters
        with open(os.path.join(templates_dir, 'dashboard.html'), 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
    
    def get_local_ip(self):
        """Get local IP address"""
        try:
            # Connect to a remote address to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except:
            return "127.0.0.1"
    
    def get_public_ip(self):
        """Get public IP address (simplified)"""
        try:
            import requests
            response = requests.get('https://api.ipify.org', timeout=5)
            return response.text.strip()
        except:
            return "Unknown"
    
    def get_uptime(self):
        """Get system uptime"""
        try:
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.readline().split()[0])
                return str(timedelta(seconds=int(uptime_seconds)))
        except:
            return "Unknown"
    
    def set_sensor_network(self, sensor_network):
        """Set the sensor network instance"""
        self.sensor_network = sensor_network
        logger.info("Sensor network connected to remote access server")
    
    def set_ai_models(self, ai_models):
        """Set the AI models instance"""
        self.ai_models = ai_models
        logger.info("AI models connected to remote access server")
    
    def start_server(self, debug=False):
        """Start the Flask server"""
        try:
            logger.info(f"Starting Smart Farming Remote Access Server...")
            logger.info(f"Local access: http://{self.get_local_ip()}:{self.port}")
            logger.info(f"Localhost access: http://localhost:{self.port}")
            
            self.is_running = True
            self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)
        except Exception as e:
            logger.error(f"Error starting server: {e}")
            self.is_running = False
    
    def stop_server(self):
        """Stop the Flask server"""
        self.is_running = False
        logger.info("Remote access server stopped")

class SecureTunnel:
    """Provides secure tunneling capabilities for remote access"""
    
    def __init__(self, local_port=5000):
        self.local_port = local_port
        self.tunnel_process = None
        self.tunnel_url = None
    
    def create_ngrok_tunnel(self):
        """Create ngrok tunnel for external access (requires ngrok installation)"""
        try:
            # Check if ngrok is available
            result = subprocess.run(['which', 'ngrok'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("ngrok not found. Install ngrok for external access.")
                return None
            
            # Start ngrok tunnel
            cmd = ['ngrok', 'http', str(self.local_port), '--log=stdout']
            self.tunnel_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait a moment for ngrok to start
            time.sleep(3)
            
            # Get tunnel URL (simplified - in practice, you'd parse ngrok's API)
            logger.info("Ngrok tunnel started. Check ngrok dashboard for public URL.")
            return "Check ngrok dashboard at http://localhost:4040"
            
        except Exception as e:
            logger.error(f"Error creating ngrok tunnel: {e}")
            return None
    
    def stop_tunnel(self):
        """Stop the tunnel"""
        if self.tunnel_process:
            self.tunnel_process.terminate()
            self.tunnel_process = None
            logger.info("Tunnel stopped")

def create_remote_access_system(sensor_network=None, ai_models=None, port=5000):
    """Create and configure the complete remote access system"""
    
    # Create server
    server = RemoteAccessServer(port=port)
    
    # Connect sensor network and AI models if provided
    if sensor_network:
        server.set_sensor_network(sensor_network)
    
    if ai_models:
        server.set_ai_models(ai_models)
    
    # Create secure tunnel capability
    tunnel = SecureTunnel(local_port=port)
    
    return {
        'server': server,
        'tunnel': tunnel
    }

if __name__ == "__main__":
    # Demo the remote access system
    print("Creating Smart Farming Remote Access System...")
    
    # Create a basic server for testing
    server = RemoteAccessServer(port=5000)
    
    print("Starting server...")
    print("Access the dashboard at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    
    try:
        server.start_server(debug=True)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop_server()
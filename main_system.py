"""
Main System Integration Module for Smart Farming Solution
Integrates all components: AI models, IoT sensors, and remote access
Provides a unified interface for the complete smart farming system
"""

import sys
import os
import time
import threading
import signal
import logging
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from ai_models import initialize_all_models
from iot_sensors import create_demo_sensor_network
from remote_access import create_remote_access_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SmartFarmingSystem:
    """
    Main class that integrates all components of the smart farming solution
    Designed for low-cost, scalable deployment on smallholder farms
    """
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.ai_models = None
        self.sensor_network = None
        self.remote_access = None
        self.is_running = False
        self.monitoring_thread = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def get_default_config(self):
        """Get default configuration for the system"""
        return {
            'system_name': 'Smart Farm IoT System',
            'farm_location': 'Demo Farm',
            'farmer_name': 'Demo Farmer',
            'remote_access': {
                'enabled': True,
                'port': 5000,
                'host': '0.0.0.0'
            },
            'sensors': {
                'monitoring_interval': 300,  # 5 minutes
                'auto_start': True
            },
            'ai': {
                'auto_train': True,
                'prediction_interval': 3600  # 1 hour
            },
            'data_retention_days': 30
        }
    
    def initialize_system(self):
        """Initialize all system components"""
        logger.info("Initializing Smart Farming System...")
        logger.info(f"System: {self.config['system_name']}")
        logger.info(f"Location: {self.config['farm_location']}")
        logger.info(f"Farmer: {self.config['farmer_name']}")
        
        try:
            # Initialize AI models
            logger.info("Initializing AI models...")
            self.ai_models = initialize_all_models()
            logger.info("✓ AI models initialized successfully")
            
            # Initialize sensor network
            logger.info("Initializing sensor network...")
            self.sensor_network = create_demo_sensor_network()
            logger.info("✓ Sensor network initialized successfully")
            
            # Initialize remote access if enabled
            if self.config['remote_access']['enabled']:
                logger.info("Initializing remote access system...")
                self.remote_access = create_remote_access_system(
                    sensor_network=self.sensor_network,
                    ai_models=self.ai_models,
                    port=self.config['remote_access']['port']
                )
                logger.info("✓ Remote access system initialized successfully")
            
            logger.info("🌱 Smart Farming System initialization complete!")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            return False
    
    def start_system(self):
        """Start all system components"""
        if not self.initialize_system():
            logger.error("Failed to initialize system")
            return False
        
        logger.info("Starting Smart Farming System...")
        self.is_running = True
        
        try:
            # Start sensor monitoring if configured
            if self.config['sensors']['auto_start']:
                logger.info("Starting sensor monitoring...")
                self.sensor_network.start_continuous_monitoring(
                    interval=self.config['sensors']['monitoring_interval']
                )
            
            # Start monitoring thread for AI predictions
            if self.config['ai']['auto_train']:
                logger.info("Starting AI monitoring thread...")
                self.monitoring_thread = threading.Thread(
                    target=self._ai_monitoring_loop,
                    daemon=True
                )
                self.monitoring_thread.start()
            
            # Start remote access server if enabled
            if self.config['remote_access']['enabled']:
                logger.info("Starting remote access server...")
                self.print_access_info()
                
                # Start server in a separate thread to avoid blocking
                server_thread = threading.Thread(
                    target=self.remote_access['server'].start_server,
                    kwargs={'debug': False},
                    daemon=True
                )
                server_thread.start()
                
                # Give server time to start
                time.sleep(2)
            
            logger.info("🚀 Smart Farming System is now running!")
            return True
            
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            self.stop_system()
            return False
    
    def stop_system(self):
        """Stop all system components gracefully"""
        logger.info("Stopping Smart Farming System...")
        self.is_running = False
        
        try:
            # Stop sensor monitoring
            if self.sensor_network:
                self.sensor_network.stop_continuous_monitoring()
                logger.info("✓ Sensor monitoring stopped")
            
            # Stop remote access server
            if self.remote_access:
                self.remote_access['server'].stop_server()
                logger.info("✓ Remote access server stopped")
            
            # Wait for monitoring thread to finish
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
                logger.info("✓ AI monitoring thread stopped")
            
            logger.info("🛑 Smart Farming System stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
    
    def _ai_monitoring_loop(self):
        """Background thread for AI predictions and analysis"""
        logger.info("AI monitoring loop started")
        
        while self.is_running:
            try:
                # Get recent sensor data
                if self.sensor_network:
                    df = self.sensor_network.get_recent_readings(hours=1)
                    
                    if not df.empty:
                        # Perform AI analysis on recent data
                        self._analyze_sensor_data(df)
                
                # Sleep for the configured interval
                time.sleep(self.config['ai']['prediction_interval'])
                
            except Exception as e:
                logger.error(f"Error in AI monitoring loop: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def _analyze_sensor_data(self, df):
        """Analyze sensor data using AI models"""
        try:
            # Get latest readings by sensor type
            latest_readings = {}
            for sensor_type in df['sensor_type'].unique():
                type_data = df[df['sensor_type'] == sensor_type]
                if not type_data.empty:
                    latest = type_data.iloc[-1]
                    latest_readings[sensor_type] = float(latest['value']) if latest['value'].replace('.', '').isdigit() else latest['value']
            
            # Perform yield prediction if we have enough data
            required_for_yield = ['temperature', 'humidity', 'soil_moisture']
            if all(sensor_type in latest_readings for sensor_type in required_for_yield):
                try:
                    yield_prediction = self.ai_models['yield_predictor'].predict_yield(
                        temperature=latest_readings['temperature'],
                        humidity=latest_readings['humidity'],
                        soil_moisture=latest_readings['soil_moisture'],
                        rainfall=5.0,  # Default value
                        days_since_planting=60,  # Default value
                        fertilizer_amount=50.0  # Default value
                    )
                    
                    logger.info(f"Yield Prediction: {yield_prediction['predicted_yield_kg_per_hectare']:.1f} kg/hectare")
                    
                except Exception as e:
                    logger.debug(f"Yield prediction error: {e}")
            
            # Perform irrigation recommendation
            required_for_irrigation = ['soil_moisture', 'temperature', 'humidity']
            if all(sensor_type in latest_readings for sensor_type in required_for_irrigation):
                try:
                    irrigation_rec = self.ai_models['irrigation_optimizer'].recommend_irrigation(
                        soil_moisture=latest_readings['soil_moisture'],
                        temperature=latest_readings['temperature'],
                        humidity=latest_readings['humidity'],
                        wind_speed=3.0,  # Default value
                        forecast_rain=2.0,  # Default value
                        crop_stage=2  # Default value
                    )
                    
                    if irrigation_rec['recommended_action'] != 'No Irrigation':
                        logger.info(f"Irrigation Recommendation: {irrigation_rec['recommended_action']} (Confidence: {irrigation_rec['confidence']:.2f})")
                    
                except Exception as e:
                    logger.debug(f"Irrigation recommendation error: {e}")
                    
        except Exception as e:
            logger.error(f"Error analyzing sensor data: {e}")
    
    def print_access_info(self):
        """Print access information for the user"""
        if self.remote_access:
            server = self.remote_access['server']
            local_ip = server.get_local_ip()
            port = self.config['remote_access']['port']
            
            print("\n" + "="*60)
            print("🌐 REMOTE ACCESS INFORMATION")
            print("="*60)
            print(f"📱 Dashboard URL (Local Network): http://{local_ip}:{port}")
            print(f"🏠 Dashboard URL (This Device):   http://localhost:{port}")
            print(f"📊 API Base URL:                 http://{local_ip}:{port}/api")
            print("="*60)
            print("📋 Available API Endpoints:")
            print(f"   • System Status:    GET  /api/status")
            print(f"   • Current Readings: GET  /api/sensors/current")
            print(f"   • Historical Data:  GET  /api/sensors/readings")
            print(f"   • Yield Prediction: POST /api/ai/predict_yield")
            print(f"   • Irrigation Advice: POST /api/ai/irrigation_recommendation")
            print("="*60)
            print("💡 Access the dashboard from any device on your network!")
            print("   (Make sure devices are connected to the same WiFi)")
            print("="*60 + "\n")
    
    def get_system_status(self):
        """Get comprehensive system status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'system_running': self.is_running,
            'config': self.config,
            'components': {
                'ai_models': self.ai_models is not None,
                'sensor_network': self.sensor_network is not None,
                'remote_access': self.remote_access is not None
            }
        }
        
        if self.sensor_network:
            status['sensor_summary'] = self.sensor_network.get_sensor_summary()
        
        if self.ai_models:
            status['ai_status'] = {
                'plant_health_trained': self.ai_models['plant_health'].is_trained,
                'yield_predictor_trained': self.ai_models['yield_predictor'].is_trained,
                'irrigation_optimizer_trained': self.ai_models['irrigation_optimizer'].is_trained
            }
        
        return status
    
    def signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_system()
        sys.exit(0)
    
    def run_interactive_mode(self):
        """Run system in interactive mode with user commands"""
        if not self.start_system():
            return
        
        print("\n🌱 Smart Farming System - Interactive Mode")
        print("Commands: status, readings, predict, irrigate, stop, help")
        
        try:
            while self.is_running:
                try:
                    command = input("\nSmart Farm> ").strip().lower()
                    
                    if command == 'help':
                        print("Available commands:")
                        print("  status   - Show system status")
                        print("  readings - Show current sensor readings")
                        print("  predict  - Get yield prediction")
                        print("  irrigate - Get irrigation recommendation")
                        print("  stop     - Stop the system")
                        print("  help     - Show this help")
                    
                    elif command == 'status':
                        status = self.get_system_status()
                        print(f"System Running: {status['system_running']}")
                        print(f"Components: {status['components']}")
                        if 'sensor_summary' in status:
                            print(f"Sensors: {status['sensor_summary']}")
                    
                    elif command == 'readings':
                        if self.sensor_network:
                            readings = self.sensor_network.read_all_sensors()
                            for reading in readings[:10]:  # Show first 10
                                print(f"  {reading.sensor_id}: {reading.value} {reading.unit}")
                        else:
                            print("Sensor network not available")
                    
                    elif command == 'predict':
                        if self.ai_models:
                            result = self.ai_models['yield_predictor'].predict_yield(
                                temperature=26, humidity=65, soil_moisture=45,
                                rainfall=8, days_since_planting=60, fertilizer_amount=55
                            )
                            print(f"Predicted Yield: {result['predicted_yield_kg_per_hectare']:.1f} kg/hectare")
                        else:
                            print("AI models not available")
                    
                    elif command == 'irrigate':
                        if self.ai_models:
                            result = self.ai_models['irrigation_optimizer'].recommend_irrigation(
                                soil_moisture=35, temperature=28, humidity=55,
                                wind_speed=4, forecast_rain=2, crop_stage=3
                            )
                            print(f"Recommendation: {result['recommended_action']}")
                            print(f"Confidence: {result['confidence']:.2f}")
                        else:
                            print("AI models not available")
                    
                    elif command == 'stop':
                        break
                    
                    elif command == '':
                        continue
                    
                    else:
                        print(f"Unknown command: {command}. Type 'help' for available commands.")
                
                except EOFError:
                    break
                except KeyboardInterrupt:
                    break
        
        finally:
            self.stop_system()

def main():
    """Main entry point for the Smart Farming System"""
    print("🌱 Smart Farming System - Low-Cost IoT & AI Solution")
    print("Designed for Smallholder Farms")
    print("-" * 50)
    
    # Create and start the system
    system = SmartFarmingSystem()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--daemon':
            # Run as daemon (non-interactive)
            if system.start_system():
                try:
                    while system.is_running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
                finally:
                    system.stop_system()
        elif sys.argv[1] == '--status':
            # Just show status and exit
            if system.initialize_system():
                status = system.get_system_status()
                print(f"System Status: {status}")
        else:
            print("Usage: python main_system.py [--daemon|--status]")
    else:
        # Run in interactive mode
        system.run_interactive_mode()

if __name__ == "__main__":
    main()


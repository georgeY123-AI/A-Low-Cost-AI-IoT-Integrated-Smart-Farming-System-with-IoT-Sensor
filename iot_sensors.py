"""
IoT Sensors Module for Smart Farming Solution
Simulates various agricultural sensors and provides data collection capabilities
Designed for low-cost implementation using common sensors like DHT22, soil moisture sensors, etc.
"""

import numpy as np
import pandas as pd
import time
import json
from datetime import datetime, timedelta
import threading
import queue
import random
from dataclasses import dataclass
from typing import Dict, List, Optional
import sqlite3
import os

@dataclass
class SensorReading:
    """Data class for sensor readings"""
    sensor_id: str
    sensor_type: str
    timestamp: datetime
    value: float
    unit: str
    location: Optional[str] = None
    battery_level: Optional[float] = None

class BaseSensor:
    """Base class for all sensor types"""
    
    def __init__(self, sensor_id: str, location: str = "Field_1"):
        self.sensor_id = sensor_id
        self.location = location
        self.is_active = False
        self.battery_level = 100.0
        self.last_reading = None
        self.reading_interval = 300  # 5 minutes default
        
    def read(self) -> SensorReading:
        """Read sensor value - to be implemented by subclasses"""
        raise NotImplementedError
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        self.is_active = True
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.is_active = False
    
    def update_battery(self):
        """Simulate battery drain"""
        self.battery_level = max(0, self.battery_level - 0.01)  # 0.01% per reading

class SoilMoistureSensor(BaseSensor):
    """Simulates a soil moisture sensor (e.g., capacitive soil moisture sensor)"""
    
    def __init__(self, sensor_id: str, location: str = "Field_1", base_moisture: float = 45.0):
        super().__init__(sensor_id, location)
        self.base_moisture = base_moisture
        self.sensor_type = "soil_moisture"
        
    def read(self) -> SensorReading:
        """Simulate soil moisture reading (0-100%)"""
        # Add realistic variations
        time_factor = np.sin(time.time() / 86400 * 2 * np.pi) * 5  # Daily cycle
        random_noise = np.random.normal(0, 2)  # Random variations
        
        # Simulate gradual drying
        hours_since_start = (datetime.now().hour + datetime.now().minute/60)
        drying_factor = -0.5 * hours_since_start if hours_since_start > 6 else 0
        
        moisture = self.base_moisture + time_factor + random_noise + drying_factor
        moisture = np.clip(moisture, 0, 100)
        
        self.update_battery()
        reading = SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=datetime.now(),
            value=round(moisture, 1),
            unit="%",
            location=self.location,
            battery_level=self.battery_level
        )
        self.last_reading = reading
        return reading

class TemperatureSensor(BaseSensor):
    """Simulates a temperature sensor (e.g., DHT22)"""
    
    def __init__(self, sensor_id: str, location: str = "Field_1", base_temp: float = 25.0):
        super().__init__(sensor_id, location)
        self.base_temp = base_temp
        self.sensor_type = "temperature"
        
    def read(self) -> SensorReading:
        """Simulate temperature reading in Celsius"""
        # Daily temperature cycle
        hour = datetime.now().hour + datetime.now().minute/60
        daily_cycle = 8 * np.sin((hour - 6) / 24 * 2 * np.pi)  # Peak at 2 PM
        random_noise = np.random.normal(0, 1)
        
        temperature = self.base_temp + daily_cycle + random_noise
        
        self.update_battery()
        reading = SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=datetime.now(),
            value=round(temperature, 1),
            unit="°C",
            location=self.location,
            battery_level=self.battery_level
        )
        self.last_reading = reading
        return reading

class HumiditySensor(BaseSensor):
    """Simulates a humidity sensor (e.g., DHT22)"""
    
    def __init__(self, sensor_id: str, location: str = "Field_1", base_humidity: float = 60.0):
        super().__init__(sensor_id, location)
        self.base_humidity = base_humidity
        self.sensor_type = "humidity"
        
    def read(self) -> SensorReading:
        """Simulate humidity reading (0-100%)"""
        # Inverse relationship with temperature cycle
        hour = datetime.now().hour + datetime.now().minute/60
        daily_cycle = -10 * np.sin((hour - 6) / 24 * 2 * np.pi)  # Low at 2 PM
        random_noise = np.random.normal(0, 3)
        
        humidity = self.base_humidity + daily_cycle + random_noise
        humidity = np.clip(humidity, 0, 100)
        
        self.update_battery()
        reading = SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=datetime.now(),
            value=round(humidity, 1),
            unit="%",
            location=self.location,
            battery_level=self.battery_level
        )
        self.last_reading = reading
        return reading

class LightSensor(BaseSensor):
    """Simulates a light intensity sensor (e.g., BH1750)"""
    
    def __init__(self, sensor_id: str, location: str = "Field_1"):
        super().__init__(sensor_id, location)
        self.sensor_type = "light_intensity"
        
    def read(self) -> SensorReading:
        """Simulate light intensity reading in lux"""
        hour = datetime.now().hour + datetime.now().minute/60
        
        if 6 <= hour <= 18:  # Daylight hours
            # Peak at noon
            light_cycle = 50000 * np.sin((hour - 6) / 12 * np.pi)
            cloud_factor = np.random.uniform(0.3, 1.0)  # Cloud cover
            light_intensity = light_cycle * cloud_factor
        else:  # Night time
            light_intensity = np.random.uniform(0, 10)  # Moonlight/artificial light
        
        light_intensity = max(0, light_intensity)
        
        self.update_battery()
        reading = SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=datetime.now(),
            value=round(light_intensity, 0),
            unit="lux",
            location=self.location,
            battery_level=self.battery_level
        )
        self.last_reading = reading
        return reading

class PHSensor(BaseSensor):
    """Simulates a soil pH sensor"""
    
    def __init__(self, sensor_id: str, location: str = "Field_1", base_ph: float = 6.5):
        super().__init__(sensor_id, location)
        self.base_ph = base_ph
        self.sensor_type = "soil_ph"
        
    def read(self) -> SensorReading:
        """Simulate pH reading (0-14)"""
        # pH changes slowly over time
        random_drift = np.random.normal(0, 0.1)
        ph_value = self.base_ph + random_drift
        ph_value = np.clip(ph_value, 0, 14)
        
        self.update_battery()
        reading = SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=datetime.now(),
            value=round(ph_value, 2),
            unit="pH",
            location=self.location,
            battery_level=self.battery_level
        )
        self.last_reading = reading
        return reading

class NPKSensor(BaseSensor):
    """Simulates an NPK (Nitrogen, Phosphorus, Potassium) sensor"""
    
    def __init__(self, sensor_id: str, location: str = "Field_1"):
        super().__init__(sensor_id, location)
        self.sensor_type = "npk"
        self.base_n = 50  # mg/kg
        self.base_p = 30  # mg/kg
        self.base_k = 200  # mg/kg
        
    def read(self) -> SensorReading:
        """Simulate NPK reading"""
        # Simulate nutrient depletion over time
        days_since_fertilization = np.random.randint(1, 30)
        depletion_factor = 1 - (days_since_fertilization / 60)  # Gradual depletion
        
        n_value = max(10, self.base_n * depletion_factor + np.random.normal(0, 5))
        p_value = max(5, self.base_p * depletion_factor + np.random.normal(0, 3))
        k_value = max(20, self.base_k * depletion_factor + np.random.normal(0, 15))
        
        # Return as combined reading
        npk_value = f"N:{n_value:.0f},P:{p_value:.0f},K:{k_value:.0f}"
        
        self.update_battery()
        reading = SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            timestamp=datetime.now(),
            value=npk_value,
            unit="mg/kg",
            location=self.location,
            battery_level=self.battery_level
        )
        self.last_reading = reading
        return reading

class SensorNetwork:
    """Manages a network of IoT sensors"""
    
    def __init__(self, db_path: str = "sensor_data.db"):
        self.sensors: Dict[str, BaseSensor] = {}
        self.data_queue = queue.Queue()
        self.is_monitoring = False
        self.monitoring_thread = None
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for sensor data storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sensor_id TEXT NOT NULL,
                sensor_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                value TEXT NOT NULL,
                unit TEXT NOT NULL,
                location TEXT,
                battery_level REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_sensor(self, sensor: BaseSensor):
        """Add a sensor to the network"""
        self.sensors[sensor.sensor_id] = sensor
        print(f"Added sensor {sensor.sensor_id} ({sensor.sensor_type}) at {sensor.location}")
    
    def remove_sensor(self, sensor_id: str):
        """Remove a sensor from the network"""
        if sensor_id in self.sensors:
            del self.sensors[sensor_id]
            print(f"Removed sensor {sensor_id}")
    
    def read_all_sensors(self) -> List[SensorReading]:
        """Read all sensors once"""
        readings = []
        for sensor in self.sensors.values():
            try:
                reading = sensor.read()
                readings.append(reading)
                self.store_reading(reading)
            except Exception as e:
                print(f"Error reading sensor {sensor.sensor_id}: {e}")
        return readings
    
    def store_reading(self, reading: SensorReading):
        """Store sensor reading in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sensor_readings 
            (sensor_id, sensor_type, timestamp, value, unit, location, battery_level)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            reading.sensor_id,
            reading.sensor_type,
            reading.timestamp.isoformat(),
            str(reading.value),
            reading.unit,
            reading.location,
            reading.battery_level
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_readings(self, hours: int = 24) -> pd.DataFrame:
        """Get recent sensor readings as DataFrame"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM sensor_readings 
            WHERE timestamp > datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        '''.format(hours)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def get_sensor_summary(self) -> Dict:
        """Get summary of all sensors"""
        summary = {
            'total_sensors': len(self.sensors),
            'active_sensors': sum(1 for s in self.sensors.values() if s.is_active),
            'sensor_types': {},
            'low_battery_sensors': []
        }
        
        for sensor in self.sensors.values():
            sensor_type = sensor.sensor_type
            if sensor_type not in summary['sensor_types']:
                summary['sensor_types'][sensor_type] = 0
            summary['sensor_types'][sensor_type] += 1
            
            if sensor.battery_level < 20:
                summary['low_battery_sensors'].append({
                    'sensor_id': sensor.sensor_id,
                    'battery_level': sensor.battery_level
                })
        
        return summary
    
    def start_continuous_monitoring(self, interval: int = 300):
        """Start continuous monitoring of all sensors"""
        if self.is_monitoring:
            print("Monitoring already active")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        print(f"Started continuous monitoring with {interval}s interval")
    
    def stop_continuous_monitoring(self):
        """Stop continuous monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        print("Stopped continuous monitoring")
    
    def _monitoring_loop(self, interval: int):
        """Internal monitoring loop"""
        while self.is_monitoring:
            try:
                readings = self.read_all_sensors()
                for reading in readings:
                    self.data_queue.put(reading)
                time.sleep(interval)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(interval)

def create_demo_sensor_network() -> SensorNetwork:
    """Create a demonstration sensor network for a small farm"""
    network = SensorNetwork()
    
    # Add various sensors for different locations
    locations = ["Field_1", "Field_2", "Greenhouse"]
    
    sensor_id = 1
    for location in locations:
        # Soil sensors
        network.add_sensor(SoilMoistureSensor(f"SM_{sensor_id:03d}", location))
        sensor_id += 1
        
        # Environmental sensors
        network.add_sensor(TemperatureSensor(f"TEMP_{sensor_id:03d}", location))
        sensor_id += 1
        
        network.add_sensor(HumiditySensor(f"HUM_{sensor_id:03d}", location))
        sensor_id += 1
        
        network.add_sensor(LightSensor(f"LIGHT_{sensor_id:03d}", location))
        sensor_id += 1
        
        # Soil chemistry sensors (fewer of these due to cost)
        if location != "Greenhouse":  # Only in fields
            network.add_sensor(PHSensor(f"PH_{sensor_id:03d}", location))
            sensor_id += 1
            
            network.add_sensor(NPKSensor(f"NPK_{sensor_id:03d}", location))
            sensor_id += 1
    
    return network

if __name__ == "__main__":
    # Demo the sensor network
    print("Creating demo sensor network...")
    network = create_demo_sensor_network()
    
    print(f"\nSensor Network Summary:")
    summary = network.get_sensor_summary()
    print(f"Total sensors: {summary['total_sensors']}")
    print(f"Sensor types: {summary['sensor_types']}")
    
    print(f"\nTaking initial readings...")
    readings = network.read_all_sensors()
    
    print(f"\nRecent readings:")
    for reading in readings[:5]:  # Show first 5
        print(f"{reading.sensor_id} ({reading.sensor_type}): {reading.value} {reading.unit}")
    
    print(f"\nStarting continuous monitoring for 30 seconds...")
    network.start_continuous_monitoring(interval=5)
    time.sleep(30)
    network.stop_continuous_monitoring()
    
    print(f"\nFinal sensor summary:")
    df = network.get_recent_readings(hours=1)
    print(f"Total readings collected: {len(df)}")
    if not df.empty:
        print(f"Sensor types in data: {df['sensor_type'].unique()}")


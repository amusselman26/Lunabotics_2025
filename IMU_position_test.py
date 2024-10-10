import pyrealsense2 as rs
import numpy as np
import time

# Initialize variables for position estimation
position = np.array([0.0, 0.0, 0.0])  # Initial position (x, y, z)
velocity = np.array([0.0, 0.0, 0.0])  # Initial velocity
dt = 0.1  # Time step (adjust as needed)

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.accel)
pipeline.start(config)
sumaccZ = 0
t = 0

try:
    while True:
        # Wait for a frame set
        frames = pipeline.wait_for_frames()


        # Get accelerometer data
        acc = frames[0].as_motion_frame().get_motion_data()
        accX = round(acc.x, 1)
        accY = round(acc.y, 1) + 9.8
        accZ = round(acc.z, 1) + 0.1
        
        # testing for 0 mean acc in x,y, and z
        # sumaccZ += accZ
        # t += dt
        # meanaccZ = sumaccZ/t
        # print(meanaccZ)
        # Convert acceleration to numpy array (in m/s²)
        acceleration = np.array([accX, accY, accZ])

        # Integrate acceleration to get velocity
        velocity += acceleration * dt

        # Integrate velocity to get position
        position += velocity * dt

        # Print the estimated position
        print(f'Estimated Position (X, Y, Z): {position}')

        # Sleep for a short duration to control the output rate
        time.sleep(dt)

finally:
    # Stop the pipeline
    pipeline.stop()
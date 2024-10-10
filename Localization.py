import cv2
import numpy as np
import pyrealsense2 as rs

# Camera intrinsic parameters (D435i)
fx = 616.57  # Focal length x
fy = 616.57  # Focal length y
cx = 319.5   # Optical center x
cy = 239.5   # Optical center y
camera_matrix = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]], dtype=float)

# Distortion coefficients (assumed negligible for simplicity)
dist_coeffs = np.zeros((5, 1))

# Define ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters = cv2.aruco.DetectorParameters()

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

try:
    while True:
        # Wait for a frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert to grayscale for ArUco detection
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray_image, aruco_dict, parameters=parameters)

        # Process each detected marker
        if ids is not None:
            for i in range(len(ids)):
                # Get the center of the marker
                center = np.mean(corners[i][0], axis=0).astype(int)

                # Get depth at the marker's center
                depth = depth_image[center[1], center[0]]
                distance = depth * 0.001  # Convert from mm to meters

                # Estimate the pose of the marker
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.1, camera_matrix, dist_coeffs)

                # Draw the marker and axes
                cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
                cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

                # Calculate angles
                angle_x = np.degrees(np.arctan2(tvec[0][0][1], tvec[0][0][2]))  # Angle around x-axis
                angle_y = np.degrees(np.arctan2(tvec[0][0][0], tvec[0][0][2]))  # Angle around y-axis

                # Output distance and angles
                print(f'Marker ID: {ids[i][0]}, Distance: {distance:.2f} m, Angle X: {angle_x:.2f} degrees, Angle Y: {angle_y:.2f} degrees')

        # Display the output image
        cv2.imshow('Detected ArUco Markers', color_image)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop the pipeline
    pipeline.stop()
    cv2.destroyAllWindows()

import cv2
import numpy as np
import depthai as dai
import contextlib
import time

# Camera intrinsic parameters (from Matlab) for (left or right) oak-d-lite
fx1 = 1515.24261837315  # Focal length x
fy1 = 1513.21547841726  # Focal length y
cx1 = 986.009156502993   # Optical center x
cy1 = 551.618039270305   # Optical center y
camera_matrix1 = np.array([[fx1, 0, cx1],
                           [0, fy1, cy1],
                           [0, 0, 1]], dtype=float)

# Distortion coefficients for (left or right) oak-d-lite
dist_coeffs1 = np.array((0.114251294509202,-0.228889968220235,0,0))

relative_position1 = [-0.145, 0, 90] # Relative position of the camera with respect to the robot base (X, Y, Theta)
relative_position2 = [0.145, 0, 90]  # Relative position of the camera with respect to the robot base (X, Y, Theta)

# Camera intrinsic parameters for the other (left or right) oak-d-lite (assumed the same for both cameras for now (03/18/25) update when other cameras are calibrated)
fx2 = 1515.24261837315  # Focal length x
fy2 = 1513.21547841726  # Focal length y
cx2 = 986.009156502993   # Optical center x
cy2 = 551.618039270305   # Optical center y
camera_matrix2 = np.array([[fx2, 0, cx2],
                           [0, fy2, cy2],
                           [0, 0, 1]], dtype=float)

# Distortion coefficients for (left or right) oak-d-lite
dist_coeffs2 = np.array((0.114251294509202,-0.228889968220235,0,0))



CAMERA_INFOS = {
 "rgb-14442C10911DC5D200-OAK-D-LITE" : {"camera_matrix" : camera_matrix1, "dist_coeffs" : dist_coeffs1, "relative_position" : relative_position1},
 "rgb-14442C10911DC5D600-OAK-D-LITE" : {"camera_matrix" : camera_matrix2, "dist_coeffs" : dist_coeffs2, "relative_position" : relative_position2}
}


# Distortion coefficients (assumed negligible for simplicity)
dist_coeffs1 = np.array((0.114251294509202,-0.228889968220235,0,0))

marker_size = 0.11

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_ITERATIVE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

# Define ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

def createPipeline():
    # Start defining a pipeline
    pipeline = dai.Pipeline()
    # Define a source - color camera
    camRgb = pipeline.create(dai.node.ColorCamera)

    camRgb.setPreviewSize(640, 480)
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)

    # Create output
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    camRgb.preview.link(xoutRgb.input)

    # Define sources and outputs
    imu = pipeline.create(dai.node.IMU)
    xlinkOut = pipeline.create(dai.node.XLinkOut)

    xlinkOut.setStreamName("imu")

    # Enable ACCELEROMETER_RAW at 500 Hz rate
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 500)
    # Enable GYROSCOPE_RAW at 400 Hz rate
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 400)

    # Set batch report thresholds
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    # Link plugins IMU -> XLINK
    imu.out.link(xlinkOut.input)

    return pipeline

def timeDeltaToMilliS(delta) -> float:
        return delta.total_seconds() * 1000

with contextlib.ExitStack() as stack:
    deviceInfos = dai.Device.getAllAvailableDevices()
    usbSpeed = dai.UsbSpeed.SUPER
    openVinoVersion = dai.OpenVINO.Version.VERSION_2021_4

    qRgbMap = []
    devices = []

    mxId_list = []

    for deviceInfo in deviceInfos:
        deviceInfo: dai.DeviceInfo
        device: dai.Device = stack.enter_context(dai.Device(openVinoVersion, deviceInfo, usbSpeed))
        devices.append(device)
        print("===Connected to ", deviceInfo.getMxId())
        mxId = device.getMxId()
        cameras = device.getConnectedCameras()
        usbSpeed = device.getUsbSpeed()
        eepromData = device.readCalibration2().getEepromData()
        print("   >>> MXID:", mxId)
        print("   >>> Num of cameras:", len(cameras))
        print("   >>> USB speed:", usbSpeed)
        if eepromData.boardName != "":
            print("   >>> Board name:", eepromData.boardName)
        if eepromData.productName != "":
            print("   >>> Product name:", eepromData.productName)

        pipeline = createPipeline()
        device.startPipeline(pipeline)

         # Output queue for imu bulk packets
        imuQueue = device.getOutputQueue(name="imu", maxSize=50, blocking=False)

        baseTs = None
        prev_gyroTs = None

        # Output queue will be used to get the rgb frames from the output defined above
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        stream_name = "rgb-" + mxId + "-" + eepromData.productName
        qRgbMap.append((q_rgb, stream_name))

        # Create resizable windows for each stream
        cv2.namedWindow(stream_name, cv2.WINDOW_NORMAL)
        mxId_list.append(mxId)

    print(mxId_list)

    last_print_time = time.time()  # Initialize time tracking

    camera_position = [0, 0, 0]  # X, Z, Theta
    pose = [0, 0, np.degrees(0)]  # X, Z, Theta

    print(qRgbMap)

    while True:
        for q_rgb, stream_name in qRgbMap:
            if q_rgb.has():
                color_image = q_rgb.get().getCvFrame()

                imuData = imuQueue.get()  # Blocking call, will wait until new data has arrived
                imuPackets = imuData.packets

                # Convert to grayscale for ArUco detection
                gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

                # Detect ArUco markers
                corners, ids, _ = aruco_detector.detectMarkers(gray_image)


                # Process each detected marker and get pose relative to id 2
                if ids is not None and 2 in ids:
                    arr = np.where(ids == 2)[0][0]
                    corners = np.array(corners[arr])
                    ids = np.array(ids[arr])
                    # Get the center of the marker
                    center = np.mean(corners[0], axis=0).astype(int)

                    camera_matrix = CAMERA_INFOS[q_rgb]["camera_matrix"]
                    dist_coeffs = CAMERA_INFOS[q_rgb]["dist_coeffs"]

                    rvec, tvec, _ = my_estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

                    rvec = np.array(rvec)
                    tvec = np.array(tvec)

                    # Draw the marker and axes
                    cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
                    rotation_matrix, _ = cv2.Rodrigues(rvec[0])  # Convert rotation vector to rotation matrix using the rodrigues formula
                    R_inv = rotation_matrix.T  # Inverse of the rotation matrix
                    camera_position = R_inv @ tvec[0]  # @ is the matrix multiplication operator in Python
                    theta = np.arcsin(-R_inv[2][0])


                    pose = [camera_position[0][0], camera_position[2][0], np.degrees(theta)]


                elif camera_position is not None:
                    for imuPacket in imuPackets:
                        # camera_position = camera_position += encoder_counts*slip_coefficient
                        
                        acceleroValues = imuPacket.acceleroMeter
                        gyroValues = imuPacket.gyroscope

                        acceleroTs = acceleroValues.getTimestampDevice()
                        gyroTs = gyroValues.getTimestampDevice()

                        if baseTs is None:
                            baseTs = acceleroTs if acceleroTs < gyroTs else gyroTs
                            prev_gyroTs = gyroTs
                            print(baseTs)

                        acceleroTs = timeDeltaToMilliS(acceleroTs - baseTs)
                        gyroTs = timeDeltaToMilliS(gyroTs - baseTs)

                        imuF = "{:.06f}"
                        tsF = "{:.03f}"

                        # Calculate the time difference between the current and previous gyroscope readings
                        dt = (gyroTs - timeDeltaToMilliS(prev_gyroTs - baseTs)) / 1000.0  # Convert milliseconds to seconds
                        prev_gyroTs = gyroValues.getTimestampDevice()

                        gyroValues = round(gyroValues.x, 2), round(gyroValues.y, 2), round(gyroValues.z, 2)

                        # Integrate the gyroscope data to get the angles
                        pose[2] += np.degrees(gyroValues[1]) * dt  # Pitch

                else:
                    print("rotate to find aruco idiot")
            
                current_time = time.time()
                if current_time - last_print_time >= 1 and camera_position is not None:
                    print(f"Camera Position: {pose}")
                    last_print_time = current_time  # Update last print time


                # Display the output image
                cv2.imshow(stream_name, color_image)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
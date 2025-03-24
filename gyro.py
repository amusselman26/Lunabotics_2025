#!/usr/bin/env python3

import cv2
import depthai as dai
import math
import contextlib

def createPipeline():
    # Create pipeline
    pipeline = dai.Pipeline()

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

def initialize(deviceInfo, openVinoVersion, usbSpeed):
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
    initial_angle = [0, 0, 0]  # Roll, Pitch, Yaw

    return imuQueue, baseTs, prev_gyroTs, initial_angle

with contextlib.ExitStack() as stack:
    deviceInfos = dai.Device.getAllAvailableDevices()
    usbSpeed = dai.UsbSpeed.SUPER
    openVinoVersion = dai.OpenVINO.Version.VERSION_2021_4

    devices = []
    imuQueues = []
    baseTss = []
    prev_gyroTss = []
    initial_angles = []

    for deviceInfo in deviceInfos:
        imuQueue, baseTs, prev_gyroTs, initial_angle = initialize(deviceInfo, openVinoVersion, usbSpeed)
        imuQueues.append(imuQueue)
        baseTss.append(baseTs)
        prev_gyroTss.append(prev_gyroTs)
        initial_angles.append(initial_angle)

    while True:
        for i, imuQueue in enumerate(imuQueues):
            imuData = imuQueue.get()  # Blocking call, will wait until new data has arrived
            imuPackets = imuData.packets

            for imuPacket in imuPackets:
                acceleroValues = imuPacket.acceleroMeter
                gyroValues = imuPacket.gyroscope

                acceleroTs = acceleroValues.getTimestampDevice()
                gyroTs = gyroValues.getTimestampDevice()

                if baseTss[i] is None:
                    baseTss[i] = acceleroTs if acceleroTs < gyroTs else gyroTs
                    prev_gyroTss[i] = gyroTs
                    print(baseTss[i])

                acceleroTs = timeDeltaToMilliS(acceleroTs - baseTss[i])
                gyroTs = timeDeltaToMilliS(gyroTs - baseTss[i])

                imuF = "{:.06f}"
                tsF = "{:.03f}"

                # Calculate the time difference between the current and previous gyroscope readings
                dt = (gyroTs - timeDeltaToMilliS(prev_gyroTss[i] - baseTss[i])) / 1000.0  # Convert milliseconds to seconds
                prev_gyroTss[i] = gyroValues.getTimestampDevice()

                gyroValues = round(gyroValues.x, 2), round(gyroValues.y, 2), round(gyroValues.z, 2)

                # Integrate the gyroscope data to get the angles
                initial_angles[i][0] += gyroValues[0] * dt  # Roll
                initial_angles[i][1] += gyroValues[1] * dt  # Pitch
                initial_angles[i][2] += gyroValues[2] * dt  # Yaw

                print(f"Camera {i} Angles [degrees]: Roll: {math.degrees(initial_angles[i][0])} Pitch: {math.degrees(initial_angles[i][1])} Yaw: {math.degrees(initial_angles[i][2])}")

        if cv2.waitKey(1) == ord('q'):
            break
'''
Project:    Raspberry Pi 4b + MCP2515 CAN Testing
Autor:      NotAWildernessExplorer
Date:       05/10/2025


MATERIALS:
 - MCP2515 CAN Bus Module with TJA1050 receiver 
 - Raspberry Pi 4b
 - Jumper wires for connections

Wiring:
-------------------------
|    MCP2515  |   Pico  |   
-------------------------
|     CS      |   GP08  | 
|     MOSI    |   GP10  | 
|     SCK     |   GP11  |
|     MISO    |   GP21  |
|     INT     |   GP25  | 
|     VCC     |   3V3   | 
|     GND     |   GND   | 
-------------------------

Dependencies:
 - canutils
 - python-can


Setup:
(1) Add the following to /boot/firmware/config.txt
dtoverlay=mcp2515-can0,oscillator=8000000,interrupt=25,spimaxfrequency=10000000

(2) Reboot the pi 
$sudo reboot

(3) Install dependencies
$sudo apt install canutils
$pip3 install python-can

(4a) Set up the CAN network, remove loopback if not testing
$sudo ip link set can0 up type can bitrate 100000 loopback on

(4b) To shut down the can network
$sudo ip link set can0 down

(5) To read can messages
$python3 socketcan_recv.py

'''
import time
import can
import struct




def unpack_data(msg:can.message.Message,telem:dict)->None:
    '''
    **Unpacks CAN data frames and stores in a dictonary**\n
    - msg:can.message.Message \n
      - the can message\n
    - telem:dict \n
      - the dictonary to store the data in\n

    Please only edit this function in the file
    '''

    ## if we are an error, don't read it
    if msg.is_error_frame:
        return None
    
    ## Read messages from Pico A
    if msg.arbitration_id == 0x700:
        last_send_time,pico_tenp = struct.unpack('<ff',msg.data)
        telem['temp_picoA'] = pico_tenp
    if msg.arbitration_id == 0x701:
        telem['enc_FL'] = struct.unpack('<q',msg.data)[0]
    if msg.arbitration_id == 0x702:
        telem['enc_FR'] = struct.unpack('<q',msg.data)[0]
    
    ## Read messages from Pico A
    if msg.arbitration_id == 0x500:
        last_send_time,pico_tenp = struct.unpack('<ff',msg.data)
        telem['temp_picoB'] = pico_tenp
    if msg.arbitration_id == 0x501:
        telem['enc_BL'] = struct.unpack('<q',msg.data)[0]
    if msg.arbitration_id == 0x502:
        telem['enc_BR'] = struct.unpack('<q',msg.data)[0]
    
    ## Read messages form Power pico


def read_can(telemetry_dat:dict,_timeout = 100_000_000)-> dict:
    '''
    **Reads data form the can bus and stores it the given dictonary**\n
    - telemetry_dat:dict \n
      - the dictonary to store the data in\n
    - timeout:int \n
      - time in nano seconds to read can messages form the queue\n

    '''

    ## Set a temp dictonary to store new data
    _telem_temp = telemetry_dat

    ## open the can line
    with can.Bus(channel='can0',interface='socketcan') as bus:

        ## Set up the queue with python-can
        queue = can.BufferedReader()
        notifier = can.Notifier(bus, [queue])

        ## Read a message from the can line
        msg = queue.get_message()
        first_read_time = time.monotonic_ns()

        ## if we have a message and are not timed out, read more messages
        while msg is not None and (time.monotonic_ns()-first_read_time < _timeout):
            
            ## unpack the data
            unpack_data(msg,_telem_temp)
            
            ## REad the next message
            msg = queue.get_message()

            ## For debugging
            #print(f"{bus.channel_info}| {hex(msg.arbitration_id)}| {msg.is_error_frame}|{msg.data}")    # print to terminal

        
        notifier.stop()

    ## BONUS Read the CPU TEMP
    with open('/sys/class/thermal/thermal_zone0/temp','r') as file:
        temp_data = file.read()
        _telem_temp['temp_pi4'] = float(temp_data)/1000


    return _telem_temp



if __name__ == "__main__":
    vars = {}
    while True:
        read_can(vars)
        print(vars)
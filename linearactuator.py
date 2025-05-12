'''
-----------------------------------------------------
-----------------------------------------------------
Lunabotics Linear Actuator Testing
Author: NotAWildernessExplorer
Date:04/11/2025 
-----------------------------------------------------
-----------------------------------------------------
Connections for BST7960 to Pi4 b
Pin 1: GP12 pin 32
Pin 2: GP13 pin 33
Pin 3: 3V
Pin 4: 3V
Pin 5: NC
Pin 6: NC
Pin 7: 3V
Pin 8: GND
-----------------------------------------------------
-----------------------------------------------------
'''

## Import libraries
import time  # For delays
import lgpio  # For GPIO handling

## Luna linear actuators start here!
class LinearActuator:
    def __init__(self):
        # Define GPIO pins
        self.R_PWM_PIN = 12  # GPIO 12 (pin 32)
        self.L_PWM_PIN = 13  # GPIO 13 (pin 33)

        # Initialize GPIO chip
        self.chip = lgpio.gpiochip_open(0)  # Open GPIO chip 0 (default for Raspberry Pi)

        # Set up pins as outputs
        lgpio.gpio_claim_output(self.chip, self.R_PWM_PIN)
        lgpio.gpio_claim_output(self.chip, self.L_PWM_PIN)

        # Initialize PWM
        self.R_PWM = lgpio.tx_pwm(self.chip, self.R_PWM_PIN, 125000)  # 125 kHz PWM on R_PWM_PIN
        self.L_PWM = lgpio.tx_pwm(self.chip, self.L_PWM_PIN, 125000)  # 125 kHz PWM on L_PWM_PIN

    def move(self, qty):
        '''
        Changes motor controller duty cycle
        qty > 0: extend
        qty < 0: retract
        qty = 0: stop
        '''
        if qty > 0:
            self.stop()  # Stop motors
            time.sleep(0.001)  # Wait
            lgpio.tx_pwm(self.chip, self.R_PWM_PIN, 125000, 100)  # Set R_PWM to 100% duty cycle
        elif qty < 0:
            self.stop()  # Stop motors
            time.sleep(0.001)  # Wait
            lgpio.tx_pwm(self.chip, self.L_PWM_PIN, 125000, 100)  # Set L_PWM to 100% duty cycle
        else:
            self.stop()  # Stop motors
            time.sleep(0.001)  # Wait

    def stop(self):
        '''Stops the motors'''
        lgpio.tx_pwm(self.chip, self.R_PWM_PIN, 125000, 0)  # Set R_PWM to 0% duty cycle
        lgpio.tx_pwm(self.chip, self.L_PWM_PIN, 125000, 0)  # Set L_PWM to 0% duty cycle

    def cleanup(self):
        '''Cleans up GPIO resources'''
        self.stop()  # Ensure motors are stopped
        lgpio.gpiochip_close(self.chip)  # Close the GPIO chip

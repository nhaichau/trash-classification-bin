#!/usr/bin/env python3
#
# This inference reuses sample code published in Nvidia dusty-nv github repo 

import sys
import Jetson.GPIO as GPIO
import time
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log

# Create video sources and outputs
inputFrames = videoSource("csi://0")
outputFrames = videoOutput()
    
net = detectNet(model="models/trash2/ssd-mobilenet.onnx", labels="models/trash2/labels.txt", input_blob="input_0", output_cvg="scores", 
                    output_bbox="boxes", threshold=0.5)

# Led pin for plastic, metal
led_pin11 = 11

# Led pin for non-recycling garbage
led_pin12 = 12

# Set up the GPIO channel
GPIO.setmode(GPIO.BOARD) 
GPIO.setup(led_pin11, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(led_pin12, GPIO.OUT, initial=GPIO.LOW)

# Start capturing camera frames and detecting trash types
def start_loop():
    print("Start controlling trash bin.")
    start_time = time.time()
    end_time = start_time + 20  # 20 seconds from start_time
    
    # Debounce variables to reduce fluctuates of leds
    detection_counter = 0
    detection_threshold = 5  # Number of consistent detections required
    
    while time.time() < end_time:
        img = inputFrames.Capture()

        if img is None: # timeout
            continue  
                
        # Detect objects in the image (with overlay)
        detections = net.Detect(img, overlay="box,labels,conf")
        
        # Count number of detection
        if detections:
            detection_counter += 1
        else:
            detection_counter = 0

        # Render the image
        outputFrames.Render(img)
        # Update the title bar
        outputFrames.SetStatus("{:s} | Network {:.0f} FPS".format("ssd-mobilenet-v2", net.GetNetworkFPS()))

        if not inputFrames.IsStreaming() or not outputFrames.IsStreaming():
            break
        
        # Control leds
        if detection_counter >= detection_threshold:
            print("Metal or plastic detected")
            GPIO.output(led_pin11, GPIO.HIGH) 
            print("LED11 is ON")
            GPIO.setup(led_pin12, GPIO.OUT, initial=GPIO.LOW)
        else:
            print("Metal or plastic NOT detected")
            GPIO.output(led_pin12, GPIO.HIGH) 
            print("LED12 is ON")
            GPIO.setup(led_pin11, GPIO.OUT, initial=GPIO.LOW)
                
    print("Loop stopped.")
    inputFrames.Close()
    outputFrames.Close()
    GPIO.setup(led_pin11, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(led_pin12, GPIO.OUT, initial=GPIO.LOW)

if __name__ == "__main__":
    while True:
        user_input = input("Enter 's' to start the detection or 'esc' to exit\n")
        if user_input.lower() == 's':
            start_loop()  

        elif user_input.lower() == chr(27):
            print("Exiting program.")
            inputFrames.Close()
            outputFrames.Close()
            GPIO.setup(led_pin11, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(led_pin12, GPIO.OUT, initial=GPIO.LOW)
            sys.exit()
# Optick
A Third Eye for Search Operations

## Description
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

## Hardware Requirements
You will need:
1. Raspberry Pi
2. Carbon Monoxide Sensor (MQ-7)
3. Carbon Dioxide Sensor (CCS811)
4. Thermal Sensor (AMG8833)
5. Analog to Digital Convertor (MCP3008)
6. SH1106 OLED Display
7. Capacitive Sensor/ button

Apart from the above, you will need wires, and a battery to wire up the Raspberry Pi.

## Instructions
1. Connect the MQ-7 Sensor to the MCP3008 
2. Connect the other sensors, button and the MCP3008 to the Raspberry Pi. 
3. Clone this repository on the Raspberry Pi.

git clone https://github.com/mahathiamencherla15/Optick.git

4. Configure the pins as per your connections and make the changes to the code.
5. Run the code for generate a dataset of 1000 images.
```bash
$python3 amgtest.py
```bash
6. The Human Detection Mode requires a pretrained ML/DL model which can be done on your machine easily. Use this link below as reference for the same. 
```bash
$ https://towardsdatascience.com/detecting-people-with-a-raspberrypi-a-thermal-camera-and-machine-learning-376d3bbcd45c
```
7. Run this code for execution
```bash
$ python3 optick_main.py
```

# QTM Data Streaming with Moticon Insole Integration

This project contains scripts to stream motion capture data from Qualisys Track Manager (QTM) triggered by Moticon pressure insoles. The UDP data from Moticon insoles triggers QTM to start recording and streaming.

## Setup

1. Make the setup script executable:
   ```
   chmod +x setup.sh
   ```

2. Run the setup script to create a virtual environment and install dependencies:
   ```
   ./setup.sh
   ```

3. Activate the virtual environment:
   ```
   source qtm_env/bin/activate
   ```

## Moticon Insole Setup

1. In the Moticon OpenGo Software:
   - Go to Settings > UDP Output
   - Configure the UDP output with the IP address of the computer running this script
   - Set the port to 8888 (or your preferred port)
   - Use the channel filter to select which data to stream (e.g., "total_force acc")
   - Turn on UDP output

## QTM Setup

1. Make sure QTM is running and ready for remote control.
2. If your QTM instance requires a password for remote control, you'll need to provide it when running the script.

## Usage

1. Start QTM and ensure it's running but **not** yet recording or streaming.

2. Run the streaming script:
   ```
   python qtm_stream.py
   ```
   If QTM requires a password for control:
   ```
   python qtm_stream.py --qtm-password YOUR_PASSWORD
   ```

3. Start the Moticon OpenGo Software and begin streaming UDP data.

4. When the first UDP data packet is received from Moticon, the script will:
   - Take control of QTM
   - Start a new measurement in QTM
   - Begin streaming data from QTM
   - Record synchronized QTM and Moticon data to a file

5. By default, the script:
   - Connects to QTM on localhost (127.0.0.1)
   - Listens for Moticon insole data on localhost:8888
   - Saves synchronized data to a 'data' directory

6. You can customize these options:
   ```
   python qtm_stream.py --qtm-host 192.168.1.100 --qtm-port 22223 --qtm-password PASSWORD --insole-ip 192.168.1.101 --insole-port 8888 --output-dir my_data
   ```

7. Press Ctrl+C to stop recording, streaming, and save the file.

## Output

The script creates a CSV file with a timestamp in the filename. The CSV file contains:
- Frame number
- Timestamp
- 3D coordinates (X, Y, Z) for each marker
- Force plate data (forces, moments, center of pressure)
- Pressure insole data (synchronized with the nearest QTM timestamp)

The file is saved in the specified output directory (default: 'data').

## Synchronization

The script synchronizes QTM and insole data based on timestamps. For each QTM frame, the system finds the closest insole data point in time (within 100ms) and combines them in the same row of the output file.

## How It Works

1. The script first connects to QTM but doesn't start recording.
2. It then listens for UDP packets from Moticon OpenGo.
3. When the first UDP packet is received, it triggers QTM to:
   - Take control of QTM
   - Start a new measurement
   - Begin streaming data
4. Both data streams are then synchronized and saved to a CSV file. 


## Kinematic
# Markers

1: L_IAS
2: L_IPS
3: R_IPS
4: R_IAS
5: L_FTC
6: L_FLE
7: L_FAX
8: L_TTC
9: L_FAL
10: L_FCC
11: L_FM1
12: L_FM5
13: R_FTC
14: R_FLE
15: R_FAX
16: R_TTC
17: R_FAL
18: R_FCC
19: R_FM1
20: R_FM5

# Bones
[
   {
      "From": "L_FAX",
      "To": "L_TTC",
      "Color": 4403493
   },
   {
      "From": "L_FLE",
      "To": "L_FAX",
      "Color": 4403493
   },
   {
      "From": "R_FAX",
      "To": "R_TTC",
      "Color": 4403493
   },
   {
      "From": "R_FLE",
      "To": "R_FAX",
      "Color": 4403493
   },
   {
      "From": "L_IPS",
      "To": "R_IPS",
      "Color": 4403493
   },
   {
      "From": "L_FAL",
      "To": "L_FCC",
      "Color": 4403493
   },
   {
      "From": "R_FAL",
      "To": "R_FCC",
      "Color": 4403493
   },
   {
      "From": "L_FM1",
      "To": "L_FM5",
      "Color": 4403493
   },
   {
      "From": "R_FM1",
      "To": "R_FM5",
      "Color": 4403493
   },
   {
      "From": "L_FAL",
      "To": "L_FM5",
      "Color": 4403493
   },
   {
      "From": "R_FAL",
      "To": "R_FM5",
      "Color": 4403493
   },
   {
      "From": "R_IPS",
      "To": "R_IAS",
      "Color": 4403493
   },
   {
      "From": "L_IAS",
      "To": "L_IPS",
      "Color": 4403493
   },
   {
      "From": "L_IAS",
      "To": "L_FTC",
      "Color": 4403493
   },
   {
      "From": "R_IAS",
      "To": "R_FTC",
      "Color": 4403493
   },
   {
      "From": "R_FTC",
      "To": "R_FLE",
      "Color": 4403493
   },
   {
      "From": "L_FTC",
      "To": "L_FLE",
      "Color": 4403493
   },
   {
      "From": "L_TTC",
      "To": "L_FAL",
      "Color": 4403493
   },
   {
      "From": "R_FAX",
      "To": "R_FAL",
      "Color": 4403493
   },
   {
      "From": "L_FCC",
      "To": "L_FM1",
      "Color": 4403493
   },
   {
      "From": "R_TTC",
      "To": "R_FAL",
      "Color": 4403493
   },
   {
      "From": "R_FCC",
      "To": "R_FM1",
      "Color": 4403493
   }
],
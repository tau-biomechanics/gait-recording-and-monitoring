#!/usr/bin/env python3
"""
Moticon UDP Receiver
A simple script to receive UDP data from Moticon insoles and print it to the console.
Use this to validate your UDP connection before integrating with QTM.
"""

import socket
import argparse
import signal
import sys
import time
import datetime

# Global variable to control the main loop
running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C to exit gracefully"""
    global running
    print("\nShutting down...")
    running = False

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Receive UDP data from Moticon insoles and print to console"
    )
    parser.add_argument(
        "--ip", 
        default="0.0.0.0", 
        help="IP address to listen on (default: 0.0.0.0, which listens on all interfaces)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=5555, 
        help="Port to listen on (default: 5555)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Print additional debugging information"
    )
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"Moticon UDP Receiver")
    print(f"Listening on {args.ip}:{args.port}")
    print("Press Ctrl+C to exit\n")
    
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.ip, args.port))
    
    # Set socket to non-blocking mode
    sock.setblocking(False)
    
    # Start receiving data
    packet_count = 0
    start_time = time.time()
    
    try:
        while running:
            try:
                # Try to receive data with timeout
                data, addr = sock.recvfrom(2048)
                current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                
                # Parse the data
                data_str = data.decode('utf-8').strip()
                values = data_str.split()
                
                # Convert values to floats where possible
                try:
                    values = [float(v) for v in values]
                except ValueError:
                    # If conversion fails, keep as strings
                    pass
                
                # Print received data
                packet_count += 1
                print(f"[{current_time}] Packet #{packet_count} from {addr[0]}:{addr[1]}")
                print(f"Data: {values}")
                
                # Print some debug info if requested
                if args.debug:
                    print(f"Raw data: {data}")
                    print(f"Length: {len(values)} values")
                
                print("-" * 40)
                
            except BlockingIOError:
                # No data available, sleep a bit to avoid maxing CPU
                time.sleep(0.001)
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(0.1)
    
    finally:
        # Close the socket
        sock.close()
        
        # Print statistics
        duration = time.time() - start_time
        print(f"\nReceived {packet_count} packets over {duration:.2f} seconds")
        if packet_count > 0:
            print(f"Average rate: {packet_count / duration:.2f} packets/second")

if __name__ == "__main__":
    main() 
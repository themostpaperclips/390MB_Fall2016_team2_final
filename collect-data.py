import socket
import sys
import json
import numpy as np

user_id = "be.af.9a.d0.e9.3f.e3.db.8f.94"

count = 0

'''
    This socket is used to send data back through the data collection server.
    It is used to complete the authentication. It may also be used to send
    data or notifications back to the phone, but we will not be using that
    functionality in this assignment.
'''
send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
send_socket.connect(("none.cs.umass.edu", 9999))

#################   Server Connection Code  ####################

'''
    This socket is used to receive data from the data collection server
'''
receive_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
receive_socket.connect(("none.cs.umass.edu", 8888))
# ensures that after 1 second, a keyboard interrupt will close
receive_socket.settimeout(1.0)

msg_request_id = "ID"
msg_authenticate = "ID,{}\n"
msg_acknowledge_id = "ACK"

def authenticate(sock):
    """
    Authenticates the user by performing a handshake with the data collection server.

    If it fails, it will raise an appropriate exception.
    """
    message = sock.recv(256).strip()
    if (message == msg_request_id):
        print("Received authentication request from the server. Sending authentication credentials...")
        sys.stdout.flush()
    else:
        print("Authentication failed!")
        raise Exception("Expected message {} from server, received {}".format(msg_request_id, message))
    sock.send(msg_authenticate.format(user_id))

    try:
        message = sock.recv(256).strip()
    except:
        print("Authentication failed!")
        raise Exception("Wait timed out. Failed to receive authentication response from server.")

    if (message.startswith(msg_acknowledge_id)):
        ack_id = message.split(",")[1]
    else:
        print("Authentication failed!")
        raise Exception("Expected message with prefix '{}' from server, received {}".format(msg_acknowledge_id, message))

    if (ack_id == user_id):
        print("Authentication successful.")
        sys.stdout.flush()
    else:
        print("Authentication failed!")
        raise Exception("Authentication failed : Expected user ID '{}' from server, received '{}'".format(user_id, ack_id))


try:
    print("Authenticating user for receiving data...")
    sys.stdout.flush()
    authenticate(receive_socket)

    print("Authenticating user for sending data...")
    sys.stdout.flush()
    authenticate(send_socket)

    print("Successfully connected to the server! Waiting for incoming data...")
    sys.stdout.flush()

    previous_json = ''

    magnetometer_data = []
    barometer_data = []
    light_data = []

    while True:
        try:
            message = receive_socket.recv(1024).strip()
            json_strings = message.split("\n")
            json_strings[0] = previous_json + json_strings[0]
            for json_string in json_strings:
                try:
                    data = json.loads(json_string)
                except:
                    previous_json = json_string
                    continue
                previous_json = '' # reset if all were successful
                sensor_type = data['sensor_type']

                # Find the sensor type and add the data to the appropriate list
                if (sensor_type == u"SENSOR_MAGNETOMETER"):
                    print("Received Magnetometer data")
                    t = data['data']['t']
                    x = data['data']['x']
                    y = data['data']['y']
                    z = data['data']['z']
                    label = data['label']
                    magnetometer_data.append([t, x, y, z, label])
                elif (sensor_type == u"SENSOR_BAROMETER"):
                    print("Received Barometer data")
                    t = data['data']['t']
                    val = data['data']['value']
                    label = data['label']
                    barometer_data.append([t, val, label])
                elif (sensor_type == u"SENSOR_LIGHT"):
                    print("Received Light data")
                    t = data['data']['t']
                    val = data['data']['value']
                    label = data['label']
                    light_data.append([t, val, label])

            sys.stdout.flush()
        except KeyboardInterrupt:
            # occurs when the user presses Ctrl-C
            print("User Interrupt. Quitting...")
            raise KeyboardInterrupt
            break
        except Exception as e:
            # ignore exceptions, such as parsing the json
            # if a connection timeout occurs, also ignore and try again. Use Ctrl-C to stop
            # but make sure the error is displayed so we know what's going on
            if (e.message != "timed out"):  # ignore timeout exceptions completely
                print(e)
            pass
except KeyboardInterrupt:
    # occurs when the user presses Ctrl-C
    print("User Interrupt. Saving labelled data...")

    # Append the data to the data files
    magnetometer_data = np.asarray(magnetometer_data)
    magFile = open('data/magnetometer_data.csv','ab')
    np.savetxt(magFile, magnetometer_data, delimiter=",")
    barometer_data = np.asarray(barometer_data)
    barFile = open('data/barometer_data.csv','ab')
    np.savetxt(barFile, barometer_data, delimiter=",")
    light_data = np.asarray(light_data)
    lightFile = open('data/light_data.csv','ab')
    np.savetxt(lightFile, light_data, delimiter=",")
finally:
    print >>sys.stderr, 'closing socket for receiving data'
    receive_socket.shutdown(socket.SHUT_RDWR)
    receive_socket.close()

    print >>sys.stderr, 'closing socket for sending data'
    send_socket.shutdown(socket.SHUT_RDWR)
    send_socket.close()

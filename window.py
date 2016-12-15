import numpy as np
import sys

class Window:
    def __init__(self, size):

        # The data itself
        self.data = {'magnetometer': [], 'barometer': [], 'light': []}

        # The window size, in milliseconds
        self.size = size

        # The first data point
        self.start = None

    def push_point(self, data):
        """
        Add a point to the window and return false if window is full
        """
        sensor_type = data['sensor_type']
        time = data['data']['t']
        if (self.start == None):
            self.start = time
        if (time - self.start > self.size):
            return False
        elif (sensor_type == u"SENSOR_MAGNETOMETER"):
            x = data['data']['x']
            y = data['data']['y']
            z = data['data']['z']
            self.data['magnetometer'].append([time, x, y, z])
            return True
        elif (sensor_type == u"SENSOR_BAROMETER"):
            val = data['data']['value']
            self.data['barometer'].append([time, val])
            return True
        elif (sensor_type == u"SENSOR_LIGHT"):
            val = data['data']['value']
            self.data['light'].append([time, val])
            return True

    def push_slices(self, data):
        """
        Add a slice to the window and return false if the window is full
        Return the data without the slice
        """
        if (self.data != {'magnetometer': [], 'barometer': [], 'light': []} or len(data['magnetometer']) == 0):
            return False
        else:
            def sizeIndex(ls, start, size):
                i = 0
                while(i < len(ls) and ls[i,0] - start < size):
                    i += 1
                return i
            self.start = data['magnetometer'][0,0]
            magIndex = sizeIndex(data['magnetometer'], self.start, self.size)
            barIndex = sizeIndex(data['barometer'], self.start, self.size)
            lightIndex = sizeIndex(data['light'], self.start, self.size)
            self.data['magnetometer'] = data['magnetometer'][:magIndex]
            self.data['barometer'] = data['barometer'][:barIndex]
            self.data['light'] = data['light'][:lightIndex]
            return {'magnetometer': data['magnetometer'][magIndex:], 'barometer': data['barometer'][barIndex:], 'light': data['light'][lightIndex:]}

    def allCheck(self):
        """
        Check if there is enough data
        """
        return 2 <= min([len(self.data['magnetometer']), len(self.data['barometer']), len(self.data['light'])])

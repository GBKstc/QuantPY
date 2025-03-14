# -*- coding: UTF-8 -*-

import threading
import time

def hello():
    print("hello, Timer")

if __name__ == '__main__':
    t = threading.Timer(1.0, hello)
    t.start()
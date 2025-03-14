# -*- coding: UTF-8 -*-

from threading import Thread
import time
class MyThread(Thread):
        def __init__(self):
                super(MyThread, self).__init__()
        def run(self):
                time.sleep(1)
                print("我是子线程：" + self.name)
 
if __name__ == "__main__":
        t1=MyThread()
        t1.daemon = True
        t1.start()
print("我是主线程！")
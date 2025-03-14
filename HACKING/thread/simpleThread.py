from threading import Thread

class SimpleCreator():
  def f(self,num):
    print("线程执行"+str(num))
    return
  
  def __init__(self):
    print("初始化")
    return

  def createThread(self):
    for i in range(3):
      Thread(target=self.f,args=(i,)).start()

if __name__ == "__main__":
  SimpleCreator().createThread()

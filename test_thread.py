import threading
import mythread


t2 = threading.Thread(target=mythread.pro)
t3 = threading.Thread(target=mythread.det)
t2.start()
t3.start()

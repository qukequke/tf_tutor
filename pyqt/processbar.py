from PyQt5.QtWidgets import QApplication, QWidget, QProgressBar, QPushButton, QDesktopWidget, QMessageBox
from PyQt5.QtCore import QBasicTimer, QCoreApplication
from PyQt5.QtGui import QIcon
import sys


class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.center()
        self.initUI()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def initUI(self):
        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(30, 30, 200, 25) #窗口位置 和进度条长宽
        self.btn = QPushButton('开始', self) #设置按钮
        self.btn.setToolTip('You can start or stop the processbar') #显示一个提示文本
        self.btn.move(50, 90) #按钮位置
        self.timer = QBasicTimer()
        self.step = 0
        # self.setGeometry(200, 300, 280, 170)
        self.setWindowTitle('haha')  #设置标题
        self.setWindowIcon(QIcon('1.jpg')) # 设置图标
        # if self.step < 100:
        # self.btn.clicked.connect(self.haha)
        self.btn.clicked.connect(self.action)
        self.show()

    def haha(self):
        print('haha')

    def action(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn.setText('start')
            print(self.step)
        elif self.timer.isActive()==False:
            self.timer.start(100, self)
            self.btn.setText('stop')
            print(self.step)

    def timerEvent(self, *args, **keargs): #重写计时器
        if self.step > 100:
            self.timer.stop()
            self.btn.setText('quit')
            return
        elif self.step == 100:
            self.btn.clicked.connect(QCoreApplication.instance().quit)
        self.step += 5  #重写了，以前是1S加1  你可以自己设置了
        self.pbar.setValue(self.step) #把timer的值给到进度条 连接起来

    def closeEvent(self, event): #这个方法 点x的时候会自动调用，重写一下
        QMessageBox.question(self, 'This is a message', 'Are you sure you want to quit ?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

app = QApplication(sys.argv)
ex = Example()
sys.exit(app.exec_())



class UsbStage():

    def __init__(self, comport):
        self.comport = comport

    def getComport(self):
        return self.comport

    def getName(self):
        return self.getComport()



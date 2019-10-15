#!python
# coding=utf-8
"""
Author: LS, WQ
Date: 2019/03/01
channel.py provides the definition of class Channel
"""
from config import options
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

np.random.seed(8)


class Channel(object):
    """
    Class Channel is to sim the channel(wireless, wired? )
    """

    def __init__(self, channelType='slow', initCC=2.5, initSNR=1.5): 
        self.type = channelType
        # Set the distriuted BW
        self.disCC = initCC

        self.initSNR = initSNR

        self.capacityList = []

        self.delay = 0.0
        self.jitter = 0
        self.loss = 0.0
        self.SNR = initSNR
        self.t = 0

    def getCC(self):
        return self.disCC

    def getSNR(self):
        return self.SNR

    def setSlotCapacity(self, capacitylist):
        """
        Set the capacity of the channel in a Slot time
        :param capacitylist:
        :return:
        """
        assert len(capacitylist) == options.JudgeDuration, "setSlotCapacity func parameter capacitylist error"
        self.capacityList = capacitylist

    def updateChannel(self):
        capacity = self.disCC
        return capacity

    def updateSNR(self):
        norm = np.random.normal(loc=self.SNR, scale=0.015)
        self.SNR = np.clip(norm, a_min=0.5, a_max=options.maxSNR)
        return self.SNR

    def setDisCC(self, newDisCC):
        self.disCC = newDisCC

    def updateJitter(self):
        # Define the fun to modify the jitter in time
        self.jitter = np.random.uniform(0.0, 0.0)

        return self.jitter

    def updateLoss(self):
        # Define the fun to modify the loss in time
        self.loss = np.random.uniform(0.0, 0.05)

        return self.loss

    def updateDelay(self, frameAmount=0.0):
        # Define the fun to modify the delay in time
        self.delay = frameAmount / self.disCC
        self.delay = 0.0

        return self.delay

    def plotCapacity(self, timeSpan=300 * options.JudgeDuration):
        insCapacityList = []
        for t in range(timeSpan):
            insCapa = self.updateChannel()
            insCapacityList.append(insCapa)

        # ploting
        print("insC", np.mean(insCapacityList))
        plt.plot(insCapacityList)
        plt.axis([0, timeSpan, 0, 34])
        plt.show()

    def plotSNR(self, timeSpan=1000 * options.JudgeDuration):
        SNRList = []
        for t in range(timeSpan):
            SNR = self.updateSNR()
            SNRList.append(SNR)
            print(SNR)

            if t % 1000 == 0:
                # ploting
                plt.plot(SNRList)
                print("SNR mean:", np.mean(SNRList))
                plt.ylim([0, 3.5])
                plt.show()
                SNRList = []

    def __str__(self):
        return "{0}; {1}".format(self.type, self.SNR)


if __name__ == "__main__":
    channel = Channel()
    # channel.plotCapacity()
    channel.plotSNR()
    pass

#!python
# coding=utf-8
"""
Author: LS, WQ
Date: 2019/03/01
video.py provides the definition of class Video
"""

from config import options


class Video(object):
    def __init__(self, frames=10000, resolution=6):
        # self.videoID = videoID
        self.frames = frames
        self.preResolution = resolution
        self.curResolution = resolution

        # Record the remained video frames
        self.remainedFrames = self.frames

        # Judge whether the video has been played over
        self.isVideoEnd = False

    def updateRemainFrames(self, receivedFrames):
        # Update the remainedFrames
        self.remainedFrames -= receivedFrames
        # Check whether the video has been played over
        if self.remainedFrames <= 0:
            # self.isVideoEnd = True
            self.remainedFrames = self.frames

    def getRemainFrames(self):
        return self.remainedFrames

    def updateResolution(self, newResolution):
        self.preResolution = self.curResolution
        self.curResolution = newResolution


class Buffer(object):
    def __init__(self, preResolution, size=options.bufferSize):
        self.size = size
        self.data = []
        self.data.append([preResolution, 2 * preResolution])
        self.amount = 0.0

    def setSize(self, newSize):
        self.size = newSize

    def updateAmount(self):
        self.amount = 0.0
        for videoItem in self.data:
            self.amount += videoItem[1]

    def isFull(self, ins_Capacity):
        """
        Check whether the buffer is full
        ins_Capacity : instant capacity of the corresponding channel
        """
        self.updateAmount()
        if self.amount + ins_Capacity > self.size:
            return True
        else:
            return False

    def isEmpty(self):
        SegmentIndex = 0
        while len(self.data) > 2:
            bufferSegment = self.data[SegmentIndex]
            if bufferSegment[1] >= bufferSegment[0]:  # 一帧图片
                return False
            else:
                del self.data[SegmentIndex]  # 不足一帧图片的数据丢弃
        assert len(self.data) == 2
        while SegmentIndex < len(self.data):
            bufferSegment = self.data[SegmentIndex]
            if bufferSegment[1] >= bufferSegment[0]:
                return False
            else:
                SegmentIndex += 1
        return True

    def update(self, deltaDict):
        """
        Update the buffer , include the increase data and del data from buffer
        Download and play the video towards every time-unit during one slot
        deltaDict: the dict to record the change towards all variants
        """
        # 1. Get the serial number of the timeUnit
        t = deltaDict.get("t")
        # 2. Adding data to the buffer
        addDelta = deltaDict.get("increase")  # 一个unit要增加的数据
        self.data[-1][1] += addDelta  # 假数据
        self.updateAmount()  # 更新buffer总量
        # 3. Getting the curFormat
        curResolution = deltaDict.get("curResolution")
        # 3. Get the state of the buffer
        buffState = deltaDict.get("bufferState")
        # print("t:{:>2d} | {:^6s} | IN {:^5s}:{:>7.4f} | BF:{:>6.2f} => "
        #       .format(t, buffState, curResolution, addDelta, self.amount), end='')
        # self.printInfo()
        # 4. Play the video as usually
        playForamt = ""
        playAmount = 0.0
        if deltaDict.get("isDecrease"):  # 是否播放
            # print(deltaDict.get("isDecrease"))
            flag = True
            while len(self.data) > 2:
                bufferSegment = self.data[0]
                if bufferSegment[1] + 0.1 >= bufferSegment[0]:
                    self.data[0][1] -= bufferSegment[0]
                    playForamt = bufferSegment[0]
                    playAmount = bufferSegment[0]
                    flag = False
                    break
                else:
                    del self.data[0]  # 丢弃

            segmentIndex = 0
            while flag:
                bufferSegment = self.data[segmentIndex]
                if bufferSegment[1] + 0.1 >= bufferSegment[0]:
                    self.data[segmentIndex][1] -= bufferSegment[0]
                    if self.data[segmentIndex][1] <= 0:
                        self.data[segmentIndex][1] = 0

                    playForamt = bufferSegment[0]
                    playAmount = bufferSegment[0]
                    break
                else:
                    if segmentIndex <= 0:
                        self.data[segmentIndex][1] = 0
                        self.updateAmount()
                        segmentIndex += 1
                    elif segmentIndex == 1:
                        segmentIndex += 1
                        pass
                    elif segmentIndex >= 2:
                        bufferData = self.data
                        print("buffer amount:{:7.2f}".format(self.amount))
                        print("buffer[0]:{:5s}:{:7.2f}".format(bufferData[0][0], bufferData[0][1]))
                        print("buffer[1]:{:5s}:{:7.2f}".format(bufferData[1][0], bufferData[1][1]))
                        raise "segmentIndex error"
                    else:
                        raise "errrrrrrrr"

        # 5. Update the buffer amount
        self.updateAmount()

        # 6. Print the help info
        # print("t:{:>2d} | {:^6s} | IN {:^5s}:{:>7.4f} | OUT {:^5s}:{:>7.4f} | BF:{:>6.2f} => "
        #       .format(t, buffState, curResolution, addDelta, playForamt, playAmount, self.amount), end='')
        # self.printInfo()

    def printInfo(self):
        # Print the detailed info of the buffer
        for index, bufferitem in enumerate(self.data):
            if index < len(self.data) - 1:
                print("{:^5s}:{:>6.2f}".format(bufferitem[0], bufferitem[1]), end='  ')
            else:
                print("{:^5s}:{:>6.2f}".format(bufferitem[0], bufferitem[1]))


# if __name__ == '__main__':
#     bf = Buffer(preResolution='360P')
#     bf.data = [["360P", 6.08], ["1080P", 19.33], ["1080P", 18.55], ["1080P", 38.60],
#                ["1080P", 38.12], ["1080P", 34.84], ["480P", 32.34], ["360P", 0.00]]
#     delta = {}
#     delta["t"] = 1
#     delta["bufferState"] = "full"
#     delta["increase"] = 0
#     delta["curResolution"] = "360P"  # clientResolution
#     delta["isDecrease"] = True
#
#     bf.update(delta)
#
#     # print(bf)
#     pass



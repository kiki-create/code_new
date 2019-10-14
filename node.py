#!python
# coding=utf-8
"""
Author: LS, WQ
Date: 2019/03/01
node.py provides the class Server, Client
"""
import numpy as np
from video import Video, Buffer
from config import options
import lib
import copy


class Server(object):
    def __init__(self, name, initChannelCapacity=1000, maxConnections=10):  # todo: initCC
        self.name = name
        self.maxConnections = maxConnections
        self.ChannelCapacity = initChannelCapacity

        # Information of clients
        self.clients = {}

    def addClient(self, client):
        assert isinstance(client, Client), \
            "The parameter client is not the instance of class Client"
        assert len(self.clients) <= self.maxConnections, \
            "The num of connections towards Server is up to maxConnections"
        self.clients[client.name] = client

    def delClient(self, clientName):
        client = self.clients.get(clientName, None)
        if not client:
            del self.clients[clientName]
        else:
            print("The client to be del is no existing!")


class Client(object):
    def __init__(self, name, channel, video=None, bufferSize=10 * 6):
        self.name = name
        self.channel = channel
        self.video = video

        self.buffer = Buffer(self.video.preResolution, size=bufferSize)

        self.bitrate = 0

        # attribute isalive is related to the status of video transporting
        self.isalive = True

    def getChannel(self):
        return self.channel

    def setChannel(self, newChannel):
        self.channel = newChannel

    def getVideo(self):
        return self.video

    def addVideo(self, video):
        assert isinstance(video, Video), "The video is not instance of class Video!"

        assert self.video is not None, \
            "The video of the client {0} is None".format(self.name)

        self.video = video

    def getBuffer(self):
        return self.buffer

    def updateVideo(self, AIDistributionScheme):
        """
        Simulation: Transport the video to the client
        """
        assert isinstance(AIDistributionScheme, dict), "The type of AIDistributionScheme is unmatch!"

        # 1. Unpack the AIDistributionScheme by items
        clientCC = AIDistributionScheme.get('CC', self.channel.getCC())
        clientResolution = AIDistributionScheme.get("RR", self.video.curResolution)

        # 2. Execute the distribution scheme by AI
        self.getChannel().setDisCC(clientCC)
        self.getVideo().updateResolution(clientResolution)
        self.getBuffer().data.append([clientResolution, 0])

        # 4. Define the Variant
        # Record the video info for every time-unit
        clientSlotInfo = {}

        # Record the video Amount during one slot
        videoAmountSlot = 0.0

        # 5. Ergodic the time-unit during one slot
        self.isEmpty = False
        self.isFull = False

        # Set the controlTime var
        emptyTime, fullTime = 0.0, 0.0
        pauseTime = 0  # options.pauseTime

        tmp = self.buffer.data
        for t in range(options.JudgeDuration):
            clientUnitInfo = {}
            delta = {}

            aa = self.buffer.data[-2][1]
            if self.buffer.data[-2][1] > tmp[-2][1]:
                a = 0
                pass
            if self.buffer.amount > 100:
                a = 0
                pass

            # 5.1 Update the bandWidth of Channel in time
            timeUnitCapacity = self.getChannel().updateChannel()
            clientUnitInfo['disCC'] = timeUnitCapacity

            # 5.2 Preprocess the buffer

            # Set the Swtich Flag among normal, full and empty
            # if not self.isEmpty and not self.isFull:
            if pauseTime <= 0:
                self.isEmpty = False
                self.isFull = False
                self.isEmpty = self.getBuffer().isEmpty()
                self.isFull = self.getBuffer().isFull(timeUnitCapacity)
                if self.isEmpty or self.isFull:
                    pauseTime = options.pauseTime

            if self.isEmpty:
                # Record time
                pauseTime -= 1
                emptyTime += 1
                # 3. Delta-variant
                delta["t"] = t
                delta["bufferState"] = "empty"
                delta["increase"] = timeUnitCapacity
                delta["curResolution"] = clientResolution
                delta["isDecrease"] = False
                # 4. Update the videoAmountSlot
                videoAmountSlot += timeUnitCapacity

            elif self.isFull:
                # Record time
                pauseTime -= 1
                fullTime += 1
                # 3. Delta-variant
                delta["t"] = t
                delta["bufferState"] = "full"
                delta["increase"] = 0
                delta["curResolution"] = clientResolution
                delta["isDecrease"] = True
                # 4. Update the videoAmountSlot
                videoAmountSlot += 0

            # if not self.isEmpty and not self.isFull:
            else:
                # 1. Delta-variant
                delta["t"] = t
                delta["bufferState"] = "normal"
                delta["increase"] = timeUnitCapacity
                delta["curResolution"] = clientResolution
                delta["isDecrease"] = True
                # 2. Update the videoAmountSlot
                videoAmountSlot += timeUnitCapacity

            # 5.3 Update the buffer
            self.getBuffer().update(deltaDict=delta)
            # Record the information of the timeUnit
            clientUnitInfo["SNR"] = self.getChannel().getSNR()
            clientUnitInfo['buffer'] = copy.deepcopy(self.getBuffer())

            # 5.4 Update and store the info of median process
            clientSlotInfo[str(t)] = copy.deepcopy(clientUnitInfo)

        # Store the emptyTime
        clientSlotInfo["reso"] = self.video.curResolution
        clientSlotInfo["emptyTime"] = emptyTime
        clientSlotInfo["fullTime"] = fullTime
        clientSlotInfo["disCC"] = self.getChannel().getCC()
        clientSlotInfo["CC_utility"] = videoAmountSlot / options.JudgeDuration

        # Update the video frames
        receivedFrames = videoAmountSlot / self.getVideo().curResolution
        self.getVideo().updateRemainFrames(receivedFrames)
        if self.getVideo().getRemainFrames() <= 0:
            self.isalive = False

        return clientSlotInfo


if __name__ == "__main__":
    pass


#!python
# coding=utf-8
"""
Author: LS, WQ
Date: 2019/03/01
mynet.py provides the class MyNet,
which creates the customosized net-topo according to the need!
"""
from channel import Channel
from config import options
from node import Server, Client
from video import Video


class myNet(object):
    def __init__(self, hostNum=4):
        # Set the total BW of the central-server
        self.serverCC = options.serverCC
        self.hostNum = hostNum
        # self.clients = [_ for _ in range(self.hostNum)]
        self.clients = {}  # hostname:host
        self.step = 0

    def createNetTopo(self):
        # 1 Create the Server
        self.server = Server(name="Server")
        initSNR = [1.8, 1.43, 1, 0.9]
        # 2 Create the clients
        resoList = [6, 4, 2, 1]
        for i in range(self.hostNum):
            # init_reso = resoList[np.random.randint(4)]
            init_reso = resoList[i]
            # init_reso = resoList[len(resoList) - 1 - i]
            c = Client(name="c{}".format(i + 1),
                       channel=Channel(initCC=options.serverCC / 4, initSNR=initSNR[i]),
                       video=Video(resolution=init_reso),
                       )
            self.clients[c.name] = c

    def getClient(self, name):
        return self.clients.get(name, None)

    def addClient(self, client):
        assert isinstance(client, Client), \
            "The parameter client is not the instance of class Client"
        assert len(self.clients) <= self.server.maxConnections, \
            "The num of connections towards Server is up to maxConnections"
        self.clients[client.name] = client

    def delClient(self, clientName):
        client = self.clients.get(clientName, None)
        if not client:
            del self.clients[clientName]
        else:
            print("The client to be del is no existing!")

    def updateClientChannelCapacity(self):
        """
        Update the capacity of the client channel
        :return:
        """
        # 1. Create a dict object to store the channelCapacity info
        allClientChannelCapacity = {}
        for index in range(options.HostNum):
            clientName = 'c' + str(index + 1)
            allClientChannelCapacity[clientName] = []
        # 2. ergode the Slot
        for t in range(options.JudgeDuration):
            restCC = options.serverCC
            for index in range(1, options.HostNum):
                clientName = 'c' + str(index)
                clientObj = self.getClient(name=clientName)
                assert isinstance(clientObj, Client), "updateClientChannelCapacity get clientObj error"
                capa = clientObj.getChannel().updateChannelCapacity()
                allClientChannelCapacity.get(clientName).append(capa)
                restCC -= capa

            # Process the last client:d = 54 - a - b - c
            clientName = 'c' + str(options.HostNum)
            clientObj = self.getClient(name=clientName)
            assert isinstance(clientObj, Client), "updateClientChannelCapacity get clientObj error"
            capa = restCC
            allClientChannelCapacity.get(clientName).append(capa)

        # 3. Update the capacityList to the clientObj
        for clientName, clientObj in self.clients.items():
            capacityList = allClientChannelCapacity.get(clientName)
            clientObj.getChannel().setSlotCapacity(capacityList)

    def updateClientVideo(self, DistributionByAI={}):
        """
        Simulate to transport the video for each client

        """
        clientsExecResult = {}
        # Before Slot
        print("=" * 100, "step:", self.step, "=" * 100)
        self.step += 1
        for clientName, clientObj in self.clients.items():
            # distribution result by the AI algorithm
            clientDistributionByAI = DistributionByAI.get(clientName, {})
            clientsExecResult[clientName] = clientObj.updateVideo(clientDistributionByAI)

        # After Slot
        # Check whether there are some non-alive clients
        for clientName, clientObj in self.clients.items():
            if clientObj.isalive is False:
                # if not isalive, remove the client from clients and clientsExecResult
                self.delClient(clientName)
                clientsExecResult.pop(clientName)

        return clientsExecResult


if __name__ == "__main__":
    pass



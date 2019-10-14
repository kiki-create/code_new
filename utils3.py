# coding=utf-8
import lib
import numpy as np
from config import options

def env_state8(clientExecResult):
    env_state = []

    SNRList = [[] for i in range(options.HostNum)]
    CCList = [[] for i in range(options.HostNum)]
    disCC_percentList = [[] for i in range(options.HostNum)]
    RRList = [[] for i in range(options.HostNum)]
    BFpreList = [[] for i in range(options.HostNum)]
    BFcurList = [[] for i in range(options.HostNum)]

    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = clientExecResult[clientName]

        for t in range(options.JudgeDuration):
            SNRList[index].append(clientInfo.get(str(t)).get("SNR"))
            CCList[index].append(clientInfo.get(str(t)).get("disCC"))
            RRList[index].append(clientInfo.get(str(t)).get("reso"))
            buffer = clientInfo.get(str(t)).get('buffer')
            buffer_cur = buffer.data[-1][1]
            buffer_pre = buffer.amount - buffer.data[-1][1]
            BFcurList[index].append(buffer_cur)
            BFpreList[index].append(buffer_pre)

    # The info of SNR
    SNR_array = np.array(SNRList)
    meanSNR = SNR_array.mean(axis=1)

    # The info of CC
    CC_array = np.array(CCList)
    meanCC = CC_array.mean(axis=1)
    sumCC = CC_array.sum(axis=0)
    CC_percent = CC_array / sumCC
    meanCC_percent = meanCC / sumCC.mean()

    # The info of RR
    RR_array = np.array(RRList)
    meanRR = RR_array.mean(axis=1)

    # The info of buffer
    bf_pre_array = np.array(BFpreList)
    bf_cur_array = np.array(BFcurList)
    bf_pre_percent = bf_pre_array / options.bufferSize
    bf_cur_percent = bf_cur_array / options.bufferSize

    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = clientExecResult[clientName]

        disCC = clientInfo.get('disCC')
        disCC_percent = disCC / options.serverCC
        CC_utility = clientInfo.get("CC_utility")
        RR = clientInfo.get('reso')
        CR = (disCC, RR)
        CR_index = lib.CR_mapping.index(CR) / len(lib.CR_mapping)  # todo

        fullTime = clientInfo.get("fullTime") / options.JudgeDuration
        emptyTime = clientInfo.get("emptyTime") / options.JudgeDuration

        client_info = [meanSNR[index], meanCC[index], meanRR[index], bf_pre_percent[index][-1],
                       bf_cur_percent[index][-1], fullTime, emptyTime, disCC_percent[index]]

        env_state.append(client_info)
    c1_state, c2_state, c3_state, c4_state = env_state[0], env_state[1], env_state[2], env_state[3]
    print("SNR_array", SNR_array)
    print("meanSNR[0]", meanSNR[0])
    print("meanCC[0]", meanCC[0])
    print("meanRR[0]", meanRR[0])
    print("bf_pre_percent[0][-1]", bf_pre_percent[0][-1])
    print("bf_cur_percent[0][-1]", bf_cur_percent[0][-1])
    print("fullTime", fullTime)
    print("disCC_percent[0]", disCC_percent[0])

    return env_state, c1_state, c2_state, c3_state, c4_state
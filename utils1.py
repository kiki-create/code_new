# coding=utf-8
import numpy as np
from operator import itemgetter
from config import options

buffer_reso = [[] for i in range(options.HostNum)]
buffer_CR = [[] for i in range(options.HostNum)]
buffer_RR = [[] for i in range(options.HostNum)]


def env_state8(clientExecResult):
    env_state = []

    SNRList = [[] for i in range(options.HostNum)]

    BFpreList = [[] for i in range(options.HostNum)]
    BFcurList = [[] for i in range(options.HostNum)]

    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = clientExecResult[clientName]

        for t in range(options.JudgeDuration):
            SNRList[index].append(clientInfo.get(str(t)).get("SNR"))
            buffer = clientInfo.get(str(t)).get('buffer')
            buffer_cur = buffer.data[-1][1]
            buffer_pre = buffer.amount - buffer.data[-1][1]
            BFcurList[index].append(buffer_cur)
            BFpreList[index].append(buffer_pre)

    # The info of SNR
    SNR_array = np.array(SNRList)
    SNRDict = {}
    meanSNR = SNR_array.mean(axis=1)
    for i in range(options.HostNum):
        SNRDict["c"+str(i)] = meanSNR[i]
    # 将SNR排序
    sortedSNR = sorted(SNRDict.items(), key=itemgetter(0))
    # todo: 模拟路由器QoS控制
    # The info of buffer
    bf_pre_array = np.array(BFpreList)
    bf_cur_array = np.array(BFcurList)
    bf_pre_percent = bf_pre_array / options.bufferSize
    bf_cur_percent = bf_cur_array / options.bufferSize

    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = clientExecResult[clientName]

        # the info of CC
        disCC = clientInfo.get('disCC')
        disCC_percent = disCC / options.serverCC
        # The info of RR
        RR = clientInfo.get('reso')

        fullTime = clientInfo.get("fullTime") / options.JudgeDuration
        emptyTime = clientInfo.get("emptyTime") / options.JudgeDuration

        client_info = [meanSNR[index], disCC, RR, bf_pre_percent[index][-1], bf_cur_percent[index][-1], fullTime, emptyTime, disCC_percent]

        env_state.append(client_info)
    c1_state, c2_state, c3_state, c4_state = env_state[0], env_state[1], env_state[2], env_state[3]

    return env_state, c1_state, c2_state, c3_state, c4_state


def reward_joint3(ExecResult):
    reward_cr_max = 100.0

    S_reward = [0.0 for _ in range(options.HostNum)]
    if len(buffer_CR[0]) >= 6:   # 存之前的5段历史
        for i in range(options.HostNum):
            del buffer_CR[i][0]

    if len(buffer_RR[0]) >= 6:   # 存之前的5段历史
        for i in range(options.HostNum):
            del buffer_RR[i][0]


    all_fulltime = 0.0
    all_emptytime = 0.0
    qoe_list = [[] for _ in range(options.HostNum)]
    all_snr = [[] for _ in range(options.HostNum)]
    SNR = [[] for _ in range(options.HostNum)]


    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = ExecResult[clientName]

        emptyTime = clientInfo.get('emptyTime')
        fullTime = clientInfo.get('fullTime')
        disCC = clientInfo.get("disCC")
        RR_uni = clientInfo.get("reso") / 13

        buffer_RR[index].append(RR_uni)
        CR = (disCC, RR_uni)
        buffer_CR.append(CR)
        a = np.array(buffer_RR[index])
        RR_var = np.var(a)

        SNRList = []
        for t in range(options.JudgeDuration):
            SNRList.append(clientInfo.get(str(t)).get("SNR"))
        SNR = np.array(SNRList)
        all_snr.append(SNR)

        qoe = 5 * disCC - fullTime * 1 - emptyTime * 2 - 2 * RR_var
        # weighted_qoe = qoe * SNR[-1]  # todo: index
        weighted_qoe = qoe
        qoe_list[index].append(weighted_qoe)


        all_fulltime += fullTime
        all_emptytime += emptyTime

        S_reward[index] += weighted_qoe


    print("qoelist\n", qoe_list)

    qoeall = SNR[0] * qoe_list[0][-1] + SNR[1] * qoe_list[1][-1] + SNR[2] * qoe_list[2][-1] + SNR[3] * qoe_list[3][-1]
    reward_cr1 = qoe_list[0][-1] + qoeall
    reward_cr2 = qoe_list[1][-1] + qoeall
    reward_cr3 = qoe_list[2][-1] + qoeall
    reward_cr4 = qoe_list[3][-1] + qoeall
    reward_cr1_uni = reward_cr1 / reward_cr_max
    reward_cr2_uni = reward_cr2 / reward_cr_max
    reward_cr3_uni = reward_cr3 / reward_cr_max
    reward_cr4_uni = reward_cr4 / reward_cr_max
    print("reward_cr1:", reward_cr1)
    print("reward_cr2:", reward_cr2)
    print("reward_cr3:", reward_cr3)
    print("reward_cr4:", reward_cr4)

    return reward_cr1_uni, reward_cr2_uni, reward_cr3_uni, reward_cr4_uni

def reward_joint2(ExecResult):    #  改最大和最小限制/加入emptytime 限制
    reward_cr_max = 100.0

    S_reward = [0.0 for _ in range(options.HostNum)]
    if len(buffer_CR[0]) >= 6:   # 存之前的5段历史
        for i in range(options.HostNum):
            del buffer_CR[i][0]

    if len(buffer_RR[0]) >= 6:   # 存之前的5段历史
        for i in range(options.HostNum):
            del buffer_RR[i][0]


    all_fulltime = 0.0
    all_emptytime = 0.0
    qoe_list = [[] for _ in range(options.HostNum)]
    all_snr = [[] for _ in range(options.HostNum)]
    SNR = [[] for _ in range(options.HostNum)]


    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = ExecResult[clientName]

        emptyTime = clientInfo.get('emptyTime')
        fullTime = clientInfo.get('fullTime')
        disCC = clientInfo.get("disCC")
        RR_uni = clientInfo.get("reso") / 13
        capa_percent = disCC / options.serverCC

        buffer_RR[index].append(RR_uni)
        CR = (disCC, RR_uni)
        buffer_CR.append(CR)
        a = np.array(buffer_RR[index])
        RR_var = np.var(a)

        if capa_percent < 0.1 or capa_percent > 0.5:
            punish = 0.5
        else:
            punish = 1

        SNRList = []
        for t in range(options.JudgeDuration):
            SNRList.append(clientInfo.get(str(t)).get("SNR"))
        SNR = np.array(SNRList)
        all_snr.append(SNR)



        qoe = 5 * disCC - fullTime * 1 - emptyTime * 5 - 2 * RR_var
        weighted_qoe = qoe * punish
        qoe_list[index].append(weighted_qoe)


        all_fulltime += fullTime
        all_emptytime += emptyTime

        S_reward[index] += weighted_qoe


    print("qoelist\n", qoe_list)

    qoeall = SNR[0] * qoe_list[0][-1] + SNR[1] * qoe_list[1][-1] + SNR[2] * qoe_list[2][-1] + SNR[3] * qoe_list[3][-1]
    reward_cr1 = qoe_list[0][-1] + qoeall
    reward_cr2 = qoe_list[1][-1] + qoeall
    reward_cr3 = qoe_list[2][-1] + qoeall
    reward_cr4 = qoe_list[3][-1] + qoeall
    reward_cr1_uni = reward_cr1 / reward_cr_max
    reward_cr2_uni = reward_cr2 / reward_cr_max
    reward_cr3_uni = reward_cr3 / reward_cr_max
    reward_cr4_uni = reward_cr4 / reward_cr_max
    print("reward_cr1:", reward_cr1)
    print("reward_cr2:", reward_cr2)
    print("reward_cr3:", reward_cr3)
    print("reward_cr4:", reward_cr4)

    return reward_cr1_uni, reward_cr2_uni, reward_cr3_uni, reward_cr4_uni




if __name__ == '__main__':
    pre_action = [1.67 for i in range(options.HostNum)]
    print(pre_action)

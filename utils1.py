# coding=utf-8
"""
提供一些对环境操作或设置reward的工具函数
"""
import numpy as np
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


def unitEnv_uni(clientExecResult):
    """
    :param clientExecResult:
    :return: 各个用户的snr list
    """
    SNRList = [[] for i in range(options.HostNum)]

    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = clientExecResult[clientName]

        for t in range(options.JudgeDuration):
            # Normalize the SNR of channel
            SNR = clientInfo.get(str(t)).get("SNR")
            SNR_norm = SNR / options.maxSNR
            SNRList[index].append(SNR_norm)

    return np.array(SNRList)


def get_snr(clientExecResult):
    """
    获取每个用户的snr
    :param clientExecResult:
    :return: 各个用户的snr dict
    """
    SNRList = {}
    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = clientExecResult[clientName]
        snr = clientInfo.get(str(0)).get("SNR")
        SNRList[clientName] = snr
    return SNRList


def adjust_CC(disCC, snr):
        """
        根据带宽调整CC，模拟路由器带宽分配规则
        :param disCC: 神经网络分配的CC list
        :param snr: 用户的snr dictionary
        :return: 每个用户的传输数据的实际速率
        """
        disBW = {}  # 按照公式算的每个人的bw
        BW_rank = {}
        CC_real = []
        for i in range(4):
            disBW["c" + str(i + 1)] = (disCC[i] / np.log2(1 + snr["c" + str(i + 1)]))
            BW_rank["c" + str(i + 1)] = 1
        print("disBW", disBW)
        sorted_Snr = sorted(snr.items(), key=lambda x: x[1], reverse=True)  # 按照snr从高到低排序
        print("sorted_Snr", sorted_Snr)
        totalBW = 20 - 4
        for i in range(4):
            bw = disBW.get(sorted_Snr[i][0])
            # 在满足所有人要求的条件下，把多余的带宽均匀分给每个人
            if i == 3:
                if totalBW + 1 > bw:
                    BW_rank[sorted_Snr[i][0]] = bw
                    redundant_bw = totalBW - bw + 1
                    for i in range(4):
                        BW_rank["c" + str(i + 1)] += redundant_bw / 4
                    break
                else:
                    BW_rank[sorted_Snr[i][0]] = totalBW + 1
                    break
            if totalBW > 0:
                if bw >= 1 and totalBW >= bw and bw <= 20 / 2:  # 不用对“分配”的BW处理的情况
                    totalBW = totalBW - bw + 1  # +1 是把垫的1M补回来
                    BW_rank[sorted_Snr[i][0]] = bw
                    continue
                else:
                    if bw > 20 / 2 or bw > totalBW:
                        bw = min(totalBW + 1, 20 / 2)
                        totalBW = totalBW - bw + 1  # +1 是把垫的1M补回来
                        BW_rank[sorted_Snr[i][0]] = bw
                        continue
            else:
                break
        print(BW_rank)
        for i in range(4):
            CC_real.append(BW_rank["c" + str(i + 1)] * np.log2(1 + snr["c" + str(i + 1)]))
        return CC_real

if __name__ == '__main__':
    disCC = [20, 20, 20, 20]
    snr = {"c1": 1, "c2": 2, "c3": 3, "c4": 4}
    print(adjust_CC(disCC, snr))

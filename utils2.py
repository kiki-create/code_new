# coding=utf-8
from config import options
import numpy as np
import lib
import copy

buffer_reso = [[] for i in range(options.HostNum)]
# reso_mapping = ["360P", '480P', "720P", "1080P"]
reso_mapping = ["1080P", '720P', "480P", "360P"]


buffer_reso_norm = [[] for i in range(options.HostNum)]
buffer_reso_avr = [[] for i in range(options.HostNum)]
buffer_reso_greedy = [[] for i in range(options.HostNum)]


def getRealC(clientExecResult):
    allC = []
    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = clientExecResult[clientName]
        disBW = clientInfo.get("disBW")
        SNRList = []
        for t in range(options.JudgeDuration):
            # Normalize the SNR of channel
            SNR = clientInfo.get(str(t)).get("SNR")
            SNRList.append(SNR)
        disC = disBW * np.array(SNRList)
        allC.append(disC)

    return np.array(allC)


def unitEnv1(clientExecResult):
    SNRList = [[] for i in range(options.HostNum)]

    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = clientExecResult[clientName]

        for t in range(options.JudgeDuration):
            # Normalize the SNR of channel
            SNR = clientInfo.get(str(t)).get("SNR")
            SNR_norm = SNR / 1.0  #  options.maxSNR
            SNRList[index].append(SNR_norm)

    return np.array(SNRList)


def unitEnv_uni(clientExecResult):
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


def unitEnv2(clientExecResult):
    env_state = []
    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = clientExecResult[clientName]
        reso = clientInfo.get('reso')
        resoIndex = reso_mapping.index(reso)
        fullTime = clientInfo.get("fullTime")
        emptyTime = clientInfo.get("emptyTime")
        disBW = clientInfo.get("disBW")
        disBW_percent = disBW / options.serverBW
        BW_utility = clientInfo.get("BW_utility")


        clientUnitInfo = clientInfo[str(options.JudgeDuration - 1)]
        capacityRate = clientUnitInfo['capacity'] / options.serverBW

        buffer = clientUnitInfo.get('buffer')

        buffer = clientUnitInfo.get("buffer")
        buffer_cur = buffer.data[-1][1]
        buffer_pre = buffer.amount - buffer.data[-1][1]
        buffer_percent = buffer.amount / buffer.size
        buffer_cur_percent = buffer_cur / buffer.size
        buffer_pre_percent = buffer_pre / buffer.size

        instantBW = clientUnitInfo.get("BW")
        instantBW_percent = instantBW / options.serverBW
        SNR = clientUnitInfo.get("SNR")

        env_state.append([resoIndex, fullTime, emptyTime, disBW_percent, BW_utility, capacityRate,
                          buffer_cur_percent, buffer_pre_percent, instantBW_percent, SNR])
    c1_state, c2_state, c3_state, c4_state = env_state[0], env_state[1], env_state[2], env_state[3]

    return c1_state, c2_state, c3_state, c4_state


def reward2norm(ExecResult):
    win_len = 5
    if len(buffer_reso_norm[0]) >= win_len:
        print("the length of buffer_reso is :", win_len)
        for i in range(options.HostNum):
            del buffer_reso_norm[i][0]

    all_fulltime = 0.0
    all_emptytime = 0.0

    all_br = 0.0

    all_break = 0

    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = ExecResult[clientName]

        emptyTime = clientInfo.get('emptyTime')
        fullTime = clientInfo.get('fullTime')

        disBW = clientInfo.get("disBW")
        reso = clientInfo.get("reso")
        resoC = lib.frameDataSize.get(reso)
        buffer_reso_norm[index].append(resoC)

        reso_index = reso_mapping.index(reso)

        reso_var = np.array(buffer_reso_norm[index]).var()
        break_times = reso_var
        all_break += break_times

        qoe = resoC - fullTime - emptyTime - np.array(buffer_reso[index]).var()

        all_fulltime += fullTime
        all_emptytime += emptyTime

        br = np.array(buffer_reso_norm[index]).mean()
        all_br += br

    rebuffer = (all_fulltime + all_emptytime) / 10 * 2
    smooth = all_break / 10
    return rebuffer, all_br, smooth


def reward2avr(ExecResult):
    win_len = 5
    if len(buffer_reso_avr[0]) >= win_len:
        print("the length of buffer_reso is :", win_len)
        for i in range(options.HostNum):
            del buffer_reso_avr[i][0]

    all_fulltime = 0.0
    all_emptytime = 0.0

    all_br = 0.0

    all_break = 0

    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = ExecResult[clientName]

        emptyTime = clientInfo.get('emptyTime')
        fullTime = clientInfo.get('fullTime')

        disBW = clientInfo.get("disBW")
        reso = clientInfo.get("reso")
        resoC = lib.frameDataSize.get(reso)
        buffer_reso_avr[index].append(resoC)

        reso_index = reso_mapping.index(reso)

        reso_var = np.array(buffer_reso_avr[index]).var()
        break_times = reso_var
        all_break += break_times

        qoe = resoC - fullTime - emptyTime - np.array(buffer_reso[index]).var()

        all_fulltime += fullTime
        all_emptytime += emptyTime

        br = np.array(buffer_reso_avr[index]).mean()
        all_br += br

    rebuffer = (all_fulltime + all_emptytime) / 10 * 2
    smooth = all_break / 10
    return rebuffer, all_br, smooth


def reward2greedy(ExecResult):
    win_len = 5
    if len(buffer_reso_greedy[0]) >= win_len:
        print("the length of buffer_reso is :", win_len)
        for i in range(options.HostNum):
            del buffer_reso_greedy[i][0]

    all_fulltime = 0.0
    all_emptytime = 0.0

    all_br = 0.0

    all_break = 0

    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = ExecResult[clientName]

        emptyTime = clientInfo.get('emptyTime')
        fullTime = clientInfo.get('fullTime')

        disBW = clientInfo.get("disBW")
        reso = clientInfo.get("reso")
        resoC = lib.frameDataSize.get(reso)
        buffer_reso_greedy[index].append(resoC)

        reso_index = reso_mapping.index(reso)

        reso_var = np.array(buffer_reso_greedy[index]).var()
        break_times = reso_var
        all_break += break_times

        qoe = resoC - fullTime - emptyTime - np.array(buffer_reso[index]).var()

        all_fulltime += fullTime
        all_emptytime += emptyTime

        br = np.array(buffer_reso_greedy[index]).mean()
        all_br += br

    rebuffer = (all_fulltime + all_emptytime) / 10 * 2
    smooth = all_break / 10
    return rebuffer, all_br, smooth


def mappingActions(action_BW, action_reso):
    new_action = {}
    for i in range(options.HostNum):
        clientName = 'c' + str(i + 1)
        clientDistribution = {}
        clientDistribution['BW'] = action_BW[i]
        clientDistribution['reso'] = reso_mapping[action_reso[i]]
        new_action[clientName] = clientDistribution

    return new_action


def mappingActions2(action_reso):
    # From probility to reso('P')
    new_action = {}
    BWList = [19.77, 8.79, 2.92, 1.67]
    reso_mapping = ["1080P", "720P", '480P', "360P"]
    for i in range(options.HostNum):
        clientName = 'c' + str(i + 1)
        clientDistribution = {}
        # clientDistribution['reso'] = lib.frameDataSize.get(reso_mapping[action_reso[i]], 15)
        # clientDistribution['reso'] = lib.frameDataSize[reso_mapping[action_reso[i]]]
        clientDistribution['reso'] = action_reso[i]
        clientDistribution['BW'] = BWList[i]
        new_action[clientName] = clientDistribution
    return new_action


def mappingActions3(action_reso):
    new_action = {}
    BWList = [39.41, 10, 2.92, 1.67]
    for i in range(options.HostNum):
        clientName = 'c' + str(i + 1)
        clientDistribution = {}
        # clientDistribution['reso'] = lib.frameDataSize.get(reso_mapping[action_reso[i]], 15)
        # clientDistribution['reso'] = lib.frameDataSize[reso_mapping[action_reso[i]]]
        clientDistribution['reso'] = action_reso[i]
        clientDistribution['BW'] = BWList[i]
        new_action[clientName] = clientDistribution
    return new_action


def env_state(unprocessedClientExecResult):
    env_state = []
    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = unprocessedClientExecResult[clientName]
        clientUnitInfo = clientInfo[str(options.JudgeDuration - 1)]
        # clientallInfo = [clientUnitInfo['capacity'], clientUnitInfo['buffer']/600, \
        #                  clientInfo['emptyTime'], clientInfo['fullTime'], int(clientInfo['reso'][:-1])]
        clientallInfo = clientUnitInfo['capacity']
        env_state.append(clientallInfo)

    return env_state


def env_state2(unprocessedClientExecResult):
    env_state = []
    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = unprocessedClientExecResult[clientName]
        clientUnitInfo = clientInfo[str(options.JudgeDuration - 1)]
        clientallInfo = [clientUnitInfo['capacity'], clientUnitInfo['buffer'] / 600, \
                         clientInfo['emptyTime'], clientInfo['fullTime'], int(clientInfo['reso'][:-1])]
        env_state.append(clientallInfo)
    return env_state


def env_state3(clientExecResult):
    env_state = []
    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = clientExecResult[clientName]
        reso = clientInfo.get('reso')
        resoIndex = reso_mapping.index(reso)

        clientUnitInfo = clientInfo[str(options.JudgeDuration - 1)]
        capacityRate = clientUnitInfo['capacity'] / options.serverBW

        buffer = clientUnitInfo.get('buffer')
        buffer_percent = buffer.amount / buffer.size

        env_state.append([capacityRate, resoIndex, buffer_percent])
    c1_state, c2_state, c3_state, c4_state = env_state[0], env_state[1], env_state[2], env_state[3]
    return c1_state, c2_state, c3_state, c4_state


def env_state4(clientExecResult):
    env_state = []
    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = clientExecResult[clientName]
        reso = clientInfo.get('reso')
        resoIndex = reso_mapping.index(reso)
        fullTime = clientInfo.get("fullTime")
        emptyTime = clientInfo.get("emptyTime")
        disBW = clientInfo.get("disBW")
        disBW_percent = disBW / options.serverBW
        BW_utility = clientInfo.get("BW_utility")


        clientUnitInfo = clientInfo[str(options.JudgeDuration - 1)]
        capacityRate = clientUnitInfo['capacity'] / options.serverBW

        buffer = clientUnitInfo.get('buffer')

        buffer = clientUnitInfo.get("buffer")
        buffer_cur = buffer.data[-1][1]
        buffer_pre = buffer.amount - buffer.data[-1][1]
        buffer_percent = buffer.amount / buffer.size
        buffer_cur_percent = buffer_cur / buffer.size
        buffer_pre_percent = buffer_pre / buffer.size

        instantBW = clientUnitInfo.get("BW")
        instantBW_percent = instantBW / options.serverBW
        SNR = clientUnitInfo.get("SNR")

        env_state.append([resoIndex, fullTime, emptyTime, disBW_percent, BW_utility, capacityRate,
                          buffer_cur_percent, buffer_pre_percent, instantBW_percent, SNR])
    c1_state, c2_state, c3_state, c4_state = env_state[0], env_state[1], env_state[2], env_state[3]

    return c1_state, c2_state, c3_state, c4_state


def env_state5(clientExecResult):
    env_state = []

    SNRList = [[] for i in range(options.HostNum)]
    BWList = [[] for i in range(options.HostNum)]
    capacityList = [[] for i in range(options.HostNum)]

    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = clientExecResult[clientName]
        for t in range(options.JudgeDuration):
            BWList[index].append(clientInfo.get(str(t)).get("BW"))
            SNRList[index].append(clientInfo.get(str(t)).get("SNR"))
            capacityList[index].append(clientInfo.get(str(t)).get("capacity"))

    meanSNR = np.array(SNRList).mean(axis=1)
    meanBW = np.array(BWList).mean(axis=1)
    sumBW = np.array(BWList).sum(axis=0)
    meanSumBW = sumBW.mean()
    meanC = np.array(capacityList).mean(axis=1)

    a = 0

    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = clientExecResult[clientName]
        reso = clientInfo.get('reso')
        resoIndex = reso_mapping.index(reso)
        fullTime = clientInfo.get("fullTime") / options.JudgeDuration
        emptyTime = clientInfo.get("emptyTime") / options.JudgeDuration
        disBW = clientInfo.get("disBW")
        disBW_percent = disBW / options.serverBW
        BW_utility = clientInfo.get("BW_utility")

        clientUnitInfo = clientInfo[str(options.JudgeDuration - 1)]
        instantC = clientUnitInfo['capacity']
        capacityRate = instantC / options.serverBW

        buffer = clientUnitInfo.get('buffer')

        buffer = clientUnitInfo.get("buffer")
        buffer_cur = buffer.data[-1][1]
        buffer_pre = buffer.amount - buffer.data[-1][1]
        buffer_percent = buffer.amount / buffer.size
        buffer_cur_percent = buffer_cur / buffer.size
        buffer_pre_percent = buffer_pre / buffer.size

        # instantBW = clientUnitInfo.get("BW")
        # instantBW_percent = instantBW / options.serverBW
        # SNR = clientUnitInfo.get("SNR")

        instantBW = meanBW[index]
        instantBW_percent = instantBW / meanSumBW
        SNR = meanSNR[index]

        env_state.append([resoIndex, fullTime, emptyTime, disBW_percent, BW_utility, instantC,
                          buffer_cur_percent, buffer_pre_percent, instantBW_percent, SNR])
    c1_state, c2_state, c3_state, c4_state = env_state[0], env_state[1], env_state[2], env_state[3]

    return env_state, c1_state, c2_state, c3_state, c4_state


def env_state6(clientExecResult):
    env_state = []

    SNRList = [[] for i in range(options.HostNum)]
    capacityList = [[] for i in range(options.HostNum)]
    BFpreList = [[] for i in range(options.HostNum)]
    BFcurList = [[] for i in range(options.HostNum)]

    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = clientExecResult[clientName]

        for t in range(options.JudgeDuration):
            SNRList[index].append(clientInfo.get(str(t)).get("SNR"))
            capacityList[index].append(clientInfo.get(str(t)).get("capacity"))
            buffer = clientInfo.get(str(t)).get('buffer')
            buffer_cur = buffer.data[-1][1]
            buffer_pre = buffer.amount - buffer.data[-1][1]
            BFcurList[index].append(buffer_cur)
            BFpreList[index].append(buffer_pre)

    # The info of SNR
    SNR_array = np.array(SNRList)
    meanSNR = SNR_array.mean(axis=1)

    # The info of capacity
    capa_array = np.array(capacityList)
    meanC = capa_array.mean(axis=1)
    sumC = capa_array.sum(axis=0)
    capa_percent = capa_array / sumC
    meanC_percent = meanC / sumC.mean()

    # The info of buffer
    bf_pre_array = np.array(BFpreList)
    bf_cur_array = np.array(BFcurList)
    bf_pre_percent = bf_pre_array / options.bufferSize
    bf_cur_percent = bf_cur_array / options.bufferSize

    a = 0

    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = clientExecResult[clientName]
        reso = clientInfo.get('reso')
        resoIndex = reso_mapping.index(reso)
        fullTime = clientInfo.get("fullTime") / options.JudgeDuration
        emptyTime = clientInfo.get("emptyTime") / options.JudgeDuration
        disBW = clientInfo.get("disBW")
        disBW_percent = disBW / options.serverBW
        BW_utility = clientInfo.get("BW_utility")

        client_info = [meanSNR[index], meanC_percent[index], bf_pre_percent[index][-1], \
                      bf_cur_percent[index][-1], resoIndex, fullTime, emptyTime, disBW_percent, BW_utility]

        env_state.append(client_info)
    c1_state, c2_state, c3_state, c4_state = env_state[0], env_state[1], env_state[2], env_state[3]

    return env_state, c1_state, c2_state, c3_state, c4_state

def env_BW_reso(env_state, bwList):
    for index in range(options.HostNum):
        env_state[index] = [bwList[index]] + env_state[index]
    return env_state


def combineBWResoProbe(BWProbs, resoProbes):
    Probes = np.concatenate((resoProbes, BWProbs), axis=1)
    return Probes


def reward1(env_s):
    reward = 0
    reso = [1080, 720, 480, 360]
    # buffer constraint
    for index, i in enumerate(env_s):
        # if i[1] > 0.1 and i[2] == 0 and i[3] == 0:
        #     reward += 2
        if i[2] == 0 and i[3] == 0:
            reward += 5
        # elif i[4] == reso[index]:
        #     reward += 1
        elif i[2] != 0:
            reward -= 2 ** (i[2] / 2) * 20
        elif i[3] != 0:
            reward -= 2 ** (i[3] / 2) * 20
        else:
            reward += 0

    return reward

#
# def reward3(ExecResult):
#     totalreward = []
#
#     max_reward = 20
#     for index in range(options.HostNum):
#         clientName = 'c' + str(index + 1)
#         clientInfo = ExecResult[clientName]
#         clientUnitInfo = clientInfo[str(options.JudgeDuration - 1)]
#         buffer = clientUnitInfo.get("buffer")
#         buffer_curformat_ratio = buffer.get('data')[-1][1] / options.bufferSize
#         buffer_preformat_ratio = (buffer.get('amount') - buffer.get('data')[-1][1]) / options.bufferSize
#         emptyTime = clientInfo['emptyTime']
#         fullTime = clientInfo['fullTime']
#
#         reward = 0
#         # if emptyTime == 0 and fullTime == 0 and buffer_curformat_ratio > 0.3:
#         if emptyTime == 0 and fullTime == 0:
#             reward += 1
#         elif emptyTime != 0:
#             reward -= 2 ** (emptyTime / options.pauseTime)
#             # reward -= 2 ** (emptyTime / 1)
#             # reward -= emptyTime
#         elif fullTime != 0:
#             reward -= 2 ** (fullTime / options.pauseTime)
#             # reward -= 2 ** (fullTime / 1)
#             # reward -= fullTime
#         elif buffer_curformat_ratio <= 0.3:
#             reward -= 2 ** ((0.3 - buffer_curformat_ratio) / 0.1)
#         else:
#             reward += 0
#         reward /= max_reward
#         totalreward.append(reward)
#         print(clientName, ":", reward, end='\t')
#
#     print("current episode totalreward:", totalreward)
#     c1_reward, c2_reward, c3_reward, c4_reward = totalreward[0], totalreward[1], totalreward[2], totalreward[3]
#     return c1_reward, c2_reward, c3_reward, c4_reward



def reward3(ExecResult):
    totalreward = []

    max_reward = 20
    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = ExecResult[clientName]
        clientUnitInfo = clientInfo[str(options.JudgeDuration - 1)]
        buffer = clientUnitInfo.get("buffer")
        buffer_curformat_ratio = buffer.get('data')[-1][1] / options.bufferSize
        buffer_preformat_ratio = (buffer.get('amount') - buffer.get('data')[-1][1]) / options.bufferSize
        emptyTime = clientInfo['emptyTime']
        fullTime = clientInfo['fullTime']

        reward = 0
        if emptyTime == 0 and fullTime == 0 and buffer_curformat_ratio > 0.3:
        # if emptyTime == 0 and fullTime == 0:
            reward += 1
        elif emptyTime != 0:
            reward -= 2 ** (emptyTime / options.pauseTime)
            # reward -= 2 ** (emptyTime / 1)
            # reward -= emptyTime
        elif fullTime != 0:
            reward -= 2 ** (fullTime / options.pauseTime)
            # reward -= 2 ** (fullTime / 1)
            # reward -= fullTime
        elif buffer_curformat_ratio <= 0.2:
            reward -= 2 ** ((0.3 - buffer_curformat_ratio) / 0.1)
        else:
            reward += 0
        reward /= max_reward
        totalreward.append(reward)
        print(clientName, ":", reward, end='\t')

    print("current episode totalreward:", totalreward)
    c1_reward, c2_reward, c3_reward, c4_reward = totalreward[0], totalreward[1], totalreward[2], totalreward[3]
    return c1_reward, c2_reward, c3_reward, c4_reward


def reward4(ExecResult):
    totalreward = []

    max_reward = 20
    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = ExecResult[clientName]
        emptyTime = clientInfo['emptyTime']
        fullTime = clientInfo['fullTime']

        clientUnitInfo = clientInfo[str(options.JudgeDuration - 1)]
        buffer = clientUnitInfo.get("buffer")
        buffer_ratio = buffer.amount / buffer.size
        buffer_curformat_ratio = buffer.data[-1][1] / buffer.size
        buffer_preformat_ratio = (buffer.amount - buffer.data[-1][1]) / buffer.size

        reward = 0
        if emptyTime == 0 and fullTime == 0 and buffer_curformat_ratio > 0.3:
        # if emptyTime == 0 and fullTime == 0:
            reward += 10
        elif emptyTime != 0:
            reward -= 2 ** (emptyTime / options.pauseTime)
            # reward -= 2 ** (emptyTime / 1)
            # reward -= emptyTime
        elif fullTime != 0:
            reward -= 2 ** (fullTime / options.pauseTime)
            # reward -= 2 ** (fullTime / 1)
            # reward -= fullTime
        elif buffer_ratio <= 0.2:
            reward -= 2 ** ((0.3 - buffer_curformat_ratio) / 0.1) * 0.4
        else:
            reward += 0
        reward /= max_reward
        totalreward.append(reward)
        print(clientName, ":", reward, end='\t')

    print("current episode totalreward:{:.3f}".format(sum(totalreward)))
    c1_reward, c2_reward, c3_reward, c4_reward = totalreward[0], totalreward[1], totalreward[2], totalreward[3]
    return c1_reward, c2_reward, c3_reward, c4_reward


def reward2BW(ExecResult):
    totalreward = []

    max_reward = 50
    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = ExecResult[clientName]
        BW_utility = clientInfo.get("BW_utility")

        emptyTime = clientInfo.get('emptyTime')
        fullTime = clientInfo.get('fullTime')
        disBW = clientInfo.get("disBW")

        clientUnitInfo = clientInfo[str(options.JudgeDuration - 1)]
        buffer = clientUnitInfo.get("buffer")
        buffer_ratio = buffer.amount / buffer.size
        buffer_curformat_ratio = buffer.data[-1][1] / buffer.size
        buffer_preformat_ratio = (buffer.amount - buffer.data[-1][1]) / buffer.size

        BWList = []
        for t in range(options.JudgeDuration):
            BWList.append(clientInfo.get(str(t)).get("BW"))
        BW_mean = np.mean(BWList)
        maxBW = max(BWList)
        minBW = min(BWList)

        reward = 0
        # if emptyTime == 0 and fullTime == 0 and buffer_curformat_ratio > 0.3 and BW_utility >= 0.7 :
        # if emptyTime == 0 and fullTime == 0 and buffer_ratio > 0.3 and BW_utility >= 0.7:
        if emptyTime == 0 and fullTime == 0 and BW_utility >= 0.75 and minBW * 0.95 <= disBW <= maxBW * 1.05:
            reward += 5
        elif emptyTime != 0:
            reward -= 2 ** (emptyTime / options.pauseTime)*5
            # reward -= 2 ** (emptyTime / 1)
            # reward -= emptyTime
        elif fullTime != 0:
            reward -= 2 ** (fullTime / options.pauseTime)*5
            # reward -= 2 ** (fullTime / 1)
            # reward -= fullTime

        # elif buffer_ratio <= 0.2:
        #     reward -= 2 ** ((0.3 - buffer_ratio) / 0.1) * 0.4

        # elif buffer_curformat_ratio <= 0.2:
        #     reward -= 2 ** ((0.3 - buffer_curformat_ratio) / 0.1) * 0.4

        elif BW_utility <= 0.7:
            reward -= 2 ** ((1 - BW_utility) / 0.1)*10

        elif disBW >= maxBW * 1.2 or disBW <= minBW * 0.8:
            reward -= abs(disBW - BW_mean) * 20

        else:
            reward += 0
        reward /= max_reward
        totalreward.append(reward)
        print(clientName, ":{:.3f}".format(reward), end='\t')

    print("current episode totalreward:", totalreward)
    c1_reward, c2_reward, c3_reward, c4_reward = totalreward[0], totalreward[1], totalreward[2], totalreward[3]
    return c1_reward, c2_reward, c3_reward, c4_reward


def reward2BW1(ExecResult):
    totalreward = []

    max_reward = 1.0
    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = ExecResult[clientName]
        BW_utility = clientInfo.get("BW_utility")

        emptyTime = clientInfo.get('emptyTime')
        fullTime = clientInfo.get('fullTime')
        disBW = clientInfo.get("disBW")

        clientUnitInfo = clientInfo[str(options.JudgeDuration - 1)]
        buffer = clientUnitInfo.get("buffer")
        buffer_ratio = buffer.amount / buffer.size
        buffer_curformat_ratio = buffer.data[-1][1] / buffer.size
        buffer_preformat_ratio = (buffer.amount - buffer.data[-1][1]) / buffer.size


        SNRList = []
        BWList = []
        capacityList = []
        for t in range(options.JudgeDuration):
            BWList.append(clientInfo.get(str(t)).get("BW"))
            SNRList.append(clientInfo.get(str(t)).get("SNR"))
            capacityList.append(clientInfo.get(str(t)).get("capacity"))
        BW_mean = np.mean(BWList)
        maxBW = max(BWList)
        minBW = min(BWList)

        SNR_mean = np.mean(SNRList)
        maxSNR = max(SNRList)
        minSNR = min(SNRList)

        C_mean = np.mean(capacityList)
        maxC = max(capacityList)
        minC = min(capacityList)

        disC = disBW * SNR_mean

        reward = 0.0
        # if emptyTime == 0 and fullTime == 0 and buffer_curformat_ratio > 0.3 and BW_utility >= 0.7 :
        # if emptyTime == 0 and fullTime == 0 and buffer_ratio > 0.3 and BW_utility >= 0.7:
        # if emptyTime == 0 and fullTime == 0 and buffer_ratio > 0.3:
        if emptyTime == 0 and fullTime == 0 and BW_utility >= 0.85 and minC * 1.2 <= disC <= maxC * 0.8:
            reward += 2
        elif emptyTime != 0:
            reward -= 2 ** (emptyTime / options.pauseTime) * 2
            # reward -= 2 ** (emptyTime / 1)
            # reward -= emptyTime
        elif fullTime != 0:
            reward -= 2 ** (fullTime / options.pauseTime) * 2
            # reward -= 2 ** (fullTime / 1)
            # reward -= fullTime

        # elif buffer_ratio <= 0.2:
        #     reward -= 2 ** ((0.3 - buffer_ratio) / 0.1) * 0.4

        # elif buffer_curformat_ratio <= 0.2:
        #     reward -= 2 ** ((0.3 - buffer_curformat_ratio) / 0.1) * 0.4

        elif BW_utility <= 0.7:
            reward -= 2 ** ((1 - BW_utility) / 0.1) * 0.1

        # elif disBW >= maxBW * 1.2 or disBW <= minBW * 0.8:
        #     reward -= (1 - disBW / BW_mean) * 3

        elif disC >= maxC * 0.9 or disC <= minC * 1.1:
            reward -= 2 ** (abs(1 - disC / C_mean) / 0.1) * 0.2

        else:
            reward += 0
        reward /= max_reward
        totalreward.append(reward)
        print(clientName, ":{:.3f}".format(reward), end='\t')

    print("current episode totalreward:{:.3f}".format(sum(totalreward)))
    c1_reward, c2_reward, c3_reward, c4_reward = totalreward[0], totalreward[1], totalreward[2], totalreward[3]
    return c1_reward, c2_reward, c3_reward, c4_reward


def reward2BW2(ExecResult):
    totalreward = []

    max_reward = 10.0
    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = ExecResult[clientName]
        BW_utility = clientInfo.get("BW_utility")

        emptyTime = clientInfo.get('emptyTime')
        fullTime = clientInfo.get('fullTime')
        disBW = clientInfo.get("disBW")
        reso = clientInfo.get("reso")
        resoBW = lib.frameDataSize.get(reso)

        clientUnitInfo = clientInfo[str(options.JudgeDuration - 1)]
        buffer = clientUnitInfo.get("buffer")
        buffer_ratio = buffer.amount / buffer.size
        buffer_curformat_ratio = buffer.data[-1][1] / buffer.size
        buffer_preformat_ratio = (buffer.amount - buffer.data[-1][1]) / buffer.size

        SNRList = []
        BWList = []
        capacityList = []
        for t in range(options.JudgeDuration):
            BWList.append(clientInfo.get(str(t)).get("BW"))
            SNRList.append(clientInfo.get(str(t)).get("SNR"))
            capacityList.append(clientInfo.get(str(t)).get("capacity"))
        BW = np.array(BWList)
        SNR = np.array(SNRList)
        a = np.log2(1+SNR)
        b = disBW * np.log2(1 + SNR)
        disC = disBW * np.log2(1 + SNR)
        C = BW * np.log2(1 + SNR)

        B_var = (BW - disBW).var()
        C_var = (C - disC).var()

        reward = 0.0
        # if emptyTime == 0 and fullTime == 0 and buffer_curformat_ratio > 0.3 and BW_utility >= 0.7 :
        # if emptyTime == 0 and fullTime == 0 and buffer_ratio > 0.3 and BW_utility >= 0.7:
        # if emptyTime == 0 and fullTime == 0 and BW_utility >= 0.85 and buffer_ratio > 0.3:
        # if emptyTime == 0 and fullTime == 0 and BW_utility >= 0.85 and C_mean * 0.9 <= disC_mean <= C_mean * 1.1:
        if emptyTime == 0 and fullTime == 0 and BW_utility >= 0.85 and B_var <= 0.3 and C_var <= 0.3:
            reward += 10.0

        if emptyTime != 0:
            # reward -= 2 ** (emptyTime / options.pauseTime) * 1 * (1 - resoBW / 7)
            reward -= 2 ** (emptyTime / options.pauseTime) * 1
            # reward -= 2 ** (emptyTime / 1)
            # reward -= emptyTime
        if fullTime != 0:
            # reward -= 2 ** (fullTime / options.pauseTime) * 1 * resoBW / 7
            reward -= 2 ** (fullTime / options.pauseTime) * 1
            # reward -= 2 ** (fullTime / 1)
            # reward -= fullTime

        # elif buffer_ratio <= 0.2:
        #     reward -= 2 ** ((0.3 - buffer_ratio) / 0.1) * 0.4
        #
        # elif buffer_curformat_ratio <= 0.2:
        #     reward -= 2 ** ((0.3 - buffer_curformat_ratio) / 0.1) * 0.4

        if BW_utility <= 0.7:
            reward -= 2 ** ((1 - BW_utility) / 0.1) * 0.5

        if B_var >= 0.3:
            reward -= 2 ** (B_var/ 0.2)

        if C_var >= 0.3:
            reward -= 2 ** (C_var/ 0.2)

        reward /= max_reward
        totalreward.append(reward)
        print(clientName, ":{:.3f}".format(reward), end='\t')

    print("current episode totalreward:{:.3f}".format(sum(totalreward)))
    c1_reward, c2_reward, c3_reward, c4_reward = totalreward[0], totalreward[1], totalreward[2], totalreward[3]
    return c1_reward, c2_reward, c3_reward, c4_reward


def reward2BW3(ExecResult):

    all_fulltime = 0.0
    all_emptytime = 0.0
    all_BW_var = 0.0
    all_C_var = 0.0
    all_bw_util = []

    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = ExecResult[clientName]
        BW_utility = clientInfo.get("BW_utility")

        emptyTime = clientInfo.get('emptyTime')
        fullTime = clientInfo.get('fullTime')
        disBW = clientInfo.get("disBW")
        reso = clientInfo.get("reso")
        resoBW = lib.frameDataSize.get(reso)

        clientUnitInfo = clientInfo[str(options.JudgeDuration - 1)]
        buffer = clientUnitInfo.get("buffer")
        buffer_ratio = buffer.amount / buffer.size
        buffer_curformat_ratio = buffer.data[-1][1] / buffer.size
        buffer_preformat_ratio = (buffer.amount - buffer.data[-1][1]) / buffer.size

        SNRList = []
        BWList = []
        capacityList = []
        for t in range(options.JudgeDuration):
            BWList.append(clientInfo.get(str(t)).get("BW"))
            SNRList.append(clientInfo.get(str(t)).get("SNR"))
            capacityList.append(clientInfo.get(str(t)).get("capacity"))
        BW = np.array(BWList)
        SNR = np.array(SNRList)

        disC = disBW * np.log2(1 + SNR)

        C = BW * np.log2(1 + SNR)

        B_var = (BW - disBW).var()
        C_var = (C - disC).var()

        # all_fulltime += fullTime * resoBW / 7
        # all_emptytime += emptyTime * (1 - resoBW / 7)
        # all_BW_var += B_var * (1 - resoBW / 7)
        # all_C_var += C_var * (1 - resoBW / 7)
        # all_bw_util.append(BW_utility * (1 - resoBW / 7))

        all_fulltime += fullTime
        all_emptytime += emptyTime
        all_BW_var += B_var
        all_C_var += C_var
        all_bw_util.append(BW_utility)

    mean_util = sum(all_bw_util) / len(all_bw_util)

    max_reward = 10.0
    reward = 0.0
    if all_fulltime == 0 and all_emptytime == 0 and mean_util >= 0.7 and all_BW_var <= 1 and all_C_var <= 1:
        reward += 10.0

    if all_fulltime != 0:
        reward -= 2 ** (all_emptytime / options.pauseTime) * 1

    if all_emptytime != 0:
        reward -= 2 ** (all_fulltime / options.pauseTime) * 1

    if mean_util <= 0.7:
        reward -= 2 ** ((1 - mean_util) / 0.1) * 1

    if all_BW_var >= 1.1:
        # reward -= 2 ** (all_BW_var/ 0.1)
        reward -= all_BW_var * 100

    if all_C_var >= 1.1:
        # reward -= 2 ** (all_C_var/ 0.1)
        reward -= all_C_var * 150

    reward /= max_reward
    print("totalreward", ":{:.3f}".format(reward), end='\t')

    return reward


def reward2BW4(ExecResult):
    max_reward = 10.0
    reward = 0.0

    if len(buffer_reso_index[0]) >= 6:
        print("the length of buffer_reso is :6")
        for i in range(options.HostNum):
            del buffer_reso_index[i][0]

    all_fulltime = 0.0
    all_emptytime = 0.0
    all_C_var = 0.0
    all_bw_util = []
    test_bw_util = []
    qoe_list = []

    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = ExecResult[clientName]
        BW_utility = clientInfo.get("BW_utility")

        emptyTime = clientInfo.get('emptyTime')
        fullTime = clientInfo.get('fullTime')
        disBW = clientInfo.get("disBW")
        reso = clientInfo.get("reso")
        reso_index = reso_mapping.index(reso)
        buffer_reso_index[index].append(reso_index)

        clientUnitInfo = clientInfo[str(options.JudgeDuration - 1)]
        buffer = clientUnitInfo.get("buffer")
        buffer_ratio = buffer.amount / buffer.size
        buffer_curformat_ratio = buffer.data[-1][1] / buffer.size
        buffer_preformat_ratio = (buffer.amount - buffer.data[-1][1]) / buffer.size

        SNRList = []
        capacityList = []
        for t in range(options.JudgeDuration):
            SNRList.append(clientInfo.get(str(t)).get("SNR"))
            capacityList.append(clientInfo.get(str(t)).get("capacity"))
        SNR = np.array(SNRList)

        resoC = lib.frameDataSize.get(reso)
        realC = disBW * np.log2(1 + SNR)

        BW_utility = (realC - resoC) ** 2

        qoe = resoC - fullTime - emptyTime - np.array(buffer_reso_index[index]).var() + BW_utility.mean()
        qoe_list.append(qoe)

        # all_fulltime += fullTime * resoBW / 7
        # all_emptytime += emptyTime * (1 - resoBW / 7)
        # all_BW_var += B_var * (1 - resoBW / 7)
        # all_C_var += C_var * (1 - resoBW / 7)
        # all_bw_util.append(BW_utility * (1 - resoBW / 7))

        all_fulltime += fullTime
        all_emptytime += emptyTime
        all_bw_util.append(BW_utility * resoC)
        test_bw_util.append(BW_utility)

    mean_util = sum(all_bw_util) / options.serverBW
    reso_var = np.array(buffer_reso_index).var()
    total_qoe = sum(qoe_list)

    # if all_fulltime == 0 and all_emptytime == 0 and reso_var <= 2 and all_C_var <= 4:
    if all_fulltime == 0 and all_emptytime == 0 and reso_var <= 2 and total_qoe <= 10:
    # if all_fulltime == 0 and all_emptytime == 0 and mean_util >= 0.6:
        reward += 10.0

    if all_fulltime != 0:
        reward -= 2 ** (all_emptytime / options.pauseTime) * 1

    if all_emptytime != 0:
        reward -= 2 ** (all_fulltime / options.pauseTime) * 1

    # reward -= reso_var * 50
    # reward += total_qoe * 5


    aa = [test_bw_util[i].mean() for i in range(4)]
    reward -= sum(2**aa)


    # if mean_util <= 0.7:
    #     reward -= 2 ** ((1 - mean_util) / 0.05) * 1

    if all_C_var >= 10:
        # reward -= 2 ** (all_C_var/ 0.1)
        reward -= all_C_var * 10

    # reward = total_qoe

    reward /= max_reward
    print("totalreward", ":{:.3f}".format(reward))

    return reward


def reward2BW5(ExecResult):
    max_reward = 10.0
    reward = 0.0

    if len(buffer_reso[0]) >= 5:
        print("the length of buffer_reso is :6")
        for i in range(options.HostNum):
            del buffer_reso[i][0]

    all_fulltime = 0.0
    all_emptytime = 0.0
    all_C_var = 0.0
    all_bw_util = []
    test_bw_util = []
    qoe_list = []

    for index in range(options.HostNum):
        clientName = 'c' + str(index + 1)
        clientInfo = ExecResult[clientName]
        BW_utility = clientInfo.get("BW_utility")

        emptyTime = clientInfo.get('emptyTime') / options.JudgeDuration
        fullTime = clientInfo.get('fullTime') / options.JudgeDuration
        disBW = clientInfo.get("disBW")
        reso = clientInfo.get("reso")
        reso_index = reso_mapping.index(reso)
        buffer_reso[index].append(reso_index)

        clientUnitInfo = clientInfo[str(options.JudgeDuration - 1)]
        buffer = clientUnitInfo.get("buffer")
        buffer_ratio = buffer.amount / buffer.size
        buffer_curformat_ratio = buffer.data[-1][1] / buffer.size
        buffer_preformat_ratio = (buffer.amount - buffer.data[-1][1]) / buffer.size

        SNRList = []
        capacityList = []
        for t in range(options.JudgeDuration):
            SNRList.append(clientInfo.get(str(t)).get("SNR"))
            capacityList.append(clientInfo.get(str(t)).get("capacity"))
        SNR = np.array(SNRList)

        resoC = lib.frameDataSize.get(reso)
        realC = disBW * np.log2(1 + SNR)

        reward += resoC
        reso_var = np.array(buffer_reso[index]).var()
        reward -= 10 ** reso_var

        BW_utility = (realC - resoC) ** 2
        reward -= 10 * BW_utility.var()

        qoe = resoC - fullTime - emptyTime - np.array(buffer_reso[index]).var() + BW_utility.mean()
        qoe_list.append(qoe)

        # all_fulltime += fullTime * resoBW / 7
        # all_emptytime += emptyTime * (1 - resoBW / 7)
        # all_BW_var += B_var * (1 - resoBW / 7)
        # all_C_var += C_var * (1 - resoBW / 7)
        # all_bw_util.append(BW_utility * (1 - resoBW / 7))

        all_fulltime += fullTime
        all_emptytime += emptyTime
        all_bw_util.append(BW_utility * resoC)
        test_bw_util.append(BW_utility)

    mean_util = sum(all_bw_util) / options.serverBW

    total_qoe = sum(qoe_list)

    # # if all_fulltime == 0 and all_emptytime == 0 and reso_var <= 2 and all_C_var <= 4:
    # if all_fulltime == 0 and all_emptytime == 0 and reso_var <= 2 and total_qoe <= 10:
    # # if all_fulltime == 0 and all_emptytime == 0 and mean_util >= 0.6:
    #     reward += 10.0
    #
    # if all_fulltime != 0:
    #     reward -= 2 ** (all_emptytime / options.pauseTime) * 1
    #
    # if all_emptytime != 0:
    #     reward -= 2 ** (all_fulltime / options.pauseTime) * 1
    #
    # # reward -= reso_var * 50
    # # reward += total_qoe * 5
    #
    #
    # aa = [test_bw_util[i].mean() for i in range(4)]
    # reward -= sum(2 ** aa)
    #
    # # if mean_util <= 0.7:
    # #     reward -= 2 ** ((1 - mean_util) / 0.05) * 1
    #
    # if all_C_var >= 10:
    #     # reward -= 2 ** (all_C_var/ 0.1)
    #     reward -= all_C_var * 10
    #
    # # reward = total_qoe
    #
    # reward /= max_reward
    # print("totalreward", ":{:.3f}".format(reward))

    reward -= 2 ** (all_fulltime / options.pauseTime)
    reward -= 2 ** (all_emptytime / options.pauseTime)
    # reward -= 5 ** reso_var

    reward /= max_reward
    return reward

def reward_window(windowInfo):
    totalreward = 0
    windowInfoLength = len(windowInfo)
    pre_action = [0 for i in range(options.HostNum)]
    max_reward = 20

    for index in range(options.HostNum):
        isSteady = True
        count = 0
        for i, item in enumerate(windowInfo):
            clientName = 'c' + str(index + 1)
            clientInfo = item[clientName]
            clientUnitInfo = clientInfo[str(options.JudgeDuration - 1)]
            buffer = clientUnitInfo.get("buffer")
            buffer_curformat_ratio = buffer.get('data')[-1][1] / options.bufferSize
            buffer_preformat_ratio = (buffer.get('amount') - buffer.get('data')[-1][1]) / options.bufferSize
            emptyTime = clientInfo['emptyTime']
            fullTime = clientInfo['fullTime']
            reso = clientInfo['reso']
            reso_BW = lib.frameDataSize.get(reso)
            if reso_BW > pre_action[index]:
                pre_action[index] = reso_BW
            else:
                pre_action[index] = reso_BW
                isSteady = False
                count += 1

            if i == windowInfoLength - 1:
                reward = 0
                if emptyTime == 0 and fullTime == 0 and count <= 1 :
                # if emptyTime == 0 and fullTime == 0 and buffer_curformat_ratio > 0.3 and isSteady:
                    reward += 5
                elif emptyTime != 0:
                    reward -= 2 ** (emptyTime / options.pauseTime)
                    # reward -= 2 ** (emptyTime / 1)
                elif fullTime != 0:
                    reward -= 2 ** (fullTime / options.pauseTime)
                    # reward -= 2 ** (fullTime / 1)
                elif buffer_curformat_ratio <= 0.2:
                    reward -= 2 ** ((0.2 - buffer_curformat_ratio) / 0.1)
                # elif not isSteady:
                #     reward -= 200
                elif count >= 2:
                    reward -= 2 ** (count)
                else:
                    reward += 0
                reward /= max_reward
                totalreward += reward
                print(clientName, ":{:8.2f}".format(reward), end='\t')

    print("current episode totalreward:{:10.2f}".format(totalreward))

    return totalreward


def reward_window1(windowInfo):
    totalreward = []
    windowInfoLength = len(windowInfo)
    pre_action = [0 for i in range(options.HostNum)]
    max_reward = 20

    for index in range(options.HostNum):
        isSteady = True
        for i, item in enumerate(windowInfo):
            clientName = 'c' + str(index + 1)
            clientInfo = item[clientName]
            clientUnitInfo = clientInfo[str(options.JudgeDuration - 1)]
            buffer = clientUnitInfo.get("buffer")
            buffer_curformat_ratio = buffer.get('data')[-1][1] / options.bufferSize
            buffer_preformat_ratio = (buffer.get('amount') - buffer.get('data')[-1][1]) / options.bufferSize
            emptyTime = clientInfo['emptyTime']
            fullTime = clientInfo['fullTime']
            reso = clientInfo['reso']
            reso_BW = lib.frameDataSize.get(reso)
            if reso_BW > pre_action[index]:
                pre_action[index] = reso_BW
            else:
                pre_action[index] = reso_BW
                isSteady = False

            if i == windowInfoLength - 1:
                reward = 0
                if emptyTime == 0 and fullTime == 0 and buffer_curformat_ratio > 0.3 and isSteady:
                    reward += 1
                elif emptyTime != 0:
                    reward -= 2 ** (emptyTime / options.pauseTime)
                    # reward -= 2 ** (emptyTime / 1)
                elif fullTime != 0:
                    reward -= 2 ** (fullTime / options.pauseTime)
                    # reward -= 2 ** (fullTime / 1)
                elif buffer_curformat_ratio <= 0.2:
                    reward -= 2 ** ((0.2 - buffer_curformat_ratio) / 0.1)
                elif not isSteady:
                    reward -= 200
                else:
                    reward += 0
                reward /= max_reward
                totalreward.append(reward)
                print(clientName, ":{:8.2f}".format(reward), end='\t')

    print("current episode totalreward:{:10.2f}".format(sum(totalreward)))
    c1_reward, c1_reward, c1_reward, c1_reward = totalreward[0], totalreward[1], totalreward[2], totalreward[3]
    return c1_reward, c1_reward, c1_reward, c1_reward


def get_bw_reso(s_bw, reso):
    bw_reso = copy.deepcopy(s_bw)
    for index in range(len(s_bw)):
        bw_reso[index].append(reso[index])

    return bw_reso


if __name__ == '__main__':
    pre_action = [1.67 for i in range(options.HostNum)]
    print(pre_action)

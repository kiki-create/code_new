import numpy as np


def adjust_CC(disCC, snr):
    """
    根据带宽调整CC，模拟路由器带宽分配规则
    :param disCC:
    :return:
    """
    disBW = {}  # 按照公式算的每个人的bw
    BW_rank = {}
    for i in range(4):
        disBW["c"+str(i+1)] = (disCC[i] / np.log2(1+snr["c"+str(i+1)]))
        BW_rank["c"+str(i+1)] = 1
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
                    BW_rank["c"+str(i+1)] += redundant_bw/4
                break
            else:
                BW_rank[sorted_Snr[i][0]] = totalBW + 1
                break
        if totalBW > 0:
            if bw >= 1 and totalBW >= bw and bw <= 20/2:  # 不用对“分配”的BW处理的情况
                totalBW = totalBW - bw + 1  # +1 是把垫的1M补回来
                BW_rank[sorted_Snr[i][0]] = bw
                continue
            else:
                if bw > 20/2 or bw > totalBW:
                    bw = min(totalBW + 1, 20/2)
                    totalBW = totalBW - bw + 1  # +1 是把垫的1M补回来
                    BW_rank[sorted_Snr[i][0]] = bw
                    continue
        else:
            break
    return BW_rank


if __name__ == '__main__':
    CR_v_target = [37.57125935054198, 37.71261051252253, 35.21042177810031, 25.078367973716976]
    CR_v_ = [[1., 1., 1., 1.]]
    td_error = 0
    for i in range(len(CR_v_target)):
        td_error += np.square(CR_v_target[i] - CR_v_[0][i])
    print(td_error)

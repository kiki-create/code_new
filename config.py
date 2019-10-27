#!/usr/bin/python3
# coding=utf-8

from optparse import OptionParser


frameDataSize = {
    # 1.25 色度的抽样率是4:1
    '1080P' : 19.77,   # (1920, 1080) -> 1920 * 1080 *  8 * 1. 25 / 1024 / 1024 = 19.77M
    '720P'  : 8.79,   # (1280, 720)  1280 * 720 * 1.5 * 8 / 1024/1024 = 8.79M
    '480P'  : 2.92,     # (640, 480) 640 * 480 * 24 / 1024 /1024 = 2.92M
    '360P'  : 1.67,  # (480, 360) 480 * 360 * 24 /1024 /1024 = 1.67M
}


usage = "%prog Parameter configuration"
description = "description are as folllows!"
program = "simnet 1.0"


parser = OptionParser(usage=usage, description=description, prog=program)

parser.add_option("--duration", action="store", type="int", dest="JudgeDuration",
                default=10, help="The duration time of a slot")

parser.add_option("--pauseTime", action="store", type="int", dest="pauseTime",
                default=1, help="The time of pause when buffer is full or empty")


parser.add_option("-n", "--hostnum", action="store", type="int", dest="HostNum",
                default=4, help="The number of cients under central-server")

parser.add_option("--totalCC", action="store", type="int", dest="serverCC",
                default=8, help="The CC of central-server referred to 802.11g")

parser.add_option("--totalBW", action="store", type="int", dest="serverBW",
                default=15, help="The CC of central-server referred to 802.11g")

parser.add_option("--bufferSize", action="store", type="int", dest="bufferSize",
                default=50 * 6,
                help="The BW of central-server referred to 802.11g")

parser.add_option("--maxSNR", action="store", type="int", dest="maxSNR", default=2.5,
                help="The maximum of SNR towards every channel")

parser.add_option("--minSNR", action="store", type="int", dest="minSNR", default=1.0,
                help="The maximum of SNR towards every channel")

parser.add_option("--saveDir", action="store", type="string", dest="modelSaveDir",
                default="./model/",
                help="The path of the model saving!")

options, args = parser.parse_args()

if __name__ == "__main__":
    print(options.JudgeDuration)
    print(options.modelSaveDir)
    pass


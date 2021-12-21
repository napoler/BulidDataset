import json
import os


def readCMeEE(datapath):
    """
    读取CMeEE数据，
    返回fileName, datajson

    :param datapath:
    :return fileName, datajson:
    """
    fileList = ["CMeEE_dev.json", "CMeEE_train.json"]

    for fileName in fileList:
        file = os.path.join(datapath, fileName)
        with open(file, "r") as f:
            datajson = json.load(f)
            yield fileName, datajson
        pass



def readDir(datapath):
    # global datajson
    for dirname, _, filenames in os.walk(datapath):
        for filename in filenames:
            file = os.path.join(dirname, filename)
            # if file.endswith(".json"):
            if ".json" in file:
                # print(file)
                pass
    datajson = []
    for dirname, _, filenames in os.walk(datapath):
        for filename in filenames:
            #         print(os.path.join(dirname, filename))
            file = os.path.join(dirname, filename)
            # if file.endswith(".json"):
            if ".json" in file:
                with open(file, "r") as f:
                    # print(file)

                    datajson = datajson + json.load(f)
    print("数据条数", len(datajson))
    return datajson

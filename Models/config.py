configuration = {
    "image_width": 227,
    "image_height": 227,
    "grid_size": 4,
    "n_boxes": 3,
    "n_channels": 3,
    "n_classes": 0,
    "model_name": "squeeze_tiny"
}


class GlobalVar:
    numConv = 0
    saveConvInputOutput = False
    scoreThreshold=0.0


def get_numConv():
    return GlobalVar.numConv


def set_numConv(num):
    GlobalVar.numConv = num

def get_saveConvInputOutput():
    return GlobalVar.saveConvInputOutput

def get_score_threshold():
    return GlobalVar.scoreThreshold

def set_scoreThreshold(num):
    GlobalVar.scoreThreshold=num
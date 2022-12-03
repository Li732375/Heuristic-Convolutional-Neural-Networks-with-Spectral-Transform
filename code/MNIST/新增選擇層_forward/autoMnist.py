from hillClimbing import hillClimbing
from net import Net
import os
from solutionNet import SolutionNet


def start():
	net = Net.base_model([28,28], [10])
	return SolutionNet(net)

def checkmodeldir(point):
	if os.path.exists("model"):
		os.rename("model", "model-" + str(point))
		os.mkdir("model")
	else:
		os.mkdir("model")

for count in range(31, 101):#一次產生數個模型，分別存放在相應數字(如"model-1")的檔名
	checkmodeldir(count + 1)
	hillClimbing(start(), max_gens=1000, max_fails=50)

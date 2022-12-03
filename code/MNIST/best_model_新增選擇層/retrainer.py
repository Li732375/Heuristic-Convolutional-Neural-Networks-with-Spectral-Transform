import torch
from net import Net
import trainer
from glob import glob
import json
import os
import numpy as np

def check_pt(): # 未使用到
	model_path = glob(r'model_[0-9]*/*.pt')[0]
	print("model_path: ", model_path)
	net = torch.load(model_path) #匯入最佳模型
	print("model:\n", net) #確認模型
	print("model len: ", len(net))
	print("model type: ", type(net))
	for key, value in net.items():
		print()
		print("key : ", key)
		print("key type: ", type(key))
		print("value type: ", type(value))
		print("value shape: ", value.shape)
	
def import_json():
	model_json_path = glob(r'model-[0-9]*/*.json')[0]
	print("model_json_path: ", model_json_path)
	jsonFile = open(model_json_path, "rt")
	jsonStr = jsonFile.read()
	jsonObj = json.loads(jsonStr)
	jsonFile.close()

	print("\nmodel_json: ", jsonObj)
	print("\nmodel_json_keys(frist level): ")
	for key in jsonObj.keys():print(key)
	print("\nmodel_json(frist level): ")
	for key, value in jsonObj.items():
		print()
		print(key, value)
		
	return jsonObj

def product_net(jsonObj):
	net = Net()
	net.build(jsonObj)
	return net

def main():
	
	if not os.path.exists("best_mode"):
		os.mkdir("best_mode")
	
	model_dict = import_json() # 匯入 model 的 json(模型架構)
	net = product_net(model_dict) # 塑造出網路 net
	
	trainer.run(net) # 訓練並測試網路
	
	torch.save(net, f"best_mode/best_mode.pt") # 完整保存整個網路

	
	train_losses_3 = trainer.load_info('train_losses_3') # 匯入數據資料
	print('train_losses_3: ', train_losses_3)

	# 繪圖
	trainer.product_table(title='train_losses',
                              line_1=train_losses_3,
                              line_1_lable='train_losses_3')

	# 視情形需要是否有利用	
	test_losses = trainer.load_info('test_losses')
	print('test_losses: ', test_losses)
	
	accuracy = trainer.load_info('accuracy')
	print('accuracy: ', accuracy)
	
if __name__ == "__main__":
	main()

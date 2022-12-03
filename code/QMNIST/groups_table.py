import os
import numpy as np
import matplotlib.pyplot as plt

def load_info(path):
	nparray = np.load(path)
	print('\nFinish load ', path)
	
	try:
		print('nparray len: ', len(nparray))
	except TypeError:
		print(path, ': ', nparray)
	return nparray

def product_table(title, line_1, line_1_lable, line_2=[], line_2_lable=None
                  , line_3=[], line_3_lable=None, line_4=[], line_4_lable=None,
                  delta=50, bottom=-3, length=-1):
	if type(line_1) != 'numpy':
		line_1 = np.array(line_1)[:length]
	
	if type(line_2) != 'numpy' and len(line_2) != 0:
		line_2 = np.array(line_2)[:length]

	if type(line_3) != 'numpy' and len(line_3) != 0:
		line_3 = np.array(line_3)[:length]

	if type(line_4) != 'numpy' and len(line_4) != 0:
		line_4 = np.array(line_4)[:length]
		
	plt.style.use('classic')
	plt.subplots()
	plt.figure(figsize=(6, 4))
	
	max_len = max([len(line_1), len(line_2), len(line_3), len(line_4)])
	max_value = max(line_1)
	plt.plot(line_1, label=line_1_lable, color='b', lineWidth=0.8)
	
	if len(line_2) != 0:
		max_value = max([max(line_1), max(line_2)])
		plt.plot(np.arange(0, len(line_2)), line_2, label=line_2_lable, color='r', lineWidth=1)
	if len(line_3) != 0:
		max_value = max([max(line_1), max(line_2), max(line_3)])
		plt.plot(np.arange(0, len(line_3)), line_3, label=line_3_lable, color='g', lineWidth=1)
	if len(line_4) != 0:
		max_value = max([max(line_1), max(line_2), max(line_3), max(line_4)])
		plt.plot(np.arange(0, len(line_4)), line_4, label=line_4_lable, color='k', lineWidth=1)
		
	plt.legend(loc='upper right')
	plt.axis([-10, len(line_1)+18,
                  bottom,
                  max_value + max_value* 0.1])
	plt.yticks(fontsize=10)
	plt.xticks(np.arange(0, len(line_1) + delta, delta), fontsize=10) 
	plt.savefig(title + '.png')
	plt.show()
	print('Finish save table', title , '.png', '！')
	plt.close()

def main():
	
	newlayer_tl = load_info('best_model_新增選擇層/best_mode/train_losses_1.npy') 
	newlayer_backward_tl = load_info('best_model_新增選擇層_backward/best_mode/train_losses_1.npy') 
	newlayer_forward_tl = load_info('best_model_新增選擇層_forward/best_mode/train_losses_1.npy') 
	newlayer_control_tl = load_info('best_model_對照組/best_mode/train_losses_1.npy') 
	
	# 繪圖
	product_table(title='groups train_losses',
                      line_1=newlayer_tl,
                      line_1_lable='newlayer_random',
                      line_2=newlayer_backward_tl,
                      line_2_lable='newlayer_backward',
                      line_3=newlayer_forward_tl,
                      line_3_lable='newlayer_forward',
                      line_4=newlayer_control_tl,
                      line_4_lable='newlayer_control',
                      delta=20,
                      bottom=0,
                      length=20)

	# 繪圖
	product_table(title='groups train_losses_1',
                      line_1=newlayer_backward_tl,
                      line_1_lable='newlayer_backward',
                      delta=20,
                      bottom=0,
                      length=20)
	
	# 繪圖
	product_table(title='groups train_losses_2',
                      line_1=newlayer_tl,
                      line_1_lable='newlayer_random',
                      line_2=newlayer_forward_tl,
                      line_2_lable='newlayer_forward',
                      line_3=newlayer_control_tl,
                      line_3_lable='newlayer_control',
                      delta=20,
                      bottom=0,
                      length=20)
	
	
if __name__ == "__main__":
	main()

import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from net import Net
import matplotlib.pyplot as plt
import numpy as np

def log(msg):
	print(msg)
	# pass

n_epochs = 3
epoch_seconds_limit = 100000
# epoch_seconds_limit = 2
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
	torchvision.datasets.FashionMNIST("../files/", train=True, download=True,
							 transform=torchvision.transforms.Compose([
								 torchvision.transforms.ToTensor(),
								 torchvision.transforms.Normalize(
								 (0.2860,), (0.3530,))
							 ])),
	batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
	torchvision.datasets.FashionMNIST("../files/", train=False, download=True,
							 transform=torchvision.transforms.Compose([
								 torchvision.transforms.ToTensor(),
								 torchvision.transforms.Normalize(
								 (0.2860,), (0.3530,))
							 ])),
	batch_size=batch_size_test, shuffle=True)

def save_info(info_list, name):
	np.save('best_mode/' + name, info_list)
	print('\nFinish save best_mode/', name, '.npy')

def load_info(name):
	nparray = np.load('best_mode/' + name + '.npy')
	print('\nFinish load best_mode/', name, '.npy')
	try:
		print(name, ' len: ', len(nparray))
	except TypeError:
		print(name, ': ', nparray)
	return nparray

def product_table(title, line_1, line_1_lable, line_2=[], line_2_lable=None
                  , line_3=[], line_3_lable=None, line_4=[], line_4_lable=None, delta=50 ):
	if type(line_1) != 'numpy':
		line_1 = np.array(line_1)
	
	if type(line_2) != 'numpy' and len(line_2) != 0:
		line_2 = np.array(line_2)

	if type(line_3) != 'numpy' and len(line_3) != 0:
		line_3 = np.array(line_3)

	if type(line_4) != 'numpy' and len(line_4) != 0:
		line_4 = np.array(line_4)
		
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
                  -3,
                  max_value + max_value* 0.1])
	plt.yticks(fontsize=10)
	plt.xticks(np.arange(0, len(line_1) + delta, delta), fontsize=10) 
	plt.savefig(title + '.png')
	print('Finish save table', title , '.png', '！')

def init(net):
	global train_losses, train_counter, test_losses, test_counter, network, optimizer
	train_losses = []
	train_counter = []
	test_losses = []
	test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
	network = net
	optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

def train(epoch):
	tstart = datetime.now()
	network.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		tnow = datetime.now()
		dt = tnow - tstart
		if dt.seconds > epoch_seconds_limit: break
		optimizer.zero_grad()
		output = network(data)
		loss = F.nll_loss(F.log_softmax(output, dim=1), target)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			log("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
			epoch, batch_idx * len(data), len(train_loader.dataset),
			100. * batch_idx / len(train_loader), loss.item()))
			train_losses.append(loss.item())
			train_counter.append(
			(batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
			
	save_info(train_losses, 'train_losses_' + str(epoch))
	#save_info(train_counter, 'train_counter_' + str(epoch))

def test():
	network.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			output = network(data)
			test_loss += F.nll_loss(F.log_softmax(output, dim=1), target, size_average=False).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
		test_loss /= len(test_loader.dataset)
		test_losses.append(test_loss)
		accuracy = 100. * correct / len(test_loader.dataset)
		log("\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
		test_loss, correct, len(test_loader.dataset), accuracy))
		network.model["accuracy"] = accuracy.item()

	save_info(test_losses, 'test_losses')
	save_info(accuracy, 'accuracy')

def run(net):
	init(net)
	for epoch in range(1, n_epochs + 1):
		train(epoch)
	test()
	#net.save()

def main():
	net = Net.cnn_model([28,28], [10])
	if net.exist():
		log("model exist!")
	else:
		run(net)

if __name__ == "__main__":
	main()

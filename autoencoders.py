import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
import time
from torchvision import datasets
import re
import matplotlib.pyplot as plt
from pylab import savefig
from itertools import permutations
import os

# Hyper Parameters
input_size = 52
hidden_size1 = 1024
hidden_size2 = 1024
num_classes = 23
num_epochs = 10
batch_size = 128
learning_rate = 0.001
l2_classes = 164

#This file converts filedata into torch tensor
def convert_file(filename):
	train = open(filename, 'r+')
	data = []
	for line in train:
		splitted = line.rstrip().split(" ")
		data.append([float(x) for x in splitted])
	data = np.array(data)
	train.close()
	return torch.from_numpy(data).float()

def learning_curve(filename, string):
	f = open(filename, 'r+')
	loss = []
	pre = []
	for line in f:
		figures = re.findall("\d+\.\d+", line.rstrip()) #Extract decimals from a string
		loss.append(float(figures[0])) #First number is the loss
		pre.append(float(figures[1]))
	y1_loss = loss[::2] #x1 is the training numbers at even positions
	y2_loss = loss[1::2] #x2 is the validation numbers at odd positions
	y1_pre = pre[::2] #x1 is the training numbers at even positions
	y2_pre = pre[1::2] #x2 is the validation numbers at odd positions
	x = np.arange(1, len(y1_loss) + 1, 1)
	#Plot loss learning curve
	loss = plt.figure(1)
	plt.plot(x, y1_loss, 'r', label="Training")
	plt.plot(x, y2_loss, 'g', label="Validation")
	plt.xlabel('Iterations')
	plt.ylabel('Loss')
	plt.legend(loc='upper left')
	loss = plt.gcf()
	plt.draw()
	loss.savefig(os.path.join('Curves', 'Loss Learning Curve' + string + '.pdf'))
	plt.close()
	#Plot precision learning curve
	pre = plt.figure(2)
	plt.plot(x, y1_pre, 'r', label="Training")
	plt.plot(x, y2_pre, 'g', label="Validation")
	plt.xlabel('Iterations')
	plt.ylabel('Precision')
	plt.legend(loc='upper left')
	pre = plt.gcf()
	plt.draw()
	pre.savefig(os.path.join('Curves','Precision Learning Curve' + string + '.pdf'))
	plt.close()

input_data = convert_file(os.path.join("Data", "app.txt"))
output_data = convert_file(os.path.join("Data","l2.txt"))

test_input = convert_file(os.path.join("Data","test_app.txt"))
test_output = convert_file(os.path.join("Data","test_l2.txt"))

train_dataset = Data.TensorDataset(data_tensor = input_data, target_tensor = output_data)
test_dataset = Data.TensorDataset(data_tensor = test_input, target_tensor = test_output)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
										   batch_size=batch_size,
										   shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
										   batch_size=batch_size,
										   shuffle=False)

#Make a dictionary defining training and validation sets
dataloders = dict()
dataloders['train'] = train_loader
dataloders['val'] = test_loader

dataset_sizes = {'train': 6000, 'val': 1000}

use_gpu = torch.cuda.is_available()

#Masking Loss function
#Loss = y(x-y)^2
class Mask_Loss(torch.nn.Module):
	def __init__(self):
		super(Mask_Loss,self).__init__()

	def forward(self,x,y):
		mseloss = (x - y) ** 2
		mask = torch.mm(torch.t(y), mseloss)
		return mask.sum()

#Initialize weights with Xavier normalization
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		nn.init.xavier_normal(m.weight.data)

# Neural Network Model (2 hidden layers)
class Net(nn.Module):
	def __init__(self, input_size, hidden_size1, hidden_size2, num_classes, dropout):
		#Dropout is a array. Each index corresponds to different layers of network
		super(Net, self).__init__()
		self.fc1 = nn.Sequential(
			nn.Linear(input_size, hidden_size1),
			nn.Dropout(dropout[0]),
			nn.ReLU())
		self.fc2 = nn.Sequential(
			nn.Linear(hidden_size1, hidden_size2),
			nn.Dropout(dropout[1]),
			nn.ReLU())
		self.fc3 = nn.Sequential(
			nn.Linear(hidden_size2, num_classes),
			nn.Dropout(dropout[2]))

	def forward(self, x):
		out = self.fc1(x)
		out = self.fc2(out)
		out = self.fc3(out)
		return out

def train_model(model, criterion, optimizer, num_epochs):
	f = open("Iterations.txt", "w+")
	since = time.time()
	best_model_wts = model.state_dict()
	best_val_acc = 0.0
	best_train_acc = 0.0
	for epoch in range(num_epochs):
		# print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		# print('-' * 10)
		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				model.train(True)  # Set model to training mode
			else:
				model.train(False)  # Set model to evaluate mode
			running_loss = 0.0
			running_corrects = 0
			# Iterate over data.
			distances = open("Output Values.txt", "w+")
			for data in dataloders[phase]:
				# get the inputs
				inputs, labels = data
				# wrap them in Variable
				if use_gpu:
					inputs = Variable(inputs.cuda())
					labels = Variable(labels.cuda())
				else:
					inputs, labels = Variable(inputs), Variable(labels)
				# zero the parameter gradients
				optimizer.zero_grad()
				# forward
				outputs = model(inputs)
				_, preds = torch.max(outputs.data, 1)
				loss = criterion(outputs, labels)
				array_string = '\n'.join(' '.join('%0.8f' %x for x in y) for y in outputs.data.cpu().numpy()) #Change this to CUDA
				distances.write(array_string + "\n")
				#Calculate the max of the label vector
				_,idx = torch.max(labels.data, 1)
				# backward + optimize only if in training phase
				if phase == 'train':
					loss.backward()
					optimizer.step()
				# statistics
				running_loss += loss.data[0]
				running_corrects += torch.sum(preds == idx)
			distances.close()
			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects / dataset_sizes[phase]
			#Print it out Loss and Accuracy and in the file torchvision
			print('{} Accuracy: {:.8f} P@1: {:.4f}'.format(phase, epoch_loss, epoch_acc))
			f.write('{} Accuracy: {:.8f} P@1: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))
			# deep copy the model
			if phase == 'val' and epoch_acc > best_val_acc:
				best_val_acc = epoch_acc
				best_model_wts = model.state_dict()
			if phase == 'train' and epoch_acc > best_train_acc:
				best_train_acc = epoch_acc
				best_model_wts = model.state_dict()
	f.close()
	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_val_acc))
	# load best model weights
	model.load_state_dict(best_model_wts)
	return model, best_train_acc, best_val_acc

#Make possible values of dropout. Use from 0.0 to 1.0 but exclude 1.0
dropout_values = np.linspace(0.0, 1.0, num=21).tolist()[:-1]
dropout_exp = open("Values.txt", "w+")
dropout_values = [0.0, 0.0, 0.0]
# net = Net(input_size, hidden_size1, hidden_size2, num_classes, dropout_values)
# net.load_state_dict(torch.load('app2l1.pkl'))

class Addlayers(nn.Module):
	def __init__(self, input_size, hidden_size1, hidden_size2, num_classes, dropout_values, l2_classes):
		super(Addlayers, self).__init__()
		net = Net(input_size, hidden_size1, hidden_size2, num_classes, dropout_values)
		net.load_state_dict(torch.load('app2l1.pkl'))
		self.prevmodel = nn.Sequential(*list(net.children()))
		self.projective = nn.Linear(num_classes, 100)
		self.nonlinearity = nn.ReLU()
		self.projective2 = nn.Linear(100, l2_classes)

	def forward(self,x):
		x = self.prevmodel(x)
		x = self.projective(x)
		x = self.nonlinearity(x)
		x = self.projective2(x)
		return x

model = Addlayers(input_size, hidden_size1, hidden_size2, num_classes, dropout_values, l2_classes)

# Loss and Optimizer
criterion = Mask_Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if use_gpu:
	model_ft, train_acc, test_acc = train_model(model.cuda(), criterion, optimizer, num_epochs)
else:
	model_ft, train_acc, test_acc = train_model(model, criterion, optimizer, num_epochs)

torch.save(model_ft.state_dict(), 'app2l2.pkl')

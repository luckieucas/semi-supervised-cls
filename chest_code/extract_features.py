import torch
import numpy as np
import random  
import faiss  
import time
import scipy
import torch.nn.functional as F
from faiss import normalize_L2
from models import DenseNet121,DenseNet121_F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from DatasetGenerator import DatasetGenerator, TwoStreamBatchSampler
from metrics import compute_metrics_test_new

model = DenseNet121_F()
# model = DenseNet121()
# model = torch.nn.DataParallel(model)
# checkpoint = torch.load('/media/luckie/vol4/semi_supervised_cls/model/vat_best_model.pth.tar')
# model.load_state_dict(checkpoint['state_dict'])


# img = torch.rand(1,3,224,224)
# img = img.cuda()
# logit = model(img)
print("666")
resize = 256
crop_size = 224

random.seed(1337)
torch.manual_seed(1337)

# generate dataloader 
# transform
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# train and validation data transform
trainval_transforms_list = []
trainval_transforms_list.append(transforms.RandomResizedCrop(crop_size))
trainval_transforms_list.append(transforms.RandomHorizontalFlip())
trainval_transforms_list.append(transforms.ToTensor())
trainval_transforms_list.append(normalize)
trainval_transform_sequence = transforms.Compose(trainval_transforms_list)

#test data transfrom
test_transforms_list = []
test_transforms_list.append(transforms.Resize(resize))
test_transforms_list.append(transforms.TenCrop(crop_size))
test_transforms_list.append(transforms.Lambda(lambda crops: \
    torch.stack([transforms.ToTensor()(crop) for crop in crops])))
test_transforms_list.append(transforms.Lambda(lambda crops: \
    torch.stack([normalize(crop) for crop in crops])))
test_transform_sequence = transforms.Compose(test_transforms_list)


datasetTrain = DatasetGenerator(pathImageDirectory='../../dataset/skin/training_data/', 
            pathDatasetFile='../../dataset/skin/train_labeled.txt',transform=trainval_transform_sequence)

dataLoaderTrain = DataLoader(dataset=datasetTrain, num_workers=8,batch_size=32,
                                    pin_memory=True)

datasetTest = DatasetGenerator(pathImageDirectory='../../dataset/skin/training_data/', 
            pathDatasetFile='../../dataset/skin/testing.txt',transform=test_transform_sequence)

dataLoaderTest = DataLoader(dataset=datasetTest, num_workers=8,batch_size=32,
                                    pin_memory=True)


# extract_features
model = model.cuda()
model.eval()
embeddings_all_train, labels_all_train, index_all_train = [], [], []
embeddings_all_test, labels_all_test, index_all_test = [], [], []

for i,(input,target) in enumerate(dataLoaderTrain):
    input_var = torch.autograd.Variable(input.cuda())
    target_var = torch.autograd.Variable(target.cuda())
    _, feats = model(input_var)
    embeddings_all_train.append(feats.data.cpu())
    labels_all_train.append(target_var.data.cpu())

outPRED = torch.FloatTensor().cuda()
# predict test data
for i, (input,target) in enumerate(dataLoaderTest):
    target = target.cuda()
    bs, n_crops, c, h, w = input.size()
    input = input.cuda()
    with torch.no_grad():
        out,feats = model(input.view(-1, c, h, w).cuda())
        #out = model(input.view(-1, c, h, w).cuda())
    featsMean = feats.view(bs, n_crops, -1).mean(1)
    embeddings_all_test.append(featsMean.data.cpu())
    labels_all_test.append(target.data.cpu())
    outMean = out.view(bs, n_crops, -1).mean(1)
    outPRED = torch.cat((outPRED, outMean.data), 0)

embeddings_all_train = np.asarray(torch.cat(embeddings_all_train).numpy())
labels_all_train = torch.cat(labels_all_train).numpy()

embeddings_all_test = np.asarray(torch.cat(embeddings_all_test).numpy())
labels_all_test = torch.cat(labels_all_test).numpy()
embeddings_all = np.concatenate((embeddings_all_train,embeddings_all_test), axis=0)

labels_all = np.concatenate((labels_all_train,labels_all_test),axis=0)
outPRED = outPRED.cpu().numpy()

# KNN search for the graph
k = 20
X = embeddings_all
d = X.shape[1]
res = faiss.StandardGpuResources()
flat_config = faiss.GpuIndexFlatConfig()
flat_config.device = int(torch.cuda.device_count()) - 1
index = faiss.GpuIndexFlatIP(res,d,flat_config)   # build the index

normalize_L2(X)
index.add(X) 
N = X.shape[0]
Nidx = index.ntotal

c = time.time()
D, I = index.search(X, k + 1)
elapsed = time.time() - c
print('kNN Search done in %d seconds' % elapsed)

# Create the graph   
D = D[:,1:] ** 3
I = I[:,1:]
row_idx = np.arange(N)
row_idx_rep = np.tile(row_idx,(k,1)).T 
W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape= (N,N))
W[W>0] = 1
D = scipy.sparse.eye(W.shape[0])* k
L = D - W

P1 = scipy.sparse.eye(1400).toarray()
P2 = np.zeros([2015,1400])
P = np.concatenate((P1,P2), axis=0)
Y_GT = labels_all_train.T
B = np.matmul(Y_GT,P.T)/N
A = np.matmul(P,P.T)/N +10*L
Y_pred = B * A.I
Y_pred_tensor = torch.from_numpy(Y_pred.T)
Y_soft = F.softmax(Y_pred_tensor*1000.0,dim=1)
AUROCs, Accus, Senss, Specs, Pre, F1 = compute_metrics_test_new(labels_all_test, Y_soft[1400:], competition=True)
print("test")

# W = W + W.T

# #Normalize the graph 
# W = W -scipy.sparse.diags(W.diagonal())
# S = W.sum(axis = 1)
# S[S==0] = 1
# D = np.array(1./ np.sqrt(S))
# D = scipy.sparse.diags(D.reshape(-1))
# Wn = D * W * D
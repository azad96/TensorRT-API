import cv2
import numpy as np
import torch
from torch import nn
import torchvision

from torch.nn import functional as F
from collections import OrderedDict as OD

'''
https://github.com/facebookresearch/multigrain
'''

def l2n(x, eps=1e-6, dim=1):
    x = x / (torch.norm(x, p=2, dim=dim, keepdim=True) + eps).expand_as(x)
    return x

def flatten(x, keepdims=False):
    """
    Flattens B C H W input to B C*H*W output, optionally retains trailing dimensions.
    """
    y = x.view(x.size(0), -1)
    if keepdims:
        for d in range(y.dim(), x.dim()):
            y = y.unsqueeze(-1)
    return y

class GEM(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,x):
        x = x.clamp(min=1e-6)
        x = F.avg_pool2d(x.pow(3.0), (x.size(-2), x.size(-1))).pow(1.0 / 3.0)
        x = flatten(x)
        return x
        
class MultiGrainNet(nn.Module):
    """
    Implement MultiGrain by changing the pooling layer of the backbone into GeM pooling with exponent p,
    and adding DistanceWeightedSampling for the margin loss.
    """
    def __init__(self):
        super().__init__()
        net = torchvision.models.resnet50(pretrained=None)
        children = list(net.named_children())
        self.features = nn.Sequential(OD(children[:-2]))
        self.pool = GEM()
            
    def forward(self, input):
        output_dict = {}
        output_dict['embedding'] = self.pool(self.features(input))
        output_dict['normalized_embedding'] = l2n(output_dict['embedding'])     
        return output_dict

class MultiGrainSimilarity:
    def __init__(self):
        self.net = MultiGrainNet()
        checkpoint = torch.load('weight.pth')
        self.net.load_state_dict(checkpoint['model_state'], strict=False)
        self.net.cuda().eval()

    def euclidian(self,feat1,feat2):
        dist = (feat1-feat2)**2
        dist = np.sum(dist,axis=1)
        dist = np.sqrt(dist)
        return dist

    def process_image(self,image):
        mean=np.array([0.485, 0.456, 0.406])
        std=np.array([0.229, 0.224, 0.225])
        image = cv2.resize(image, (224, 224),interpolation=cv2.INTER_AREA)
        image = (image/255. - mean) / std
        return torch.from_numpy(image).float().permute(2,0,1)[None].cuda()
    
    def similarityScore(self,image1,image2):
        feat1 = self.net(self.process_image(image1))['normalized_embedding'].cpu().numpy().reshape(1,2048)
        feat2 = self.net(self.process_image(image2))['normalized_embedding'].cpu().numpy().reshape(1,2048)        
        score = self.euclidian(feat1,feat2)
        return 1-score[0]
       

if __name__=='__main__':
    IMAGE_PATH1 = '../image/dog1.jpg'
    IMAGE_PATH2 = '../image/dog2.jpg'
    image1 = cv2.imread(IMAGE_PATH1)
    image2 = cv2.imread(IMAGE_PATH2)

    with torch.no_grad():
        score = MultiGrainSimilarity()
        print(score.similarityScore(image1,image2))
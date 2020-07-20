import torch
import torchvision
import struct

def main():   
    model = torchvision.models.resnet50(pretrained=True)
    model.cuda().eval()
    # print(model)
    # return
    f = open('resnet50.wts', 'w')
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        print('key: ', k)
        print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f',float(vv)).hex())
        f.write('\n')

if __name__=='__main__':
    main()
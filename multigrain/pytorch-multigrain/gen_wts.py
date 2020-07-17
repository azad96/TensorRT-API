from multigrain import MultiGrainNet
import torch
import struct

def main():   
    model = MultiGrainNet()
    checkpoint = torch.load('weight.pth')
    model.load_state_dict(checkpoint['model_state'], strict=False)
    model.cuda().eval()
    # print(model)
    # return
    f = open('multigrain.wts', 'w')
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
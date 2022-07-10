import struct
import torch
from model.simple import Simple
import argparse


def main(args):
    model = Simple()
    # torch.save(model.state_dict(), 'weight.pth')
    state_dict = torch.load(args.model_weight_file)
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()

    # dummy_input = torch.rand(3, 4, 4).cuda()
    dummy_input = torch.arange(3*4*4, dtype=torch.float).reshape(3, 4, 4).cuda()
    output = model(dummy_input)
    print(f'output shape: {output.shape}')
    print(f'output:')
    tmp = output.flatten()
    for val in tmp:
        print(f'{val.item():.3f}', end=' ')
    print()
    exit()

    f = open('simple.wts', 'w')
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weight_file', type=str, default='weight.pth')   
    args = parser.parse_args()

    main(args)

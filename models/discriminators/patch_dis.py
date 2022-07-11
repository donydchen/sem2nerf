# patch discriminator, base on GRAF's code
import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    def __init__(self, size, input_nc=3, ndf=64, hflip=False):
        super(PatchDiscriminator, self).__init__()
        assert(size == 32 or size == 64 or size == 128)
        self.imsize = size
        self.hflip = hflip

        SN = torch.nn.utils.spectral_norm
        def IN(x): return nn.InstanceNorm2d(x)

        blocks = []
        if self.imsize == 128:
            blocks += [
                # input is (nc) x 128 x 128
                SN(nn.Conv2d(input_nc, ndf//2, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                # input is (ndf//2) x 64 x 64
                SN(nn.Conv2d(ndf//2, ndf, 4, 2, 1, bias=False)),
                IN(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                SN(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
                #nn.BatchNorm2d(ndf * 2),
                IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        elif self.imsize == 64:
            blocks += [
                # input is (nc) x 64 x 64
                SN(nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                SN(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
                #nn.BatchNorm2d(ndf * 2),
                IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        else:
            blocks += [
                # input is (nc) x 32 x 32
                SN(nn.Conv2d(input_nc, ndf * 2, 4, 2, 1, bias=False)),
                #nn.BatchNorm2d(ndf * 2),
                IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        blocks += [
            # state size. (ndf*2) x 16 x 16
            SN(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            #nn.BatchNorm2d(ndf * 4),
            IN(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            SN(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            #nn.BatchNorm2d(ndf * 8),
            IN(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            SN(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
            # nn.Sigmoid()
        ]
        blocks = [x for x in blocks if x]
        self.main = nn.Sequential(*blocks)
        
    def forward(self, input):
        if self.hflip:      # Randomly flip input horizontally
            input_flipped = input.flip(3)
            mask = torch.randint(0, 2, (len(input), 1, 1, 1)).bool().expand(-1, *input.shape[1:])
            input = torch.where(mask, input, input_flipped)

        return self.main(input)

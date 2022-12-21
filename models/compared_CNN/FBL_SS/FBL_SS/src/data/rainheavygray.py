import os
from data import srdata

class RainHeavyGray(srdata.SRData):
    def __init__(self, args, name='RainHeavyGray', train=True, benchmark=False):
        super(RainHeavyGray, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(RainHeavyGray, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(RainHeavyGray, self)._set_filesystem(dir_data)
        self.apath = '/home/yangwenhan/pytorch_project/data_dir/rain_removal/rain_heavy_gray/'

        print(self.apath)
        self.dir_hr = os.path.join(self.apath, 'norain')
        self.dir_lr = os.path.join(self.apath, 'rain')


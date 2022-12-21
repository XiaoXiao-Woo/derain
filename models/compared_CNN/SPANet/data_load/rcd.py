import os
from data_load import srdata

class RainHeavy(srdata.SRData):
    def __init__(self, args, name='RainHeavy', train=True, benchmark=False):
        super(RainHeavy, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    # def _scan(self):
    #     names_hr, names_lr = super(RainHeavy, self)._scan()
    #     names_hr = names_hr[self.begin - 1:self.end]
    #     names_lr = [n[self.begin - 1:self.end] for n in names_lr]
    #
    #     return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(RainHeavy, self)._set_filesystem(dir_data)

        # self.apath = 'D:/Datasets/derain/Rain100L/train'
        self.apath = self.args.dir_data
        print(self.apath)
        self.dir_hr = os.path.join(self.apath, 'norain')
        self.dir_lr = os.path.join(self.apath, 'rain')

class RainHeavyTest(srdata.SRData):
    def __init__(self, args, name='RainHeavyTest', train=True, benchmark=False):
        super(RainHeavyTest, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )


    # def _scan(self):
    #     names_hr, names_lr = super(RainHeavyTest, self)._scan()
    #     names_hr = names_hr[self.begin - 1:self.end]
    #     names_lr = [n[self.begin - 1:self.end] for n in names_lr]
    #
    #     return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(RainHeavyTest, self)._set_filesystem(dir_data)
        # self.apath = f'D:/Datasets/derain/Rain100L/test'
        self.apath = self.args.dir_data
        print(self.apath)
        self.dir_hr = os.path.join(self.apath, 'norain')
        self.dir_lr = os.path.join(self.apath, 'rain')
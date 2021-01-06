import os 
import mxnet as mx 
import mxnet.gluon.data.Dataset as Dataset 

from gbox import DATASETS


@DATASETS.register()
class BaseDataset(Dataset):
    def __init__(self,
                 ann_file, 
                 img_root,
                 processer,
                 test_mode=False,
                 filter_empty=True,
                 min_size=32):
        super(BaseDataset, self).__init__()
        self.ann_file = ann_file 
        self.img_root = img_root 
        self.test_mode = test_mode 
        self.filter_empty = filter_empty
        self.min_size = min_size
        self._processer = get_processer(processer)
        self._path_check()


    def _load_annotations(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def processer(self, img_info):
        for f in self._processer:
            img_info = f(img_info)
        return img_info

    def _filter(self):
        raise NotImplementedError 

    def _path_check(self):
        if not os.path.isdir(self.img_root):
            raise ValueError('{} is not a iamge dir!'.format(self.img_root))

        if not os.path.isfile(self.ann_file):
            if not os.path.isfile(os.path.join(self.img_root, self.ann_file)):
                raise ValueError('{} is not a annotation file!'.format(self.ann_file))
            else:
                self.ann_file = os.path.join(self.img_root, self.ann_file)

    def prepare_test_img(self, idx):
        raise NotImplementedError 

    def prepare_train_img(self, idx):
        raise NotImplementedError 
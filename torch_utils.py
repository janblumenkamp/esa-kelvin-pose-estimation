import torch
from torch.utils.data import Dataset
from torchvision import transforms

class PyTorchSatellitePoseEstimationDataset(Dataset):

    """ SPEED dataset that can be used with DataLoader for PyTorch training. """

    def __init__(self, split='train', speed_root='', transform=None):

        if not has_pytorch:
            raise ImportError('Pytorch was not imported successfully!')

        if split not in {'train', 'test', 'real_test'}:
            raise ValueError('Invalid split, has to be either \'train\', \'test\' or \'real_test\'')

        with open(os.path.join(speed_root, split + '.json'), 'r') as f:
            label_list = json.load(f)

        self.sample_ids = [label['filename'] for label in label_list]
        self.train = split == 'train'

        if self.train:
            self.labels = {label['filename']: {'q': label['q_vbs2tango'], 'r': label['r_Vo2To_vbs_true']}
                           for label in label_list}
        self.image_root = os.path.join(speed_root, 'images', split)

        self.transform = transform

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        img_name = os.path.join(self.image_root, sample_id)

        # note: despite grayscale images, we are converting to 3 channels here,
        # since most pre-trained networks expect 3 channel input
        pil_image = Image.open(img_name).convert('RGB')

        if self.train:
            q, r = self.labels[sample_id]['q'], self.labels[sample_id]['r']
            y = np.concatenate([q, r])
        else:
            y = sample_id

        if self.transform is not None:
            torch_image = self.transform(pil_image)
        else:
            torch_image = pil_image

        return torch_image, y

import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from deep_sort.deep.model import Net


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=torch.device(self.device))[
            'net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def preprocessing(self, img_crop):
        
        def img_resize(im, size):
            resize = cv2.resize(im.astype(np.float32)/255., size)
            return resize

        im_batch = torch.cat([self.norm(img_resize(im, self.size)).unsqueeze(
            0) for im in img_crop], dim=0).float()
        return im_batch

    def __call__(self, img_crop):
        im_batch = self.preprocessing(img_crop)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)

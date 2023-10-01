import numpy as np
import os
import cv2
import torch
import kornia.feature as KF
from extract_patches.core import extract_patches


def extimate_affine_shape(kpts,img, affnet, dev = torch.device('cpu'), ellipse=False):
    affnet = affnet.to(dev)
    affnet.eval()
    patches = np.array(
        extract_patches(
            kpts, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 32, 12., 'cv2'
        )
    ).astype(np.float32)
    bs = 128
    aff = np.zeros((len(patches), 3))
    for i in range(0, len(patches), bs):
        data_a = torch.from_numpy(
            patches[i:min(i + bs, len(patches)),  :, :]
        ).unsqueeze(1).to(dev)
        with torch.no_grad():
            out_a = affnet(data_a)
            aff[i:i + bs] = out_a.view(-1, 3).cpu().detach().numpy()
    aff = torch.from_numpy(aff).to(dev)
    if ellipse:
        aff = aff.unsqueeze(1)
        laf = KF.ellipse_to_laf(
            torch.cat([torch.zeros_like(aff[...,:2]),aff], dim=2)
        )
    else:
        aff2 = torch.cat([
            aff[:,0:1],
            torch.zeros_like(aff[:,0:1]),
            aff[:,1:2],
            aff[:,2:3]],
            dim=1
        ).reshape(-1,2,2)
        laf = torch.cat([
            aff2, torch.zeros_like(aff2[:,:, 0:1])
        ],dim=2).unsqueeze(1)
    
    ls = KF.get_laf_scale(laf)
    laf2 = KF.scale_laf(
        KF.make_upright(laf), 1./ls
    ).squeeze(1)
    return laf2[:,:2,:2].detach().cpu().numpy()


def orinet_radians(inp, orinet):
    yx = orinet(inp)
    return torch.atan2(yx[:,0],yx[:,1])


def estimate_orientation(kpts, img, As, orinet, dev = torch.device('cpu')):
    orinet = orinet.to(dev)
    orinet.eval()
    patches = np.array(
        extract_patches(
            (kpts,As),
            cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 32, 12., 'cv2+A'
        )
    ).astype(np.float32)
    bs = 128
    aff = np.zeros((len(patches)))
    for i in range(0, len(patches), bs):
        data_a = torch.from_numpy(
            patches[i:min(i + bs, len(patches)),  :, :]
        ).unsqueeze(1).to(dev)
        with torch.no_grad():
            out_a = orinet_radians(data_a, orinet)
            aff[i:i + bs] = out_a.cpu().detach().numpy()
    aff = np.rad2deg(-aff)
    return aff


def extract_descriptors(kpts, img, As, descnet, dev=torch.device('cpu')):
    descnet = descnet.to(dev)
    descnet.eval()
    patches = np.array(
        extract_patches(
            (kpts,As),
            cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
            32, 12., 'cv2+A'
        )
    ).astype(np.float32)
    bs = 128
    desc = np.zeros((len(patches), 128))
    for i in range(0, len(patches), bs):
        data_a = torch.from_numpy(
            patches[i:min(i + bs, len(patches)),  :, :]
        ).unsqueeze(1).to(dev)
        with torch.no_grad():
            out_a = descnet(data_a)
            desc[i:i + bs] = out_a.cpu().detach().numpy()
    return desc


def extract_sift_keypoints_upright(img, n_feat = 5000):
    sift = cv2.xfeatures2d.SIFT_create(2 * n_feat, 
            contrastThreshold=-10000, edgeThreshold=-10000)
    keypoints = sift.detect(img, None)
    response = np.array([kp.response for kp in keypoints])
    respSort = np.argsort(response)[::-1]
    kpts = [
        cv2.KeyPoint(
            keypoints[i].pt[0], keypoints[i].pt[1], keypoints[i].size, 0
        )
        for i in respSort
    ]
    kpts_unique = []
    for x in kpts:
        if x not in kpts_unique:
            kpts_unique.append(x)
    return kpts_unique[:n_feat]


def match_snn(desc1, desc2, snn_th = 0.8):
    index_params = dict(algorithm=1, trees=4)
    search_params = dict(checks=128)  # or pass empty dictionary
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(
        desc1.astype(np.float32),
        desc2.astype(np.float32), k=2
    )
    good_matches = []
    for m,n in matches:
        if m.distance < snn_th * n.distance:
            good_matches.append(m)
    return good_matches

def detect_affnet_descriptors(img, nfeats = 5000, dev=torch.device('cpu')):
    hardnet = KF.HardNet(True).to(dev).eval()

    affnet = torch.jit.load('AffNetJIT.pt').to(dev).eval()
    affnet.eval()

    orinet = torch.jit.load('OriNetJIT.pt').to(dev).eval()
    orinet.eval()

    kpts = extract_sift_keypoints_upright(img, nfeats)
    As = extimate_affine_shape(kpts, img, affnet, dev)
    ori = estimate_orientation(kpts, img, As, orinet, dev)
    kpts_new = [
        cv2.KeyPoint(x.pt[0], x.pt[1], x.size, ang)
        for x, ang in zip(kpts,ori)
    ]
    descs = extract_descriptors(
        kpts_new, img, As, hardnet, dev
    )
    print('number of descs: ', len(descs))
    return kpts_new, descs, As


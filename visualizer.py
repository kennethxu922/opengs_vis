from ast import arg, parse
import re
from visergui import ViserViewer
from renderer import renderer
import torch 


import argparse

def parser():
    parser = argparse.ArgumentParser()


    parser.add_argument('--gs_ckpt', required=True, help="The place one put the gaussian ckpt, i.e. pretrained model")
    parser.add_argument('--feat_pt', required=True, help="place where one keep there features as pt")

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parser()



    with torch.no_grad():
        gs_renderer = renderer(
            gaussian_ckpt=args.gs_ckpt,
            feature_path=args.feat_pt,
        )

        viser = ViserViewer(
            device='cuda',
            viewer_port=6777
        )

        viser.set_renderer(gs_renderer)

        while True:
            viser.update()
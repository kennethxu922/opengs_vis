from pydoc import text
from visergui import ViserViewer
from renderer import renderer
import torch 
from feature_mapper import feature_lift_mapper, feature_lift_mapper_config
from kernel_loader import base_kernel_loader_config, general_gaussian_loader
import argparse
import os

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gs_ckpt', required=True, help="Path to the Gaussian PLY file")
    parser.add_argument('--feat_pt', required=False, default=None, 
                        help="Path to feature file (optional for OpenGaussian PLY files)")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parser()

    # Check if the file exists
    if not os.path.exists(args.gs_ckpt):
        print(f"Error: File not found: {args.gs_ckpt}")
        exit(1)

    # Print info about file types
    print(f"Loading model from: {args.gs_ckpt}")
    if args.feat_pt:
        print(f"Loading external features from: {args.feat_pt}")
    else:
        if args.gs_ckpt.endswith('.ply'):
            print("No external feature file provided. Will look for embedded features in PLY file.")
        else:
            print("Warning: No feature file provided. Running in RGB-only mode.")

    feature_lift_config = feature_lift_mapper_config(text_tokenizer='featup', text_encoder='maskclip')
    feature_mapper = feature_lift_mapper(config=feature_lift_config)

    gs_kernel_loader_config = base_kernel_loader_config(
        kernel_location=args.gs_ckpt,
        feature_location=args.feat_pt
    )

    gs_kernel_loader = general_gaussian_loader(gs_kernel_loader_config)

    with torch.no_grad():
        gs_renderer = renderer(
            gaussian_loader=gs_kernel_loader,
            feature_mapper=feature_mapper
        )

        viser = ViserViewer(
            device='cuda',
            viewer_port=6777
        )

        viser.set_renderer(gs_renderer)
        
        print("Visualization server started at http://localhost:6777")
        print("Use text prompts in the UI to visualize feature attention")

        while True:
            viser.update()
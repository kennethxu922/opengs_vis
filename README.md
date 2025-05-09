# Embedding Visualization Module
To visualize the result, we provide additional viser package that can visualize our attneion score in rendering mode

Since we are using FeatUP package, the user require to also use featup. 


# Embedding Visualization Robust Module
- Here, we want to visualize different input Gaussian and Text Feature Mapper


## Current Support Package:
We might support non-Gaussian Kernel Rendering
- Feature Lifting(Ours)
- OpenGaussian
- LangSplat Model
- Feature Splat

## Code Architecture
- Kernel_Loader
- Feature_Mapper
- Renderer(Kernel_Loader, Feature_Mapper)

# Feature Viewer
- We seperate the loading module, to additional space comapre to master branck
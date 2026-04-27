# Stereo Matching GA (Quadtree) 🌲

[![C++](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

An evolutionary approach to the stereo matching problem using Quadtree structures to handle multi-resolution disparity maps. Based on the research by **Gong & Yang**.

## ⚡ Quick Start
1. Ensure **vcpkg** is installed and integrated.
2. Open the solution in Visual Studio 2022.
3. Switch build configuration to **Release / x64**.
4. Press `F5` to build and run.

## 🧠 Algorithm Overview
- **Representation:** Quadtree leaf nodes encode local disparity values.
- **Selection:** Elitist strategy to prevent energy increase over generations.
- **Mutation:** Adaptive Split/Merge/Alter operations based on energy minimization.
- **Fitness:** Energy function $f(D) = \text{Data Term} + \lambda \times \text{Smoothness Term}$.

## 🚧 Current Status
- [x] Binocular Stereo Support (Aloe dataset)
- [x] Vcpkg integration (Manifest Mode)
- [ ] Multi-view occlusion detection

## 📚 References
* Gong, M., & Yang, Y. H. (2002). "Genetic-Based Stereo Algorithm and Disparity Map Evaluation". *International Journal of Computer Vision*.

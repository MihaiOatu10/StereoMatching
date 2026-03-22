#pragma once

    namespace Config {
        const int WIDTH = 427;
        const int HEIGHT = 370;
        const int MAX_DISPARITY = 28;
        const int MAX_DEPTH = 9;
        const int POP_SIZE = 60;
	    const int GENERATIONS = 50;
        const float CONVERGENCE_THRESHOLD = 0.0001f;
        const int WINDOW_RADIUS = 0;
        const float SMOOTHNESS_LAMBDA = 4.0f;  
        const float INITIAL_SPLIT_PROB = 0.5f;
        const float MRF_SYMMETRY_FACTOR = 1.0f;
		const float OCCLUSION_THRESHOLD = 30.0f;
    }

struct NodeLocation {
    int k;
    int depth;
    int x, y;
};

struct NodeInfo {
    int depth;
    int x, y;
};
#pragma once

    namespace Config {
        extern int WIDTH;
        extern int HEIGHT;

        const int MAX_DISPARITY = 50;
		const int MIN_DISPARITY = 0;
        const int MAX_DEPTH = 7;
        const int POP_SIZE = 40;
	    const int GENERATIONS = 800;
        const float CONVERGENCE_THRESHOLD = 0.0001f;
        const int WINDOW_RADIUS = 1;
        const float SMOOTHNESS_LAMBDA = 1.0f;  
        const float INITIAL_SPLIT_PROB = 0.5f;
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
#pragma once

    namespace Config {
        extern int WIDTH;
        extern int HEIGHT;

        constexpr int MAX_DISPARITY = 46;
		constexpr int MIN_DISPARITY = 0;
        constexpr int MAX_DEPTH = 7;
        constexpr int POP_SIZE = 40;
	    constexpr int GENERATIONS = 800;
        constexpr float CONVERGENCE_THRESHOLD = 0.0001f;
        constexpr int WINDOW_RADIUS = 1;
        constexpr float SMOOTHNESS_LAMBDA = 1.0f;  
        constexpr float INITIAL_SPLIT_PROB = 0.5f;
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
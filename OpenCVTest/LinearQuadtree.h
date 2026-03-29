#pragma once
#include <vector>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include "Common.h"

class StereoMatcher;

class LinearQuadtree{
private:
	int max_depth;
	int base_size;
	std::vector<int> D;
	double fitness;
	bool m_isDirty = true;

public:
	LinearQuadtree(int maxDepth, int baseSize)
		: max_depth(maxDepth), base_size(baseSize), fitness(0.0) {
		int total_nodes = ((1 << (2 * (max_depth + 1))) - 1) / 3;
		D.resize(total_nodes, -1);
	}
	int getIndex(int depth, int x, int y) const;
	cv::Rect getRect(int depth, int x, int y) const;
	std::vector<int>& getD() { return D; }
	int getMaxDepth() const { return max_depth; }
	int getBaseSize() const { return base_size; }
	double getFitness() const { return fitness; }
	int getGene(int k) const { return D[k]; }
	void setGene(int k, int value);
	void setFitness(double value) { fitness = value; }
	void buildRandom(int depth, int x, int y, StereoMatcher& matcher);
	void simplify(int depth, int x, int y);
	void drawOutline(cv::Mat& canvas, int depth, int x, int y) const;
	void drawDisparity(cv::Mat& canvas, int depth, int x, int y) const;
	NodeLocation findLeafAt(int px, int py) const;
	bool isDirty() const { return m_isDirty; }
	void makeDirty() { m_isDirty = true; }
	void cleanDirt() { m_isDirty = false; }
};
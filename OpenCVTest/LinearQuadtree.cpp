#include "LinearQuadtree.h"
#include "StereoMatcher.h"

int LinearQuadtree::getIndex(int depth, int x, int y) const{
	int offset = ((1 << (2 * depth)) - 1) / 3;
	int positioninGrid = (y * (1 << depth) + x);
	return offset + positioninGrid;
}

cv::Rect LinearQuadtree::getRect(int depth, int x, int y) const
{
	int nodeW = Config::WIDTH / (1 << depth);
	int nodeH = Config::HEIGHT / (1 << depth);

	cv::Rect target(x * nodeW, y * nodeH, nodeW, nodeH);

	cv::Rect bounds(0, 0, Config::WIDTH, Config::HEIGHT);
	return target & bounds;
}

void LinearQuadtree::setGene(int k, int value)
{
	if (D[k] != value) {
		D[k] = value;
		m_isDirty = true;
	}
}

void LinearQuadtree::buildRandom(int depth, int x, int y, StereoMatcher& matcher)
{
	int k = getIndex(depth, x, y);

	float randomProb = (float)rand() / RAND_MAX;
	bool shouldSplit = (randomProb > Config::INITIAL_SPLIT_PROB) && (depth < max_depth);

	if (shouldSplit)
	{
		D[k] = -1;
		buildRandom(depth + 1, x * 2, y * 2, matcher);
		buildRandom(depth + 1, x * 2 + 1, y * 2, matcher);
		buildRandom(depth + 1, x * 2, y * 2 + 1, matcher);
		buildRandom(depth + 1, x * 2 + 1, y * 2 + 1, matcher);
	}
	else
	{
		cv::Rect region = getRect(depth, x, y);
		int bestDisparity = Config::MIN_DISPARITY;
		float minCost = FLT_MAX;
		for (int d = Config::MIN_DISPARITY; d < Config::MAX_DISPARITY; d++) {
			int idx = d - Config::MIN_DISPARITY;
			cv::Scalar sumCost = cv::sum(matcher.getCostVolume()[idx](region));
			if (sumCost[0] < minCost) {
				minCost = sumCost[0];
				bestDisparity = d;
			}
		}
		D[k] = bestDisparity;
	}
	if (depth == 0) {
		simplify(0, 0, 0);
	}
}

void LinearQuadtree::simplify(int depth, int x, int y)
{
	if (depth >= max_depth) return;

	simplify(depth + 1, x * 2, y * 2);
	simplify(depth + 1, x * 2 + 1, y * 2);
	simplify(depth + 1, x * 2, y * 2 + 1);
	simplify(depth + 1, x * 2 + 1, y * 2 + 1);

	int idx1 = getIndex(depth + 1, x * 2, y * 2);
	int idx2 = getIndex(depth + 1, x * 2 + 1, y * 2);
	int idx3 = getIndex(depth + 1, x * 2, y * 2 + 1);
	int idx4 = getIndex(depth + 1, x * 2 + 1, y * 2 + 1);

	int v1 = D[idx1];
	int v2 = D[idx2];
	int v3 = D[idx3];
	int v4 = D[idx4];

	if (v1 != -1 && v2 != -1 && v3 != -1 && v4 != -1)
	{
		bool canMerge = (v1 == v2 && v1 == v3 && v1 == v4);

		if (canMerge)
		{
			D[getIndex(depth, x, y)] = v1;
			D[idx1] = -1;
			D[idx2] = -1;
			D[idx3] = -1;
			D[idx4] = -1;
		}
	}
}

void LinearQuadtree::drawOutline(cv::Mat& canvas, int depth, int x, int y) const
{
	int k = getIndex(depth, x, y);
	if(D[k] == -1 && depth < max_depth)
	{
		drawOutline(canvas, depth + 1, x * 2, y * 2);
		drawOutline(canvas, depth + 1, x * 2 + 1, y * 2);
		drawOutline(canvas, depth + 1, x * 2, y * 2 + 1);
		drawOutline(canvas, depth + 1, x * 2 + 1, y * 2 + 1);
	}
	else
	{
		cv::Rect region = getRect(depth, x, y);
		cv::rectangle(canvas, region, cv::Scalar(0, 255, 0), 1);
	}
}

void LinearQuadtree::drawDisparity(cv::Mat& canvas, int depth, int x, int y) const
{
	int k = getIndex(depth, x, y);
	if (D[k] == -1 && depth < max_depth)
	{
		drawDisparity(canvas, depth + 1, x * 2, y * 2);
		drawDisparity(canvas, depth + 1, x * 2 + 1, y * 2);
		drawDisparity(canvas, depth + 1, x * 2, y * 2 + 1);
		drawDisparity(canvas, depth + 1, x * 2 + 1, y * 2 + 1);
	}
	else
	{
		cv::Rect region = getRect(depth, x, y);
		cv::rectangle(canvas, region, cv::Scalar(D[k]), -1);
	}
}

NodeLocation LinearQuadtree::findLeafAt(int px, int py) const
{
	if (px < 0 || px >= Config::WIDTH || py < 0 || py >= Config::HEIGHT) {
		return { 0, 0, 0, 0 };
	}
	int curDepth = 0, curX = 0, curY = 0;
	while (curDepth < max_depth) {
		int k = getIndex(curDepth, curX, curY);
		if (D[k] != -1) return { k, curDepth, curX, curY };

		int nodeW = Config::WIDTH / (1 << curDepth);
		int nodeH = Config::HEIGHT / (1 << curDepth);
		int midX = curX * nodeW + nodeW / 2;
		int midY = curY * nodeH + nodeH / 2;

		curX = curX * 2 + (px >= midX ? 1 : 0);
		curY = curY * 2 + (py >= midY ? 1 : 0);
		curDepth++;
	}
	return { getIndex(curDepth, curX, curY), curDepth, curX, curY };
}



#ifndef GENETICALGORITHM_H
#define GENETICALGORITHM_H

#include <vector>
#include "Common.h"

class LinearQuadtree;
class StereoMatcher;

class GeneticAlgorithm {
private:
	struct NodeInfo { int depth, x, y; };
	int popSize;
	int mutRate;
	int d_max;
	std::vector<LinearQuadtree> population;

	void copySubTree(const LinearQuadtree& source, LinearQuadtree& dest, int depth, int x, int y);

public:
	GeneticAlgorithm(int populationSize, int mutationRate, int maxDisparity);
	void initialize(StereoMatcher& matcher, int baseSize);
	void findSeedNodes(const LinearQuadtree& p1, const LinearQuadtree& p2, int d, int x, int y, std::vector<NodeInfo>& seeds);
	LinearQuadtree graftCrossover(const LinearQuadtree& p1, const LinearQuadtree& p2);
	void mutate(LinearQuadtree& qt, StereoMatcher& matcher);
	void evolve(StereoMatcher& matcher);
	LinearQuadtree getBestIndividual() const;
	void mutateAlteration(LinearQuadtree& qt, NodeLocation k, StereoMatcher& matcher);
	void mutateSplitting(LinearQuadtree& qt, NodeLocation k, StereoMatcher& matcher);
	void mutateMerging(LinearQuadtree& qt, NodeLocation k, StereoMatcher& matcher);
};

#endif
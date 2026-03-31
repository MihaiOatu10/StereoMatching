#include "GeneticAlgorithm.h"
#include "LinearQuadtree.h"
#include "StereoMatcher.h"
#include "Common.h"
#include <chrono>

void GeneticAlgorithm::copySubTree(const LinearQuadtree& source, LinearQuadtree& dest, int depth, int x, int y)
{
	int k = source.getIndex(depth, x, y);
	dest.setGene(k, source.getGene(k));
	if (source.getGene(k) == -1 && depth < source.getMaxDepth())
	{
		copySubTree(source, dest, depth + 1, x * 2, y * 2);
		copySubTree(source, dest, depth + 1, x * 2 + 1, y * 2);
		copySubTree(source, dest, depth + 1, x * 2, y * 2 + 1);
		copySubTree(source, dest, depth + 1, x * 2 + 1, y * 2 + 1);
	}
}

GeneticAlgorithm::GeneticAlgorithm(int populationSize, int mutationRate, int maxDisparity)
	: popSize(populationSize), mutRate(mutationRate), d_max(maxDisparity) {
}

void GeneticAlgorithm::initialize(StereoMatcher& matcher, int baseSize)
{
	population.clear();
	for (int i = 0; i < popSize; i++) {
		LinearQuadtree individual(Config::MAX_DEPTH, baseSize);
		individual.buildRandom(0, 0, 0, matcher);
		population.push_back(individual);
	}
}

void GeneticAlgorithm::findSeedNodes(const LinearQuadtree& p1, const LinearQuadtree& p2, int d, int x, int y, std::vector<NodeInfo>& seeds)
{
	int k = p1.getIndex(d, x, y);
	int gene1 = p1.getGene(k);
	int gene2 = p2.getGene(k);

	bool isLeaf1 = (gene1 != -1);
	bool isLeaf2 = (gene2 != -1);

	if (isLeaf1 && isLeaf2) {
		if (gene1 != gene2) seeds.push_back({ d, x, y });
		return;
	}

	if (!isLeaf1 && !isLeaf2 && d < p1.getMaxDepth()) {
		findSeedNodes(p1, p2, d + 1, x * 2, y * 2, seeds);
		findSeedNodes(p1, p2, d + 1, x * 2 + 1, y * 2, seeds);
		findSeedNodes(p1, p2, d + 1, x * 2, y * 2 + 1, seeds);
		findSeedNodes(p1, p2, d + 1, x * 2 + 1, y * 2 + 1, seeds);
	}
}

LinearQuadtree GeneticAlgorithm::graftCrossover(const LinearQuadtree& p1, const LinearQuadtree& p2)
{
	std::vector<NodeInfo> seeds;
	findSeedNodes(p1, p2, 0, 0, 0, seeds);

	if (seeds.empty()) return p1;

	int randomIndex = rand() % seeds.size();
	NodeInfo seed = seeds[randomIndex];

	int coverDepth = std::max(0, seed.depth - 1);
	int coverX = seed.x / 2;
	int coverY = seed.y / 2;

	LinearQuadtree child = p1;
	copySubTree(p2, child, coverDepth, coverX, coverY);

	child.simplify(0, 0, 0);

	child.makeDirty();

	return child;
}

LinearQuadtree GeneticAlgorithm::getBestIndividual() const
{
	float bestFitness = FLT_MAX;
	int bestIdx = 0;
	for (size_t i = 0; i < population.size(); i++) {
		if (population[i].getFitness() < bestFitness) {
			bestFitness = population[i].getFitness();
			bestIdx = i;
		}
	}
	return population[bestIdx];
}

void GeneticAlgorithm::mutate(LinearQuadtree& qt, StereoMatcher& matcher)
{
	int width = Config::WIDTH;
	int height = Config::HEIGHT;
	int numpixels = width + height;
	float currentEnergy = matcher.calculateFitness(qt);

	for (int i = 0; i < numpixels; i++) {
		int rx = rand() % width;
		int ry = rand() % height;
		NodeLocation k = qt.findLeafAt(rx, ry);
		int choice = rand() % 3;
		if (choice == 0)
		{
			mutateAlteration(qt, k, matcher, currentEnergy);
		}
		else if (choice == 1)
		{
			mutateSplitting(qt, k, matcher, currentEnergy);
		}
		else {
			mutateMerging(qt, k, matcher, currentEnergy);
		}
	}
	qt.simplify(0, 0, 0);
}

void GeneticAlgorithm::mutateAlteration(LinearQuadtree& qt, NodeLocation k, StereoMatcher& matcher, float& oldEnergy)
{
	int originalDisp = qt.getGene(k.k);
	if (originalDisp == -1) return;

	float localEnergyBefore = matcher.calculateFitness(qt, k);

	std::set<int> neighborLabels;
	int nodeW = Config::WIDTH / (1 << k.depth);
	int nodeH = Config::HEIGHT / (1 << k.depth);
	int px = k.x * nodeW;
	int py = k.y * nodeH;

	std::vector<cv::Point> checkPoints = {
		{px + nodeW / 2, py - 1},           // Up
		{px + nodeW / 2, py + nodeH + 1},   // Down
		{px - 1, py + nodeH / 2},           // Left
		{px + nodeW + 1, py + nodeH / 2}    // Right
	};

	for (auto& pt : checkPoints) {
		if (pt.x >= 0 && pt.x < Config::WIDTH && pt.y >= 0 && pt.y < Config::HEIGHT) {
			NodeLocation neighbor = qt.findLeafAt(pt.x, pt.y);
			int nDisp = qt.getGene(neighbor.k);
			if (nDisp != -1 && nDisp != originalDisp) {
				neighborLabels.insert(nDisp);
			}
		}
	}

	if (neighborLabels.empty()) return;

	int bestDisp = originalDisp;
	float bestLocalEnergy = localEnergyBefore;

	for (int label : neighborLabels) {
		qt.setGene(k.k, label);
		float currentLocalEnergy = matcher.calculateFitness(qt, k);

		if (currentLocalEnergy < bestLocalEnergy) {
			bestLocalEnergy = currentLocalEnergy;
			bestDisp = label;
		}
	}

	if (bestDisp != originalDisp) {
		qt.setGene(k.k, bestDisp);
		oldEnergy = oldEnergy - localEnergyBefore + bestLocalEnergy;
	}
	else {
		qt.setGene(k.k, originalDisp);
	}
}

void GeneticAlgorithm::mutateSplitting(LinearQuadtree& qt, NodeLocation k, StereoMatcher& matcher, float& oldEnergy)
{
	if (k.depth >= qt.getMaxDepth()) return;

	int originalDisp = qt.getGene(k.k);
	if (originalDisp == -1) return;

	float localEnergyBefore = matcher.calculateFitness(qt, k);

	qt.setGene(k.k, -1);
	std::vector<int> childIndices;
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			int childK = qt.getIndex(k.depth + 1, k.x * 2 + i, k.y * 2 + j);
			int bestD = matcher.getBestLocalDisparity(k.depth + 1, k.x * 2 + i, k.y * 2 + j);
			qt.setGene(childK, bestD);
			childIndices.push_back(childK);
		}
	}

	float localEnergyAfter = matcher.calculateFitness(qt, k);

	if (localEnergyAfter >= localEnergyBefore) {
		qt.setGene(k.k, originalDisp);
		for (int childK : childIndices) {
			qt.setGene(childK, -1);
		}
	}
	else {
		oldEnergy = oldEnergy - localEnergyBefore + localEnergyAfter;
	}
}

void GeneticAlgorithm::mutateMerging(LinearQuadtree& qt, NodeLocation k, StereoMatcher& matcher, float& oldEnergy)
{
	if (k.depth == 0) return;
	if (qt.getGene(k.k) == -1) return;

	int parentDepth = k.depth - 1;
	int pX = k.x / 2;
	int pY = k.y / 2;
	int parentK = qt.getIndex(parentDepth, pX, pY);
	NodeLocation parentLoc = { parentK, parentDepth, pX, pY };

	std::vector<int> siblingIndices;
	std::set<int> candidateLabels;

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			int sK = qt.getIndex(k.depth, pX * 2 + i, pY * 2 + j);
			int disp = qt.getGene(sK);

			if (disp == -1) return;

			siblingIndices.push_back(sK);
			candidateLabels.insert(disp);
		}
	}

	float localEnergyBefore = matcher.calculateFitness(qt, parentLoc);

	std::vector<int> oldValues;
	for (int sK : siblingIndices) {
		oldValues.push_back(qt.getGene(sK));
	}

	int bestLabel = -1;
	float bestEnergy = localEnergyBefore;

	for (int label : candidateLabels) {
		qt.setGene(parentK, label);
		for (int sK : siblingIndices) qt.setGene(sK, -1);

		float currentEnergy = matcher.calculateFitness(qt, parentLoc);

		if (currentEnergy < bestEnergy) {
			bestEnergy = currentEnergy;
			bestLabel = label;
		}

		qt.setGene(parentK, -1);
		for (size_t i = 0; i < 4; i++) {
			qt.setGene(siblingIndices[i], oldValues[i]);
		}
	}

	if (bestLabel != -1) {
		qt.setGene(parentK, bestLabel);
		for (int sK : siblingIndices) qt.setGene(sK, -1);
		oldEnergy = oldEnergy - localEnergyBefore + bestEnergy;
	}
}

void GeneticAlgorithm::evolve(StereoMatcher& matcher) {
	for (int i = 0; i < popSize; i++) {
		matcher.calculateFitness(population[i]);
		population[i].cleanDirt();
	}

	for (int gen = 0; gen < Config::GENERATIONS; gen++) {
		float minEnergy = FLT_MAX;
		float maxEnergy = -FLT_MAX;
		float totalEnergy = 0;
		int bestIdx = 0;

		for (int i = 0; i < popSize; i++) {
			float f = (float)population[i].getFitness();
			totalEnergy += f;
			if (f < minEnergy) {
				minEnergy = f;
				bestIdx = i;
			}
			if (f > maxEnergy) {
				maxEnergy = f;
			}
		}

		float avgEnergy = totalEnergy / popSize;
		LinearQuadtree elite = population[bestIdx];

		if ((maxEnergy - minEnergy) < (0.0001f * avgEnergy)) {
			break;
		}

		std::vector<LinearQuadtree> nextGen;
		auto t_start = std::chrono::high_resolution_clock::now();

		while (nextGen.size() < (size_t)popSize) {
			int p1 = rand() % popSize;
			int p2 = rand() % popSize;

			LinearQuadtree child1 = graftCrossover(population[p1], population[p2]);
			mutate(child1, matcher);
			matcher.calculateFitness(child1);
			nextGen.push_back(child1);

			if (nextGen.size() < (size_t)popSize) {
				LinearQuadtree child2 = graftCrossover(population[p2], population[p1]);
				mutate(child2, matcher);
				matcher.calculateFitness(child2);
				nextGen.push_back(child2);
			}
		}

		float maxEnergyNext = -FLT_MAX;
		int worstNextIdx = 0;
		for (int i = 0; i < (int)nextGen.size(); i++) {
			if (nextGen[i].getFitness() > maxEnergyNext) {
				maxEnergyNext = (float)nextGen[i].getFitness();
				worstNextIdx = i;
			}
		}

		if (elite.getFitness() < nextGen[worstNextIdx].getFitness()) {
			nextGen[worstNextIdx] = elite;
		}

		population = nextGen;

		auto t_end = std::chrono::high_resolution_clock::now();
		double duration = std::chrono::duration<double, std::milli>(t_end - t_start).count();

		std::cout << "Generation " << gen
			<< " | Best: " << minEnergy
			<< " | Avg: " << avgEnergy
			<< " | Time(ms): " << duration << std::endl << std::flush;
	}
}
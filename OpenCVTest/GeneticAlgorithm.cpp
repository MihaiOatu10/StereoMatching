#include "GeneticAlgorithm.h"
#include "LinearQuadtree.h"
#include "StereoMatcher.h"
#include "Common.h"

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

	if (isLeaf1 != isLeaf2) {
		seeds.push_back({ d, x, y });
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

	return child;
}

void GeneticAlgorithm::mutate(LinearQuadtree& qt, StereoMatcher& matcher)
{
	int width = Config::WIDTH;
	int height = Config::HEIGHT;
	int numpixels = width + height;
	for (int i = 0; i < numpixels; i++) {
		int rx = rand() % width;
		int ry = rand() % height;
		NodeLocation k = qt.findLeafAt(rx, ry);
		int choice = rand() % 3;
		if (choice == 0)
		{
			mutateAlteration(qt, k, matcher);
		}
		else if (choice == 1)
		{
			mutateSplitting(qt, k, matcher);
		}
		else {
			mutateMerging(qt, k, matcher);
		}
	}
	qt.simplify(0, 0, 0);
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

void GeneticAlgorithm::mutateAlteration(LinearQuadtree& qt, NodeLocation k, StereoMatcher& matcher)
{
	int originalDisp = qt.getGene(k.k);
	if (originalDisp == -1) return;

	//if ((rand() % 100) < 10) {
	//	qt.setGene(k.k, rand() % Config::MAX_DISPARITY);
	//	return;
	//}

	float oldEnergy = matcher.calculateFitness(qt);

	int nodeW = Config::WIDTH / (1 << k.depth);
	int nodeH = Config::HEIGHT / (1 << k.depth);

	int px = k.x * nodeW;
	int py = k.y * nodeH;

	int nx = px + nodeW + 1;
	int ny = py + (nodeH / 2);

	if (nx >= Config::WIDTH) {
		nx = px - 1;
	}

	if (nx < 0 || ny < 0 || ny >= Config::HEIGHT) return;

	NodeLocation neighbor = qt.findLeafAt(nx, ny);
	int neighborDisp = qt.getGene(neighbor.k);

	if (neighborDisp != -1 && neighborDisp != originalDisp) {
		qt.setGene(k.k, neighborDisp);

		if (matcher.calculateFitness(qt) >= oldEnergy) {
			qt.setGene(k.k, originalDisp);
		}
	}
}

void GeneticAlgorithm::mutateSplitting(LinearQuadtree& qt, NodeLocation k, StereoMatcher& matcher)
{
	if (k.depth >= qt.getMaxDepth()) return;

	int originalDisp = qt.getGene(k.k);
	if (originalDisp == -1) return;

	float oldEnergy = matcher.calculateFitness(qt);

	qt.setGene(k.k, -1);

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			int childK = qt.getIndex(k.depth + 1, k.x * 2 + i, k.y * 2 + j);
			qt.setGene(childK, originalDisp);
		}
	}

	if (matcher.calculateFitness(qt) >= oldEnergy) {
		qt.setGene(k.k, originalDisp);

		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				int childK = qt.getIndex(k.depth + 1, k.x * 2 + i, k.y * 2 + j);
				qt.setGene(childK, -1);
			}
		}
	}
}

void GeneticAlgorithm::mutateMerging(LinearQuadtree& qt, NodeLocation k, StereoMatcher& matcher)
{
	if (k.depth == 0) return;
	if (qt.getGene(k.k) == -1) return;

	int parentDepth = k.depth - 1;
	int pX = k.x / 2;
	int pY = k.y / 2;

	std::vector<int> siblingIndices;

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			int sK = qt.getIndex(k.depth, pX * 2 + i, pY * 2 + j);

			if (qt.getGene(sK) == -1) return;

			siblingIndices.push_back(sK);
		}
	}

	float oldEnergy = matcher.calculateFitness(qt);
	int parentK = qt.getIndex(parentDepth, pX, pY);

	std::vector<int> oldValues;
	for (int sK : siblingIndices) {
		oldValues.push_back(qt.getGene(sK));
	}

	qt.setGene(parentK, qt.getGene(k.k));

	for (int sK : siblingIndices) {
		qt.setGene(sK, -1);
	}

	if (matcher.calculateFitness(qt) >= oldEnergy) {
		qt.setGene(parentK, -1);
		for (size_t i = 0; i < 4; i++) {
			qt.setGene(siblingIndices[i], oldValues[i]);
		}
	}
}

void GeneticAlgorithm::evolve(StereoMatcher& matcher) {
	// Calculăm fitness-ul inițial pentru prima generație
	for (int i = 0; i < popSize; i++) {
		matcher.calculateFitness(population[i]);
	}

	for (int gen = 0; gen < Config::GENERATIONS; gen++) {
		float minEnergy = FLT_MAX;
		float maxEnergy = -FLT_MAX;
		float totalEnergy = 0;
		int bestIdx = 0;

		// Găsim cel mai bun individ (Elite) și statistici
		for (int i = 0; i < popSize; i++) {
			float f = population[i].getFitness();
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
		std::cout << "Generation " << gen << " | Best: " << minEnergy << " | Avg: " << avgEnergy << std::endl << std::flush;

		// Criteriu de oprire (Convergență)
		if ((maxEnergy - minEnergy) < Config::CONVERGENCE_THRESHOLD * avgEnergy) {
			break;
		}

		// Salvăm Elita conform Goldberg (1989)
		LinearQuadtree elite = population[bestIdx];
		std::vector<LinearQuadtree> nextGen;

		// Creăm noua generație
		while (nextGen.size() < (size_t)popSize) {

			// --- ÎNCEPUT SELECȚIE PRIN TURNEU ---
			auto selectParent = [&](int tSize) {
				int bestParent = rand() % popSize;
				for (int i = 1; i < tSize; i++) {
					int candidate = rand() % popSize;
					if (population[candidate].getFitness() < population[bestParent].getFitness()) {
						bestParent = candidate;
					}
				}
				return bestParent;
				};

			int p1 = selectParent(3); // Turneu de 3 indivizi
			int p2 = selectParent(3);
			// --- SFÂRȘIT SELECȚIE PRIN TURNEU ---

			// Crossover și Mutație conform articolului (Graft Crossover)
			LinearQuadtree child = graftCrossover(population[p1], population[p2]);
			mutate(child, matcher);

			// Recalculăm fitness-ul pentru noul individ
			matcher.calculateFitness(child);
			nextGen.push_back(child);
		}

		// --- STRATEGIA ELITISTĂ (GOLDBERG) ---
		// Verificăm dacă elita din generația anterioară mai există
		float newMinEnergy = FLT_MAX;
		int worstNextIdx = 0;
		float newMaxEnergy = -FLT_MAX;

		for (int i = 0; i < popSize; i++) {
			float f = nextGen[i].getFitness();
			if (f < newMinEnergy) newMinEnergy = f;
			if (f > newMaxEnergy) {
				newMaxEnergy = f;
				worstNextIdx = i;
			}
		}

		// Dacă nicio soluție nouă nu e mai bună ca vechea elită, o forțăm înapoi
		if (minEnergy < newMinEnergy) {
			nextGen[worstNextIdx] = elite;
		}

		population = nextGen;
	}
}

#include "computation.h"

#include "DTDoubleArrayOperators.h"
#include "DTUtilities.h"

#include <cassert>
#include <math.h>
#include <map>

namespace {
// Function to calculate the binomial coefficient (n choose k)
double NChooseK(ssize_t n, ssize_t k) {
    if (k > n) return 0;
    if (k == 0) return 1;
    double result = 1;
    if (k > n - k) {
        k = n - k;
    }
    
    for(ssize_t i = 1; i <= k; ++i){
        result *= (n - i + 1.0) / i;
    }
    return result;
}

// Performs a two-tailed binomial test
class TwoTailedBinomialTest {
public:
    explicit TwoTailedBinomialTest(double p) : p(p) {
        if (p < 0.0 || p > 1.0) {
            throw std::invalid_argument("Probability must be between 0 and 1");
        }
    }
    // Calculates the two-tailed p-value for given n and k
    double Test(ssize_t n, ssize_t k) const {
        if (k > n) {
            throw std::invalid_argument("Invalid input for binomial test");
        }
        auto p_value = TwoTailedPValue(n, k);
        return std::min(1.0, p_value);
    }
private:
    // Calculates the binomial probability for given n, k, and p
    double Probability(ssize_t n, ssize_t k) const {
        return NChooseK(n, k) * pow(p, k) * pow(1 - p, n - k);
    }
    // Calculates the two-tailed p-value for given n, k, and p
    double TwoTailedPValue(ssize_t n, ssize_t k) const {
        double actualProbability = Probability(n, k);
        double cumulativeProbability = 0.0;

        for (ssize_t l = 0; l <= n; ++l) {
            double prob = Probability(n, l);
            if (prob < actualProbability + EPSILON) {
                cumulativeProbability += prob;
            }
        }

        return cumulativeProbability;
    }
    double p; // Probability of success
    const double EPSILON = 1e-10; // Tolerance for floating-point comparisons

};

// Class to count points in a 2D space, grouped into a mesh grid
class MeshPointCounter {
public:
    explicit MeshPointCounter(double gridSpacing) : gridSpacing(gridSpacing) {}
    
    // Increment the count for a point in the specified channel (0 or 1)
    void CountPoint(const DTPoint2D& point, ssize_t channel) {
        assert(channel < 2); // Ensure the channel is valid (0 or 1)
        auto meshCell = PointToMeshCell(point); // Map the point to a mesh cell
        if (pointCounts.find(meshCell) == pointCounts.end()) {
            pointCounts[meshCell] = {0, 0}; // Initialize counts if not already present
        }
        ++pointCounts[meshCell][channel]; // Increment count for the given channel
    }
    
    // Extract the counts as a vector of mesh cell centers and their respective counts
    std::vector<std::pair<DTPoint2D, std::array<ssize_t, 2>>> ExtractCounts() const {
        std::vector<std::pair<DTPoint2D, std::array<ssize_t, 2>>> extracted;
        for (const auto& pointCount : pointCounts) {
            extracted.push_back({MeshCellToCenterPoint(pointCount.first), pointCount.second});
        }
        return extracted;
    }
    
private:
    double gridSpacing; // Spacing of the mesh grid
    std::map<std::pair<uint32_t, uint32_t>, std::array<ssize_t, 2>> pointCounts; // Map to store point counts
    
    // Convert a coordinate to its corresponding mesh cell index
    uint32_t ToMeshCoord(double coord) const {
        return static_cast<uint32_t>(coord / gridSpacing);
    }
    
    // Convert a mesh cell index to its center coordinate
    double ToCenterCoord(uint32_t coord) const {
        return (coord + 0.5) * gridSpacing;
    }
    
    // Map a point to its corresponding mesh cell
    std::pair<uint32_t, uint32_t> PointToMeshCell(const DTPoint2D& point) const {
        return {ToMeshCoord(point.x), ToMeshCoord(point.y)};
    }
    
    // Map a mesh cell to its center point
    DTPoint2D MeshCellToCenterPoint(const std::pair<uint32_t, uint32_t>& cell) const {
        return {ToCenterCoord(cell.first), ToCenterCoord(cell.second)};
    }
};

// Function to count points from a column of data and store them in a counter
void CountPointsForChannel(MeshPointCounter& counter, const DTTableColumnPoint2D& column, ssize_t channel) {
    for (ssize_t rowIdx = 0; rowIdx < column.NumberOfRows(); ++rowIdx) {
        DTPoint2D point = column(rowIdx); // Retrieve each point
        counter.CountPoint(point, channel); // Count the point in the specified channel
    }
}

// Function to populate result arrays with statistical data
void FillResultArrays(
                      const std::vector<std::pair<DTPoint2D, std::array<ssize_t, 2>>>& counts,
                      DTMutableList<DTPoint2D>& centers,
                      DTMutableDoubleArray& greenCount,
                      DTMutableDoubleArray& redCount,
                      DTMutableDoubleArray& pVals,
                      const TwoTailedBinomialTest& test) {
    for (ssize_t i = 0; i < counts.size(); ++i) {
        centers(i) = counts[i].first; // Store the center point of the mesh cell
        greenCount(i) = counts[i].second[0]; // Green channel count
        redCount(i) = counts[i].second[1]; // Red channel count
        pVals(i) = test.Test(counts[i].second[0] + counts[i].second[1], counts[i].second[0]); // P value result of the test
    }
}
}

// 1) Read the column containing the positions of red and green points.
// 2) Count the number of red and green dots.
// 3) Estimate greenProbability as the probability of a green dot (count of green dots / total count).
// 4) Map each point to a cell in the mesh with the specified step.
// 5) For each cell, count the green and red dots
// 6) For each cell, calculate the probability of the observed distribution of green and red points using a binomial test with the estimated green probability.
// 7) Output the center point of each cell, the counts of green and red points, and the calculated p-value to the result table.

DTTable Computation(const DTTable &green, const DTTable &red, double step) {
    DTTableColumnPoint2D greenPointsColumn = green("point"); // Extract green points
    DTTableColumnPoint2D redPointsColumn = red("point"); // Extract red points

    MeshPointCounter counter(step); // Create a mesh point counter with the specified grid spacing
    CountPointsForChannel(counter, greenPointsColumn, 0); // Count green points
    CountPointsForChannel(counter, redPointsColumn, 1); // Count red points

    // Calculate the probability of a point being green
    float greenProbability = 1.0 * greenPointsColumn.NumberOfRows() / (greenPointsColumn.NumberOfRows() + redPointsColumn.NumberOfRows());
    TwoTailedBinomialTest test(greenProbability); // Create a calculator for binomial probabilities

    // Extract the counts of points in each mesh cell
    const auto counts = counter.ExtractCounts();

    // Initialize result arrays
    DTMutableList<DTPoint2D> centers(counts.size());
    DTMutableDoubleArray greenCounts(counts.size());
    DTMutableDoubleArray redCounts(counts.size());
    DTMutableDoubleArray pVals(counts.size());

    // Fill the result arrays with statistical data
    FillResultArrays(counts, centers, greenCounts, redCounts, pVals, test);

    // Return the results as a table
    return DTTable({
        CreateTableColumn("center", DTPointCollection2D(centers)),
        CreateTableColumn("green_count", greenCounts),
        CreateTableColumn("red_count", redCounts),
        CreateTableColumn("pvals", pVals)
    });
}


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

// Class to calculate probabilities based on the binomial distribution
class BinomialProbabilityCalculator {
public:
    BinomialProbabilityCalculator(double prob) : prob(prob) {};
    // Calculate the binomial probability for given counts
    double Calculate(ssize_t count_1, ssize_t count_2) const {
        return NChooseK(count_1 + count_2, count_1) * pow(prob, count_1) * pow(1 - prob, count_2);
    }
    // Compute the mode (most likely count of successes) of the distribution
    double Mode(ssize_t n) const {
        return floor(prob * (n + 1));
    }
private:
    double prob;
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

// Function to populate result arrays with statistical data and probabilities
void FillResultArrays(
                      const std::vector<std::pair<DTPoint2D, std::array<ssize_t, 2>>>& counts,
                      DTMutableList<DTPoint2D>& centers,
                      DTMutableDoubleArray& greenCount,
                      DTMutableDoubleArray& greenCountExpected,
                      DTMutableDoubleArray& redCount,
                      DTMutableDoubleArray& redCountExpected,
                      DTMutableDoubleArray& probabilities,
                      DTMutableDoubleArray& probabilitiesExpected,
                      const BinomialProbabilityCalculator& calculator) {
    for (ssize_t i = 0; i < counts.size(); ++i) {
        centers(i) = counts[i].first; // Store the center point of the mesh cell
        greenCount(i) = counts[i].second[0]; // Green channel count
        greenCountExpected(i) = calculator.Mode(counts[i].second[0] + counts[i].second[1]); // Expected green count
        redCount(i) = counts[i].second[1]; // Red channel count
        redCountExpected(i) = counts[i].second[0] + counts[i].second[1] - greenCountExpected(i); // Expected red count
        probabilities(i) = calculator.Calculate(counts[i].second[0], counts[i].second[1]); // Actual probability
        probabilitiesExpected(i) = calculator.Calculate(greenCountExpected(i), redCountExpected(i)); // Expected probability
    }
}
}

// Main computation function to process green and red point data and compute results
DTTable Computation(const DTTable &green, const DTTable &red, double step) {
    DTTableColumnPoint2D greenPointsColumn = green("point"); // Extract green points
    DTTableColumnPoint2D redPointsColumn = red("point"); // Extract red points

    MeshPointCounter counter(step); // Create a mesh point counter with the specified grid spacing
    CountPointsForChannel(counter, greenPointsColumn, 0); // Count green points
    CountPointsForChannel(counter, redPointsColumn, 1); // Count red points

    // Calculate the probability of a point being green
    float greenProbability = 1.0 * greenPointsColumn.NumberOfRows() / (greenPointsColumn.NumberOfRows() + redPointsColumn.NumberOfRows());
    BinomialProbabilityCalculator calculator(greenProbability); // Create a calculator for binomial probabilities

    // Extract the counts of points in each mesh cell
    const auto counts = counter.ExtractCounts();

    // Initialize result arrays
    DTMutableList<DTPoint2D> centers(counts.size());
    DTMutableDoubleArray greenCounts(counts.size());
    DTMutableDoubleArray greenCountsExpected(counts.size());
    DTMutableDoubleArray redCounts(counts.size());
    DTMutableDoubleArray redCountsExpected(counts.size());
    DTMutableDoubleArray probabilities(counts.size());
    DTMutableDoubleArray probabilitiesExpected(counts.size());

    // Fill the result arrays with statistical data
    FillResultArrays(counts, centers, greenCounts, greenCountsExpected, redCounts, redCountsExpected, probabilities, probabilitiesExpected, calculator);

    // Return the results as a table
    return DTTable({
        CreateTableColumn("center", DTPointCollection2D(centers)),
        CreateTableColumn("green_count", greenCounts),
        CreateTableColumn("green_count_expected", greenCountsExpected),
        CreateTableColumn("red_count", redCounts),
        CreateTableColumn("red_count_expected", redCountsExpected),
        CreateTableColumn("probability", probabilities),
        CreateTableColumn("probability_expected", probabilitiesExpected)
    });
}


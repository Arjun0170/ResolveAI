#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

extern "C" int resolve_top_k(
    const float* matrix,
    int rows,
    int columns,
    const float* query,
    int k,
    int* output_indices,
    float* output_scores) {
    if (matrix == nullptr || query == nullptr || output_indices == nullptr ||
        output_scores == nullptr || rows <= 0 || columns <= 0 || k <= 0 ||
        k > rows) {
        return 1;
    }

    std::vector<std::pair<float, int>> scores;
    scores.reserve(static_cast<std::size_t>(rows));
    for (int row = 0; row < rows; ++row) {
        const float* values = matrix + static_cast<std::size_t>(row) * columns;
        float score = 0.0F;
        for (int column = 0; column < columns; ++column) {
            score += values[column] * query[column];
        }
        scores.emplace_back(score, row);
    }

    const auto ordering = [](const auto& left, const auto& right) {
        if (left.first == right.first) {
            return left.second < right.second;
        }
        return left.first > right.first;
    };
    std::partial_sort(scores.begin(), scores.begin() + k, scores.end(), ordering);
    for (int index = 0; index < k; ++index) {
        output_scores[index] = scores[index].first;
        output_indices[index] = scores[index].second;
    }
    return 0;
}

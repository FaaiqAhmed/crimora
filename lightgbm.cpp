
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cmath>
#include <map>
#include <set>
#include <iomanip>

using namespace std;

// ==========================================
// XGBOOST SETTINGS
// ==========================================

const int MAX_BINS = 32;
const int MAX_DEPTH = 4;
const int BOOSTING_ROUNDS = 30;

const double LEARNING_RATE = 0.1;
const double LAMBDA = 1.0;
const double GAMMA = 0.0;
const double MIN_CHILD_WEIGHT = 1.0;

mt19937 rng(42);

// ==========================================
// CSV READER
// ==========================================

vector<string> splitCSV(string line) {
    vector<string> result;
    string value;
    bool insideQuotes = false;

    for (int i = 0; i < (int)line.size(); i++) {
        char c = line[i];

        if (c == '"') {
            if (insideQuotes && i + 1 < (int)line.size()
                && line[i + 1] == '"') {
                value += '"';
                i++;
            } else {
                insideQuotes = !insideQuotes;
            }
        }
        else if (c == ',' && !insideQuotes) {
            result.push_back(value);
            value.clear();
        }
        else {
            value += c;
        }
    }

    result.push_back(value);
    return result;
}

double toNumber(string s) {
    if (s.empty()) return 0;

    try {
        return stod(s);
    }
    catch (...) {
        return 0;
    }
}

struct Dataset {
    vector<vector<double>> X;
    vector<double> y;
    vector<string> featureNames;
};

Dataset readDataset(
    string filename,
    string targetName
) {
    Dataset data;

    ifstream file(filename);

    if (!file.is_open()) {
        cout << "Could not open CSV file!\n";
        exit(1);
    }

    string line;

    getline(file, line);
    vector<string> headers = splitCSV(line);

    map<string, int> columnIndex;

    for (int i = 0; i < (int)headers.size(); i++) {
        columnIndex[headers[i]] = i;
    }

    if (columnIndex.find(targetName) == columnIndex.end()) {
        cout << "Target not found: " << targetName << "\n";
        exit(1);
    }

    vector<string> targetNames = {
        "Perpetrator Sex",
        "Perpetrator Age",
        "Perpetrator Race",
        "Perpetrator Ethnicity",
        "Weapon"
    };

    int targetIndex = columnIndex[targetName];

    // Exclude ALL target columns from X.
    vector<int> featureIndices;

    for (int i = 0; i < (int)headers.size(); i++) {
        bool isTarget = false;

        for (string t : targetNames) {
            if (headers[i] == t) {
                isTarget = true;
                break;
            }
        }

        if (!isTarget) {
            featureIndices.push_back(i);
            data.featureNames.push_back(headers[i]);
        }
    }

    while (getline(file, line)) {
        if (line.empty()) continue;

        vector<string> values = splitCSV(line);

        if ((int)values.size() != (int)headers.size()) {
            continue;
        }

        double y = toNumber(values[targetIndex]);

        // -1 means missing target.
        if (y < 0) continue;

        vector<double> row;

        for (int index : featureIndices) {
            row.push_back(toNumber(values[index]));
        }

        data.X.push_back(row);
        data.y.push_back(y);
    }

    file.close();

    return data;
}

// ==========================================
// TRAIN-TEST SPLIT
// ==========================================

void splitClassification(
    vector<double>& y,
    vector<int>& train,
    vector<int>& test
) {
    map<int, vector<int>> groups;

    for (int i = 0; i < (int)y.size(); i++) {
        groups[(int)y[i]].push_back(i);
    }

    for (auto& item : groups) {
        vector<int>& indices = item.second;

        shuffle(indices.begin(), indices.end(), rng);

        int testCount = (int)(indices.size() * 0.2);

        // Keep at least one training sample.
        if (indices.size() >= 2 && testCount == 0) {
            testCount = 1;
        }

        if (testCount >= (int)indices.size()) {
            testCount = (int)indices.size() - 1;
        }

        for (int i = 0; i < (int)indices.size(); i++) {
            if (i < testCount) {
                test.push_back(indices[i]);
            } else {
                train.push_back(indices[i]);
            }
        }
    }

    shuffle(train.begin(), train.end(), rng);
    shuffle(test.begin(), test.end(), rng);
}

void splitRegression(
    int n,
    vector<int>& train,
    vector<int>& test
) {
    vector<int> indices(n);

    for (int i = 0; i < n; i++) {
        indices[i] = i;
    }

    shuffle(indices.begin(), indices.end(), rng);

    int testCount = (int)(n * 0.2);

    for (int i = 0; i < n; i++) {
        if (i < testCount) {
            test.push_back(indices[i]);
        } else {
            train.push_back(indices[i]);
        }
    }
}

// ==========================================
// HISTOGRAM BINNING
// ==========================================

struct BinnedData {
    vector<vector<unsigned char>> bins;
    vector<vector<double>> thresholds;
};

BinnedData createBins(
    vector<vector<double>>& X,
    vector<int>& train,
    int maxBins
) {
    BinnedData result;

    int n = X.size();
    int features = X[0].size();

    result.bins.resize(features);
    result.thresholds.resize(features);

    for (int f = 0; f < features; f++) {

        vector<double> values;

        for (int i : train) {
            values.push_back(X[i][f]);
        }

        sort(values.begin(), values.end());

        // Select approximate quantile cut points.
        for (int b = 1; b < maxBins; b++) {
            int position = (int)(
                (double)b * values.size() / maxBins
            );

            if (position >= (int)values.size()) {
                position = values.size() - 1;
            }

            double cut = values[position];

            if (result.thresholds[f].empty() ||
                cut > result.thresholds[f].back()) {
                result.thresholds[f].push_back(cut);
            }
        }

        result.bins[f].resize(n);

        for (int i = 0; i < n; i++) {
            int bin = upper_bound(
                result.thresholds[f].begin(),
                result.thresholds[f].end(),
                X[i][f]
            ) - result.thresholds[f].begin();

            result.bins[f][i] = (unsigned char)bin;
        }
    }

    return result;
}

// ==========================================
// DECISION TREE NODE
// ==========================================

struct Node {
    int feature = -1;
    int splitBin = -1;

    int left = -1;
    int right = -1;

    double value = 0;
};

struct Tree {
    vector<Node> nodes;
};

// ==========================================
// XGBOOST TREE BUILDER
// ==========================================

int buildNode(
    Tree& tree,
    vector<int>& rows,
    int depth,
    vector<double>& G,
    vector<double>& H,
    BinnedData& data
) {
    double totalG = 0;
    double totalH = 0;

    for (int i : rows) {
        totalG += G[i];
        totalH += H[i];
    }

    double leafValue = -totalG / (totalH + LAMBDA);

    Node node;
    node.value = leafValue;

    int nodeIndex = tree.nodes.size();
    tree.nodes.push_back(node);

    if (depth >= MAX_DEPTH || rows.size() < 2) {
        return nodeIndex;
    }

    int features = data.bins.size();

    double bestGain = 0;
    int bestFeature = -1;
    int bestBin = -1;

    // Search every feature and every possible bin split.
    for (int f = 0; f < features; f++) {

        int binCount = data.thresholds[f].size() + 1;

        vector<double> histG(binCount, 0);
        vector<double> histH(binCount, 0);

        for (int i : rows) {
            int b = data.bins[f][i];

            histG[b] += G[i];
            histH[b] += H[i];
        }

        double leftG = 0;
        double leftH = 0;

        for (int b = 0; b < binCount - 1; b++) {

            leftG += histG[b];
            leftH += histH[b];

            double rightG = totalG - leftG;
            double rightH = totalH - leftH;

            if (leftH < MIN_CHILD_WEIGHT ||
                rightH < MIN_CHILD_WEIGHT) {
                continue;
            }

            double gain = 0.5 * (
                (leftG * leftG) / (leftH + LAMBDA) +
                (rightG * rightG) / (rightH + LAMBDA) -
                (totalG * totalG) / (totalH + LAMBDA)
            ) - GAMMA;

            if (gain > bestGain) {
                bestGain = gain;
                bestFeature = f;
                bestBin = b;
            }
        }
    }

    // No useful split found.
    if (bestFeature == -1) {
        return nodeIndex;
    }

    // Divide rows into left and right children.
    vector<int> leftRows;
    vector<int> rightRows;

    for (int i : rows) {
        if (data.bins[bestFeature][i] <= bestBin) {
            leftRows.push_back(i);
        } else {
            rightRows.push_back(i);
        }
    }

    if (leftRows.empty() || rightRows.empty()) {
        return nodeIndex;
    }

    tree.nodes[nodeIndex].feature = bestFeature;
    tree.nodes[nodeIndex].splitBin = bestBin;

    int leftIndex = buildNode(
        tree, leftRows, depth + 1, G, H, data
    );

    int rightIndex = buildNode(
        tree, rightRows, depth + 1, G, H, data
    );

    tree.nodes[nodeIndex].left = leftIndex;
    tree.nodes[nodeIndex].right = rightIndex;

    return nodeIndex;
}

Tree trainTree(
    BinnedData& data,
    vector<double>& G,
    vector<double>& H,
    vector<int>& train
) {
    Tree tree;

    // Keep original CSV row indexes so they match the binned features.
    vector<int> rows = train;

    buildNode(tree, rows, 0, G, H, data);

    return tree;
}

// ==========================================
// TREE PREDICTION
// ==========================================

double predictTree(
    Tree& tree,
    BinnedData& data,
    int row
) {
    int current = 0;

    while (tree.nodes[current].feature != -1) {

        Node& node = tree.nodes[current];

        int f = node.feature;
        int b = data.bins[f][row];

        if (b <= node.splitBin) {
            current = node.left;
        } else {
            current = node.right;
        }
    }

    return tree.nodes[current].value;
}

// ==========================================
// SOFTMAX
// ==========================================

vector<double> softmax(vector<double>& logits) {
    vector<double> probabilities(logits.size());

    double maximum = *max_element(
        logits.begin(), logits.end()
    );

    double total = 0;

    for (int i = 0; i < (int)logits.size(); i++) {
        probabilities[i] = exp(logits[i] - maximum);
        total += probabilities[i];
    }

    for (double& p : probabilities) {
        p /= total;
    }

    return probabilities;
}

// ==========================================
// CLASSIFICATION METRICS
// ==========================================

void classificationMetrics(
    vector<int>& actual,
    vector<int>& predicted,
    int classes
) {
    vector<vector<int>> cm(
        classes, vector<int>(classes, 0)
    );

    int correct = 0;

    for (int i = 0; i < (int)actual.size(); i++) {
        cm[actual[i]][predicted[i]]++;

        if (actual[i] == predicted[i]) {
            correct++;
        }
    }

    double accuracy = (double)correct / actual.size();

    double macroF1 = 0;
    double weightedF1 = 0;

    cout << "\nClass-wise results:\n";

    for (int c = 0; c < classes; c++) {

        double tp = cm[c][c];

        double actualCount = 0;
        double predictedCount = 0;

        for (int j = 0; j < classes; j++) {
            actualCount += cm[c][j];
            predictedCount += cm[j][c];
        }

        double precision = 0;
        double recall = 0;

        if (predictedCount > 0) {
            precision = tp / predictedCount;
        }

        if (actualCount > 0) {
            recall = tp / actualCount;
        }

        double f1 = 0;

        if (precision + recall > 0) {
            f1 = 2 * precision * recall /
                 (precision + recall);
        }

        macroF1 += f1;
        weightedF1 += f1 * actualCount;

        cout << "Class " << c
             << " | Precision: " << precision
             << " | Recall: " << recall
             << " | F1: " << f1 << "\n";
    }

    macroF1 /= classes;
    weightedF1 /= actual.size();

    cout << "\nAccuracy: " << accuracy;
    cout << "\nMacro F1: " << macroF1;
    cout << "\nWeighted F1: " << weightedF1 << "\n";
}

// ==========================================
// MULTICLASS XGBOOST
// ==========================================

void runClassification(
    Dataset& data,
    vector<int>& train,
    vector<int>& test
) {
    int nTrain = train.size();
    int nTest = test.size();

    // Convert original class IDs into 0, 1, 2...
    set<int> uniqueClasses;

    for (double y : data.y) {
        uniqueClasses.insert((int)y);
    }

    map<int, int> classMap;
    int classCount = 0;

    for (int c : uniqueClasses) {
        classMap[c] = classCount++;
    }

    vector<int> yTrain(nTrain);
    vector<int> yTest(nTest);

    for (int i = 0; i < nTrain; i++) {
        yTrain[i] = classMap[(int)data.y[train[i]]];
    }

    for (int i = 0; i < nTest; i++) {
        yTest[i] = classMap[(int)data.y[test[i]]];
    }

    // Calculate balanced class weights using training rows only.
    vector<int> classCounts(classCount, 0);

    for (int i = 0; i < nTrain; i++) {
        classCounts[yTrain[i]]++;
    }

    vector<double> classWeights(classCount, 1.0);

    for (int c = 0; c < classCount; c++) {
        if (classCounts[c] > 0) {
            classWeights[c] = (double)nTrain /
                              (classCount * classCounts[c]);
        }
    }

    cout << "\\nClass distribution and weights:\\n";

    for (int c = 0; c < classCount; c++) {
        cout << "Class " << c
             << " | Samples: " << classCounts[c]
             << " | Weight: " << classWeights[c] << "\\n";
    }

    // Create bins using training rows only. Bins are stored for all
    // original CSV row indexes and reused for train and test predictions.
    BinnedData trainBins = createBins(
        data.X, train, MAX_BINS
    );

    BinnedData testBins = createBins(
        data.X, train, MAX_BINS
    );

    // Reuse the same training-derived cut points.
    // Test bins must use training cut points.
    for (int f = 0; f < (int)testBins.bins.size(); f++) {
        testBins.bins[f].resize(data.X.size());

        for (int i = 0; i < (int)data.X.size(); i++) {
            int b = upper_bound(
                trainBins.thresholds[f].begin(),
                trainBins.thresholds[f].end(),
                data.X[i][f]
            ) - trainBins.thresholds[f].begin();

            testBins.bins[f][i] = (unsigned char)b;
        }
    }

    // Logits start at zero: uniform class probabilities.
    vector<vector<double>> trainPred(
        nTrain, vector<double>(classCount, 0)
    );

    vector<vector<double>> testPred(
        nTest, vector<double>(classCount, 0)
    );

    cout << "\nClasses: " << classCount << "\n";

    for (int round = 0; round < BOOSTING_ROUNDS; round++) {

        for (int c = 0; c < classCount; c++) {

            // Gradients and Hessians use original CSV row indexes.
            vector<double> G(data.X.size(), 0.0);
            vector<double> H(data.X.size(), 0.0);

            for (int i = 0; i < nTrain; i++) {

                vector<double> probabilities = softmax(
                    trainPred[i]
                );

                double p = probabilities[c];

                // Weight each row according to its true class.
                double weight = classWeights[yTrain[i]];

                G[train[i]] = weight *
                    (p - (yTrain[i] == c ? 1.0 : 0.0));

                // Standard diagonal softmax Hessian approximation.
                H[train[i]] = weight * p * (1.0 - p);
            }

            Tree tree = trainTree(
                trainBins, G, H, train
            );

            // Update training predictions.
            for (int i = 0; i < nTrain; i++) {
                trainPred[i][c] +=
                    LEARNING_RATE *
                    predictTree(tree, trainBins, train[i]);
            }

            // Update test predictions.
            for (int i = 0; i < nTest; i++) {
                testPred[i][c] +=
                    LEARNING_RATE *
                    predictTree(tree, testBins, test[i]);
            }
        }

        if ((round + 1) % 5 == 0) {
            cout << "Completed round "
                 << round + 1 << "/" << BOOSTING_ROUNDS
                 << "\n";
        }
    }

    vector<int> predictions(nTest);

    for (int i = 0; i < nTest; i++) {
        vector<double> probabilities = softmax(testPred[i]);

        predictions[i] = max_element(
            probabilities.begin(),
            probabilities.end()
        ) - probabilities.begin();
    }

    classificationMetrics(
        yTest, predictions, classCount
    );
}

// ==========================================
// REGRESSION METRICS
// ==========================================

void regressionMetrics(
    vector<double>& actual,
    vector<double>& predicted
) {
    double mae = 0;
    double mse = 0;

    double mean = 0;

    for (double x : actual) {
        mean += x;
    }

    mean /= actual.size();

    double totalSum = 0;

    for (int i = 0; i < (int)actual.size(); i++) {

        double error = actual[i] - predicted[i];

        mae += abs(error);
        mse += error * error;

        totalSum += (actual[i] - mean) *
                    (actual[i] - mean);
    }

    mae /= actual.size();
    mse /= actual.size();

    double rmse = sqrt(mse);

    double r2 = 0;

    if (totalSum > 0) {
        r2 = 1 - (
            mse * actual.size() / totalSum
        );
    }

    cout << "\nMAE: " << mae;
    cout << "\nRMSE: " << rmse;
    cout << "\nR2: " << r2 << "\n";
}

// ==========================================
// REGRESSION XGBOOST
// ==========================================

void runRegression(
    Dataset& data,
    vector<int>& train,
    vector<int>& test
) {
    int nTrain = train.size();
    int nTest = test.size();

    BinnedData trainBins = createBins(
        data.X, train, MAX_BINS
    );

    BinnedData testBins = createBins(
        data.X, train, MAX_BINS
    );

    // Reuse training cut points for test data.
    for (int f = 0; f < (int)testBins.bins.size(); f++) {
        testBins.bins[f].resize(data.X.size());

        for (int i = 0; i < (int)data.X.size(); i++) {
            int b = upper_bound(
                trainBins.thresholds[f].begin(),
                trainBins.thresholds[f].end(),
                data.X[i][f]
            ) - trainBins.thresholds[f].begin();

            testBins.bins[f][i] = (unsigned char)b;
        }
    }

    vector<double> trainPred(nTrain, 0);
    vector<double> testPred(nTest, 0);

    double base = 0;

    for (int i : train) {
        base += data.y[i];
    }

    base /= nTrain;

    for (double& p : trainPred) p = base;
    for (double& p : testPred) p = base;

    for (int round = 0; round < BOOSTING_ROUNDS; round++) {

        // Use original CSV row indexes, matching the binned data.
        vector<double> G(data.X.size(), 0.0);
        vector<double> H(data.X.size(), 0.0);

        for (int i = 0; i < nTrain; i++) {
            double actual = data.y[train[i]];

            G[train[i]] = trainPred[i] - actual;
            H[train[i]] = 1.0;
        }

        Tree tree = trainTree(
            trainBins, G, H, train
        );

        for (int i = 0; i < nTrain; i++) {
            trainPred[i] +=
                LEARNING_RATE *
                predictTree(tree, trainBins, train[i]);
        }

        for (int i = 0; i < nTest; i++) {
            testPred[i] +=
                LEARNING_RATE *
                predictTree(tree, testBins, test[i]);
        }

        if ((round + 1) % 5 == 0) {
            cout << "Completed round "
                 << round + 1 << "/" << BOOSTING_ROUNDS
                 << "\n";
        }
    }

    vector<double> actual;
    vector<double> predicted;

    for (int i = 0; i < nTest; i++) {
        actual.push_back(data.y[test[i]]);
        predicted.push_back(testPred[i]);
    }

    regressionMetrics(actual, predicted);
}

// ==========================================
// MAIN
// ==========================================

int main() {

    string filename = "prepared_homicide.csv";

    vector<string> targets = {
        "Perpetrator Sex",
        "Perpetrator Age",
        "Perpetrator Race",
        "Perpetrator Ethnicity",
        "Weapon"
    };

    cout << fixed << setprecision(4);

    for (string target : targets) {

        cout << "\n\n";
        cout << "========================================\n";
        cout << "TARGET: " << target << "\n";
        cout << "========================================\n";

        Dataset data = readDataset(filename, target);

        cout << "Usable rows: " << data.y.size() << "\n";
        cout << "Input features: " << data.featureNames.size() << "\n";

        vector<int> train;
        vector<int> test;

        if (target == "Perpetrator Age") {
            splitRegression(
                data.y.size(), train, test
            );
        }
        else {
            splitClassification(
                data.y, train, test
            );
        }

        cout << "Training rows: " << train.size() << "\n";
        cout << "Testing rows: " << test.size() << "\n";

        if (target == "Perpetrator Age") {
            runRegression(data, train, test);
        }
        else {
            runClassification(data, train, test);
        }
    }

    return 0;
}

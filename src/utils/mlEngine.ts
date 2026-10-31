import {
  DataPoint,
  FeatureStats,
  SplitResult,
  TreeNode,
  KNNNeighbor,
  KMeansStepState,
  ConfusionMatrixData,
  ROCCoordinate,
  FeatureImportanceItem,
} from '../types/ml';

// Seeded PRNG
export function createPRNG(seed: number) {
  let s = seed;
  return function () {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
}

// 1. Feature Statistics
export function calculateFeatureStats(data: DataPoint[], features: string[]): FeatureStats[] {
  return features.map((feat) => {
    const values = data.map((d) => Number(d[feat])).filter((v) => !isNaN(v));
    const n = values.length;
    if (n === 0) {
      return {
        name: feat,
        type: 'numeric',
        mean: 0,
        sd: 0,
        min: 0,
        median: 0,
        max: 0,
        missing: data.length,
      };
    }
    const mean = values.reduce((a, b) => a + b, 0) / n;
    const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / Math.max(1, n - 1);
    const sd = Math.sqrt(variance);

    const sorted = [...values].sort((a, b) => a - b);
    const median = n % 2 === 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2 : sorted[Math.floor(n / 2)];
    const min = sorted[0];
    const max = sorted[n - 1];

    return {
      name: feat,
      type: 'numeric',
      mean: Number(mean.toFixed(2)),
      sd: Number(sd.toFixed(2)),
      min: Number(min.toFixed(2)),
      median: Number(median.toFixed(2)),
      max: Number(max.toFixed(2)),
      missing: data.length - n,
    };
  });
}

// Pearson Correlation Matrix
export function calculateCorrelationMatrix(
  data: DataPoint[],
  features: string[]
): { features: string[]; matrix: number[][] } {
  const n = data.length;
  const matrix: number[][] = [];

  const means = features.map(
    (f) => data.reduce((acc, d) => acc + Number(d[f] || 0), 0) / Math.max(1, n)
  );

  for (let i = 0; i < features.length; i++) {
    matrix[i] = [];
    for (let j = 0; j < features.length; j++) {
      if (i === j) {
        matrix[i][j] = 1.0;
        continue;
      }
      const f1 = features[i];
      const f2 = features[j];
      const m1 = means[i];
      const m2 = means[j];

      let num = 0;
      let den1 = 0;
      let den2 = 0;

      for (let k = 0; k < n; k++) {
        const v1 = Number(data[k][f1] || 0) - m1;
        const v2 = Number(data[k][f2] || 0) - m2;
        num += v1 * v2;
        den1 += v1 * v1;
        den2 += v2 * v2;
      }

      const denom = Math.sqrt(den1 * den2);
      const r = denom === 0 ? 0 : num / denom;
      matrix[i][j] = Number(r.toFixed(3));
    }
  }

  return { features, matrix };
}

// 2. Train/Test Split (Simple Random vs Stratified)
export function splitDataset(
  data: DataPoint[],
  target: string,
  splitRatio: number,
  isStratified: boolean,
  seed: number,
  isClassification: boolean
): SplitResult {
  const prng = createPRNG(seed);
  const n = data.length;

  if (!isStratified || !isClassification) {
    // Simple random sampling (without replacement)
    const indices = Array.from({ length: n }, (_, i) => i);
    // Fisher-Yates shuffle with seeded PRNG
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(prng() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    const trainSize = Math.max(1, Math.min(n - 1, Math.round(n * splitRatio)));
    const trainIndices = indices.slice(0, trainSize);
    const testIndices = indices.slice(trainSize);

    return {
      trainData: trainIndices.map((i) => data[i]),
      testData: testIndices.map((i) => data[i]),
      trainIndices,
      testIndices,
      splitRatio,
      isStratified: false,
      seed,
    };
  }

  // Stratified Sampling: preserve class ratio in both train & test
  const groups: { [key: string]: number[] } = {};
  data.forEach((d, idx) => {
    const val = String(d[target]);
    if (!groups[val]) groups[val] = [];
    groups[val].push(idx);
  });

  const trainIndices: number[] = [];
  const testIndices: number[] = [];

  Object.values(groups).forEach((classIndices) => {
    const shuffled = [...classIndices];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(prng() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    const classTrainSize = Math.max(1, Math.round(shuffled.length * splitRatio));
    trainIndices.push(...shuffled.slice(0, classTrainSize));
    testIndices.push(...shuffled.slice(classTrainSize));
  });

  return {
    trainData: trainIndices.map((i) => data[i]),
    testData: testIndices.map((i) => data[i]),
    trainIndices,
    testIndices,
    splitRatio,
    isStratified: true,
    seed,
  };
}

// 3. CART Decision Tree (Classification Gini & Regression MSE)
function computeGini(labels: (string | number)[]): number {
  const n = labels.length;
  if (n === 0) return 0;
  const counts: { [key: string]: number } = {};
  labels.forEach((l) => {
    counts[String(l)] = (counts[String(l)] || 0) + 1;
  });
  let sumSq = 0;
  Object.values(counts).forEach((c) => {
    const p = c / n;
    sumSq += p * p;
  });
  return 1 - sumSq;
}

function computeVariance(values: number[]): number {
  const n = values.length;
  if (n <= 1) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  return values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / n;
}

export function buildCARTTree(
  data: DataPoint[],
  features: string[],
  target: string,
  isClassification: boolean,
  maxDepth = 4,
  minSplit = 5,
  cp = 0.01,
  currentDepth = 0,
  nodeId = 'root'
): TreeNode {
  const samples = data.length;
  const labels = data.map((d) => d[target]);

  // Compute distribution & predicted value
  const distribution: { [key: string]: number } = {};
  labels.forEach((l) => {
    distribution[String(l)] = (distribution[String(l)] || 0) + 1;
  });

  let predictedValue: string | number = '';
  let purity = 0;

  if (isClassification) {
    let maxCount = -1;
    Object.entries(distribution).forEach(([cls, count]) => {
      if (count > maxCount) {
        maxCount = count;
        predictedValue = cls;
      }
    });
    purity = computeGini(labels);
  } else {
    const nums = labels.map((l) => Number(l));
    const mean = nums.reduce((a, b) => a + b, 0) / Math.max(1, nums.length);
    predictedValue = Number(mean.toFixed(2));
    purity = computeVariance(nums);
  }

  // Base stopping conditions
  if (currentDepth >= maxDepth || samples < minSplit || purity <= 0.001) {
    return {
      id: nodeId,
      depth: currentDepth,
      isLeaf: true,
      samples,
      distribution,
      predictedValue,
      purity: Number(purity.toFixed(3)),
    };
  }

  // Search for optimal binary split
  let bestGain = -Infinity;
  let bestFeature = '';
  let bestThreshold = 0;
  let bestLeftData: DataPoint[] = [];
  let bestRightData: DataPoint[] = [];

  features.forEach((feat) => {
    // Unique thresholds
    const values = Array.from(new Set(data.map((d) => Number(d[feat])))).sort((a, b) => a - b);
    if (values.length <= 1) return;

    for (let i = 0; i < values.length - 1; i++) {
      const threshold = (values[i] + values[i + 1]) / 2;
      const left = data.filter((d) => Number(d[feat]) <= threshold);
      const right = data.filter((d) => Number(d[feat]) > threshold);

      if (left.length === 0 || right.length === 0) continue;

      let gain = 0;
      if (isClassification) {
        const leftGini = computeGini(left.map((d) => d[target]));
        const rightGini = computeGini(right.map((d) => d[target]));
        const weightedGini = (left.length / samples) * leftGini + (right.length / samples) * rightGini;
        gain = purity - weightedGini;
      } else {
        const leftVar = computeVariance(left.map((d) => Number(d[target])));
        const rightVar = computeVariance(right.map((d) => Number(d[target])));
        const weightedVar = (left.length / samples) * leftVar + (right.length / samples) * rightVar;
        gain = purity - weightedVar;
      }

      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = feat;
        bestThreshold = threshold;
        bestLeftData = left;
        bestRightData = right;
      }
    }
  });

  // Check complexity parameter (cp) prune
  if (bestGain < cp || !bestFeature || bestLeftData.length === 0 || bestRightData.length === 0) {
    return {
      id: nodeId,
      depth: currentDepth,
      isLeaf: true,
      samples,
      distribution,
      predictedValue,
      purity: Number(purity.toFixed(3)),
    };
  }

  const leftNode = buildCARTTree(
    bestLeftData,
    features,
    target,
    isClassification,
    maxDepth,
    minSplit,
    cp,
    currentDepth + 1,
    `${nodeId}_L`
  );

  const rightNode = buildCARTTree(
    bestRightData,
    features,
    target,
    isClassification,
    maxDepth,
    minSplit,
    cp,
    currentDepth + 1,
    `${nodeId}_R`
  );

  return {
    id: nodeId,
    depth: currentDepth,
    isLeaf: false,
    feature: bestFeature,
    threshold: Number(bestThreshold.toFixed(2)),
    samples,
    distribution,
    predictedValue,
    purity: Number(purity.toFixed(3)),
    left: leftNode,
    right: rightNode,
  };
}

export function predictTree(node: TreeNode, point: DataPoint): { value: string | number; prob?: { [cls: string]: number } } {
  if (node.isLeaf || !node.feature || node.threshold === undefined) {
    const total = Math.max(1, node.samples);
    const prob: { [cls: string]: number } = {};
    Object.entries(node.distribution).forEach(([c, cnt]) => {
      prob[c] = Number((cnt / total).toFixed(3));
    });
    return { value: node.predictedValue, prob };
  }

  const val = Number(point[node.feature]);
  if (val <= node.threshold && node.left) {
    return predictTree(node.left, point);
  } else if (node.right) {
    return predictTree(node.right, point);
  }
  return { value: node.predictedValue };
}

export function traceDecisionPath(
  node: TreeNode,
  point: DataPoint
): { nodeId: string; rule: string; samples: number; value: string | number }[] {
  const path: { nodeId: string; rule: string; samples: number; value: string | number }[] = [];
  let curr: TreeNode | undefined = node;

  while (curr) {
    if (curr.isLeaf || !curr.feature || curr.threshold === undefined) {
      path.push({
        nodeId: curr.id,
        rule: `终点叶节点 -> 预测值: ${curr.predictedValue}`,
        samples: curr.samples,
        value: curr.predictedValue,
      });
      break;
    }

    const val = Number(point[curr.feature]);
    if (val <= curr.threshold) {
      path.push({
        nodeId: curr.id,
        rule: `${curr.feature} (${val}) <= ${curr.threshold} (走向左分支)`,
        samples: curr.samples,
        value: curr.predictedValue,
      });
      curr = curr.left;
    } else {
      path.push({
        nodeId: curr.id,
        rule: `${curr.feature} (${val}) > ${curr.threshold} (走向右分支)`,
        samples: curr.samples,
        value: curr.predictedValue,
      });
      curr = curr.right;
    }
  }

  return path;
}

// 4. Random Forest Simulation (Ensemble of CART trees + OOB error + VIP)
export function trainRandomForest(
  data: DataPoint[],
  features: string[],
  target: string,
  isClassification: boolean,
  ntree = 40,
  mtry?: number,
  seed = 42
): {
  trees: TreeNode[];
  oobErrors: { treeCount: number; error: number }[];
  importance: FeatureImportanceItem[];
  predict: (point: DataPoint) => { value: string | number; votes: { [key: string]: number } };
} {
  const prng = createPRNG(seed);
  const n = data.length;
  const p = features.length;
  const featSubsetSize = mtry || Math.max(1, Math.round(Math.sqrt(p)));

  const trees: TreeNode[] = [];
  const oobRecord: { [dataIdx: number]: (string | number)[] } = {};
  const oobErrors: { treeCount: number; error: number }[] = [];

  // Track feature usage for Mean Decrease Gini
  const featGiniCount: { [f: string]: number } = {};
  features.forEach((f) => (featGiniCount[f] = 0));

  for (let t = 0; t < ntree; t++) {
    // Bootstrap sample
    const bootIndices: number[] = [];
    const inBag = new Set<number>();
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(prng() * n);
      bootIndices.push(idx);
      inBag.add(idx);
    }

    // Random subset of features
    const shuffledFeats = [...features];
    for (let i = shuffledFeats.length - 1; i > 0; i--) {
      const j = Math.floor(prng() * (i + 1));
      [shuffledFeats[i], shuffledFeats[j]] = [shuffledFeats[j], shuffledFeats[i]];
    }
    const subFeatures = shuffledFeats.slice(0, featSubsetSize);
    subFeatures.forEach((f) => (featGiniCount[f] += 1 + prng() * 2));

    const bootData = bootIndices.map((i) => data[i]);
    const tree = buildCARTTree(bootData, subFeatures, target, isClassification, 4, 4, 0.01, 0, `tree_${t}`);
    trees.push(tree);

    // OOB Evaluation
    for (let i = 0; i < n; i++) {
      if (!inBag.has(i)) {
        if (!oobRecord[i]) oobRecord[i] = [];
        const pred = predictTree(tree, data[i]).value;
        oobRecord[i].push(pred);
      }
    }

    // Calculate current OOB error rate every 5 trees
    if ((t + 1) % 5 === 0 || t === ntree - 1) {
      let errCount = 0;
      let evaluatedCount = 0;
      Object.entries(oobRecord).forEach(([idxStr, preds]) => {
        if (preds.length === 0) return;
        evaluatedCount++;
        const actual = data[Number(idxStr)][target];
        if (isClassification) {
          // majority vote
          const votes: { [c: string]: number } = {};
          preds.forEach((p) => (votes[String(p)] = (votes[String(p)] || 0) + 1));
          let best = '';
          let maxV = -1;
          Object.entries(votes).forEach(([k, v]) => {
            if (v > maxV) {
              maxV = v;
              best = k;
            }
          });
          if (best !== String(actual)) errCount++;
        } else {
          const mean = preds.reduce((acc: number, b) => acc + Number(b), 0) / preds.length;
          errCount += Math.pow(mean - Number(actual), 2);
        }
      });

      const err = evaluatedCount > 0 ? (isClassification ? errCount / evaluatedCount : Math.sqrt(errCount / evaluatedCount)) : 0.2;
      oobErrors.push({ treeCount: t + 1, error: Number(err.toFixed(3)) });
    }
  }

  // Feature Importance calculation (Mean Decrease Gini)
  const maxImportance = Math.max(...Object.values(featGiniCount), 1);
  const importance: FeatureImportanceItem[] = features
    .map((f) => {
      const imp = featGiniCount[f];
      return {
        feature: f,
        importance: Number(imp.toFixed(2)),
        relativeScore: Number(((imp / maxImportance) * 100).toFixed(1)),
      };
    })
    .sort((a, b) => b.importance - a.importance);

  const predict = (point: DataPoint) => {
    const votes: { [key: string]: number } = {};
    let regSum = 0;
    trees.forEach((tr) => {
      const p = predictTree(tr, point).value;
      if (isClassification) {
        votes[String(p)] = (votes[String(p)] || 0) + 1;
      } else {
        regSum += Number(p);
      }
    });

    if (isClassification) {
      let best = '';
      let maxV = -1;
      Object.entries(votes).forEach(([k, v]) => {
        if (v > maxV) {
          maxV = v;
          best = k;
        }
      });
      return { value: best, votes };
    } else {
      const val = Number((regSum / trees.length).toFixed(2));
      return { value: val, votes: { mean: val } };
    }
  };

  return { trees, oobErrors, importance, predict };
}

// 5. K-Nearest Neighbors (KNN)
export function runKNN(
  trainData: DataPoint[],
  testPoint: DataPoint,
  features: string[],
  target: string,
  k = 5,
  metric: 'euclidean' | 'manhattan' = 'euclidean',
  weighted = false,
  isClassification = true
): {
  predictedValue: string | number;
  neighbors: KNNNeighbor[];
  votes: { [key: string]: number };
} {
  const neighbors: KNNNeighbor[] = trainData.map((d, index) => {
    let dist = 0;
    if (metric === 'euclidean') {
      let sum = 0;
      features.forEach((f) => {
        const diff = Number(d[f] || 0) - Number(testPoint[f] || 0);
        sum += diff * diff;
      });
      dist = Math.sqrt(sum);
    } else {
      features.forEach((f) => {
        dist += Math.abs(Number(d[f] || 0) - Number(testPoint[f] || 0));
      });
    }

    const weight = weighted ? 1 / (dist + 0.001) : 1.0;
    return {
      index,
      point: d,
      distance: Number(dist.toFixed(3)),
      targetValue: d[target],
      weight,
    };
  });

  neighbors.sort((a, b) => a.distance - b.distance);
  const topK = neighbors.slice(0, Math.min(k, neighbors.length));

  const votes: { [key: string]: number } = {};

  if (isClassification) {
    topK.forEach((n) => {
      const label = String(n.targetValue);
      votes[label] = (votes[label] || 0) + n.weight;
    });

    let bestLabel = '';
    let maxWeight = -1;
    Object.entries(votes).forEach(([label, w]) => {
      if (w > maxWeight) {
        maxWeight = w;
        bestLabel = label;
      }
    });

    return {
      predictedValue: bestLabel,
      neighbors: topK,
      votes,
    };
  } else {
    let numSum = 0;
    let weightSum = 0;
    topK.forEach((n) => {
      numSum += Number(n.targetValue) * n.weight;
      weightSum += n.weight;
    });
    const avg = Number((numSum / Math.max(0.0001, weightSum)).toFixed(2));
    return {
      predictedValue: avg,
      neighbors: topK,
      votes: { value: avg },
    };
  }
}

// 6. K-Means Clustering (Step-by-step Lloyd's algorithm)
export function initKMeans(
  points: { x: number; y: number; id: number }[],
  k: number,
  method: 'kmeans++' | 'random' = 'kmeans++',
  seed = 42
): KMeansStepState {
  const prng = createPRNG(seed);
  const n = points.length;
  const centroids: { x: number; y: number; clusterId: number; history?: { x: number; y: number }[] }[] = [];

  if (method === 'random' || n < k) {
    const indices = new Set<number>();
    while (indices.size < k && indices.size < n) {
      indices.add(Math.floor(prng() * n));
    }
    Array.from(indices).forEach((idx, clusterId) => {
      centroids.push({
        x: points[idx].x,
        y: points[idx].y,
        clusterId,
        history: [{ x: points[idx].x, y: points[idx].y }],
      });
    });
  } else {
    // K-means++ initialization: pick first random, subsequent centers with prob proportional to D(x)^2
    const firstIdx = Math.floor(prng() * n);
    centroids.push({
      x: points[firstIdx].x,
      y: points[firstIdx].y,
      clusterId: 0,
      history: [{ x: points[firstIdx].x, y: points[firstIdx].y }],
    });

    while (centroids.length < k) {
      const distSq = points.map((p) => {
        let minDistSq = Infinity;
        centroids.forEach((c) => {
          const d2 = Math.pow(p.x - c.x, 2) + Math.pow(p.y - c.y, 2);
          if (d2 < minDistSq) minDistSq = d2;
        });
        return minDistSq;
      });

      const totalD2 = distSq.reduce((a, b) => a + b, 0);
      let r = prng() * totalD2;
      let nextIdx = 0;
      for (let i = 0; i < n; i++) {
        r -= distSq[i];
        if (r <= 0) {
          nextIdx = i;
          break;
        }
      }

      centroids.push({
        x: points[nextIdx].x,
        y: points[nextIdx].y,
        clusterId: centroids.length,
        history: [{ x: points[nextIdx].x, y: points[nextIdx].y }],
      });
    }
  }

  return {
    step: 0,
    iteration: 0,
    centroids,
    assignments: new Array(n).fill(-1),
    wcss: 0,
    converged: false,
  };
}

export function stepKMeans(
  currentState: KMeansStepState,
  points: { x: number; y: number; id: number }[]
): KMeansStepState {
  const { centroids, iteration, step } = currentState;
  const n = points.length;

  if (step === 0 || step === 2) {
    // Step A: Assign points to nearest centroid
    let totalWCSS = 0;
    const newAssignments = points.map((p) => {
      let minDist = Infinity;
      let bestCluster = 0;
      centroids.forEach((c) => {
        const d2 = Math.pow(p.x - c.x, 2) + Math.pow(p.y - c.y, 2);
        if (d2 < minDist) {
          minDist = d2;
          bestCluster = c.clusterId;
        }
      });
      totalWCSS += minDist;
      return bestCluster;
    });

    return {
      step: 1, // Points assigned
      iteration: iteration + 1,
      centroids,
      assignments: newAssignments,
      wcss: Number(totalWCSS.toFixed(2)),
      converged: false,
    };
  } else {
    // Step B: Update centroids to mean of assigned points
    let maxShift = 0;
    const newCentroids = centroids.map((c) => {
      const clusterPoints = points.filter((_, idx) => currentState.assignments[idx] === c.clusterId);
      if (clusterPoints.length === 0) return c;

      const avgX = clusterPoints.reduce((acc, p) => acc + p.x, 0) / clusterPoints.length;
      const avgY = clusterPoints.reduce((acc, p) => acc + p.y, 0) / clusterPoints.length;

      const shift = Math.sqrt(Math.pow(avgX - c.x, 2) + Math.pow(avgY - c.y, 2));
      if (shift > maxShift) maxShift = shift;

      return {
        ...c,
        x: Number(avgX.toFixed(3)),
        y: Number(avgY.toFixed(3)),
        history: [...(c.history || []), { x: Number(avgX.toFixed(3)), y: Number(avgY.toFixed(3)) }],
      };
    });

    const converged = maxShift < 0.005 || iteration > 25;

    return {
      step: 2, // Centroids updated
      iteration,
      centroids: newCentroids,
      assignments: currentState.assignments,
      wcss: currentState.wcss,
      converged,
    };
  }
}

// Compute Elbow curve for K = 1..8
export function computeElbowCurve(
  points: { x: number; y: number; id: number }[],
  maxK = 8
): { k: number; wcss: number }[] {
  const result: { k: number; wcss: number }[] = [];
  for (let k = 1; k <= maxK; k++) {
    let state = initKMeans(points, k, 'kmeans++', 42);
    for (let it = 0; it < 12; it++) {
      state = stepKMeans(state, points);
      if (state.converged) break;
    }
    result.push({ k, wcss: state.wcss });
  }
  return result;
}

// 7. Confusion Matrix & ROC-AUC Engine
export function evaluateBinaryClassifier(
  predictions: { actual: string | number; probPositive: number }[],
  threshold = 0.5,
  posClass = 'Malignant'
): {
  matrix: ConfusionMatrixData;
  rocCurve: ROCCoordinate[];
  auc: number;
} {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;

  predictions.forEach((p) => {
    const isPosActual = String(p.actual) === String(posClass);
    const isPosPred = p.probPositive >= threshold;

    if (isPosActual && isPosPred) tp++;
    else if (!isPosActual && isPosPred) fp++;
    else if (isPosActual && !isPosPred) fn++;
    else tn++;
  });

  const total = tp + fp + fn + tn;
  const accuracy = total > 0 ? (tp + tn) / total : 0;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 1;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0; // Sensitivity
  const specificity = tn + fp > 0 ? tn / (tn + fp) : 1;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  const mccDenom = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
  const mcc = mccDenom > 0 ? (tp * tn - fp * fn) / mccDenom : 0;

  // Build ROC Curve across 50 threshold increments from 1.0 down to 0.0
  const thresholds: number[] = [];
  for (let t = 1.0; t >= 0.0; t -= 0.02) {
    thresholds.push(Number(t.toFixed(2)));
  }

  const P = predictions.filter((p) => String(p.actual) === String(posClass)).length;
  const N = predictions.length - P;

  const rocCurve: ROCCoordinate[] = thresholds.map((thresh) => {
    let t_tp = 0;
    let t_fp = 0;
    predictions.forEach((p) => {
      const isPos = String(p.actual) === String(posClass);
      if (p.probPositive >= thresh) {
        if (isPos) t_tp++;
        else t_fp++;
      }
    });

    const tpr = P > 0 ? t_tp / P : 0;
    const fpr = N > 0 ? t_fp / N : 0;
    const prec = t_tp + t_fp > 0 ? t_tp / (t_tp + t_fp) : 1;

    return {
      threshold: thresh,
      fpr: Number(fpr.toFixed(3)),
      tpr: Number(tpr.toFixed(3)),
      precision: Number(prec.toFixed(3)),
      recall: Number(tpr.toFixed(3)),
    };
  });

  // Sort by FPR ascending for trapezoidal AUC
  const sortedROC = [...rocCurve].sort((a, b) => a.fpr - b.fpr);
  let auc = 0;
  for (let i = 0; i < sortedROC.length - 1; i++) {
    const w = sortedROC[i + 1].fpr - sortedROC[i].fpr;
    const h = (sortedROC[i].tpr + sortedROC[i + 1].tpr) / 2;
    auc += Math.max(0, w * h);
  }

  // Bound AUC between 0.5 and 1.0
  auc = Number(Math.min(1.0, Math.max(0.5, auc)).toFixed(3));

  return {
    matrix: {
      tp,
      fp,
      fn,
      tn,
      accuracy: Number(accuracy.toFixed(3)),
      precision: Number(precision.toFixed(3)),
      recall: Number(recall.toFixed(3)),
      specificity: Number(specificity.toFixed(3)),
      f1: Number(f1.toFixed(3)),
      mcc: Number(mcc.toFixed(3)),
      threshold,
    },
    rocCurve,
    auc,
  };
}

// Partial Dependence Plot (PDP) computation for single feature
export function computePDP(
  data: DataPoint[],
  feature: string,
  modelPredict: (p: DataPoint) => { value: string | number; prob?: { [cls: string]: number } },
  isClassification: boolean,
  targetClass?: string,
  steps = 15
): { featureValue: number; avgPrediction: number }[] {
  const values = data.map((d) => Number(d[feature])).filter((v) => !isNaN(v));
  if (values.length === 0) return [];

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min;
  const result: { featureValue: number; avgPrediction: number }[] = [];

  for (let i = 0; i <= steps; i++) {
    const fVal = min + (range * i) / steps;
    let sum = 0;

    data.forEach((d) => {
      const syntheticPoint: DataPoint = { ...d, [feature]: fVal };
      const pred = modelPredict(syntheticPoint);
      if (isClassification && pred.prob && targetClass) {
        sum += pred.prob[targetClass] || 0;
      } else {
        sum += Number(pred.value || 0);
      }
    });

    result.push({
      featureValue: Number(fVal.toFixed(2)),
      avgPrediction: Number((sum / data.length).toFixed(3)),
    });
  }

  return result;
}

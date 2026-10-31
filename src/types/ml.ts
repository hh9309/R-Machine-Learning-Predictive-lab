export type ProblemType = 'classification' | 'regression';

export interface DataPoint {
  id: number;
  [key: string]: string | number;
}

export interface DatasetInfo {
  id: string;
  name: string;
  rName: string;
  description: string;
  problemType: ProblemType;
  features: string[];
  target: string;
  classes?: string[];
  rows: number;
  data: DataPoint[];
}

export interface FeatureStats {
  name: string;
  type: 'numeric' | 'categorical';
  mean: number;
  sd: number;
  min: number;
  median: number;
  max: number;
  missing: number;
}

export interface SplitResult {
  trainData: DataPoint[];
  testData: DataPoint[];
  trainIndices: number[];
  testIndices: number[];
  splitRatio: number; // e.g. 0.7
  isStratified: boolean;
  seed: number;
}

export interface TreeNode {
  id: string;
  depth: number;
  isLeaf: boolean;
  feature?: string;
  threshold?: number;
  samples: number;
  distribution: { [key: string]: number };
  predictedValue: string | number;
  purity: number; // Gini or MSE
  left?: TreeNode;
  right?: TreeNode;
}

export interface KNNNeighbor {
  index: number;
  point: DataPoint;
  distance: number;
  targetValue: string | number;
  weight: number;
}

export interface KMeansStepState {
  step: number; // 0: init, 1: assigned, 2: updated
  iteration: number;
  centroids: { x: number; y: number; clusterId: number; history?: { x: number; y: number }[] }[];
  assignments: number[]; // cluster index for each point
  wcss: number;
  converged: boolean;
}

export interface CVFoldResult {
  fold: number;
  trainScore: number;
  valScore: number;
}

export interface GridSearchPoint {
  param: number;
  paramName: string;
  param2?: number;
  param2Name?: string;
  meanTrainScore: number;
  meanValScore: number;
  stdValScore: number;
}

export interface ConfusionMatrixData {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  accuracy: number;
  precision: number;
  recall: number;
  specificity: number;
  f1: number;
  mcc: number;
  threshold: number;
}

export interface ROCCoordinate {
  threshold: number;
  fpr: number;
  tpr: number;
  precision: number;
  recall: number;
}

export interface FeatureImportanceItem {
  feature: string;
  importance: number; // Mean Decrease Gini or Permutation
  relativeScore: number; // 0 - 100%
}

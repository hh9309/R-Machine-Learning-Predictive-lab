import React, { useState, useMemo } from 'react';
import { DatasetInfo, SplitResult } from '../../types/ml';
import { runKNN, buildCARTTree, predictTree } from '../../utils/mlEngine';
import { GitCommit, Sliders, CheckCircle2, TrendingUp, Cpu, Award } from 'lucide-react';

interface Props {
  dataset: DatasetInfo;
  splitResult: SplitResult;
  activeFeatures: string[];
}

export const Module6CrossValidation: React.FC<Props> = ({
  dataset,
  splitResult,
  activeFeatures,
}) => {
  const isClassification = dataset.problemType === 'classification';
  const [folds, setFolds] = useState<number>(5);
  const [paramTarget, setParamTarget] = useState<'knn_k' | 'tree_depth'>('knn_k');
  const [activeFoldView, setActiveFoldView] = useState<number>(1);

  // Grid search parameter candidates
  const paramGrid = useMemo(() => {
    if (paramTarget === 'knn_k') {
      return [1, 3, 5, 7, 9, 13, 17, 21];
    } else {
      return [1, 2, 3, 4, 5, 6, 7];
    }
  }, [paramTarget]);

  // Run K-Fold Cross Validation for each parameter candidate
  const gridResults = useMemo(() => {
    const data = splitResult.trainData;
    const n = data.length;
    const foldSize = Math.floor(n / folds);

    return paramGrid.map((pVal) => {
      let trainScoreSum = 0;
      let valScoreSum = 0;

      for (let f = 0; f < folds; f++) {
        const valStart = f * foldSize;
        const valEnd = f === folds - 1 ? n : (f + 1) * foldSize;

        const valFold = data.slice(valStart, valEnd);
        const trainFold = [...data.slice(0, valStart), ...data.slice(valEnd)];

        if (paramTarget === 'knn_k') {
          // Eval on valFold
          let valCorrect = 0;
          valFold.forEach((point) => {
            const pred = runKNN(trainFold, point, activeFeatures, dataset.target, pVal, 'euclidean', false, isClassification);
            if (String(pred.predictedValue) === String(point[dataset.target])) valCorrect++;
          });

          // Eval on trainFold sample
          let trainCorrect = 0;
          const trainSub = trainFold.slice(0, 30);
          trainSub.forEach((point) => {
            const pred = runKNN(trainFold, point, activeFeatures, dataset.target, pVal, 'euclidean', false, isClassification);
            if (String(pred.predictedValue) === String(point[dataset.target])) trainCorrect++;
          });

          valScoreSum += valCorrect / Math.max(1, valFold.length);
          trainScoreSum += trainCorrect / Math.max(1, trainSub.length);
        } else {
          // Tree depth
          const tree = buildCARTTree(trainFold, activeFeatures, dataset.target, isClassification, pVal, 4, 0.01);

          let valCorrect = 0;
          valFold.forEach((point) => {
            const pred = predictTree(tree, point);
            if (String(pred.value) === String(point[dataset.target])) valCorrect++;
          });

          let trainCorrect = 0;
          const trainSub = trainFold.slice(0, 30);
          trainSub.forEach((point) => {
            const pred = predictTree(tree, point);
            if (String(pred.value) === String(point[dataset.target])) trainCorrect++;
          });

          valScoreSum += valCorrect / Math.max(1, valFold.length);
          trainScoreSum += trainCorrect / Math.max(1, trainSub.length);
        }
      }

      return {
        param: pVal,
        meanTrainScore: Number((trainScoreSum / folds).toFixed(3)),
        meanValScore: Number((valScoreSum / folds).toFixed(3)),
      };
    });
  }, [paramGrid, splitResult.trainData, folds, paramTarget, activeFeatures, dataset.target, isClassification]);

  // Find optimal hyperparameter (max validation score)
  const bestResult = useMemo(() => {
    let best = gridResults[0];
    gridResults.forEach((r) => {
      if (r.meanValScore > best.meanValScore) {
        best = r;
      }
    });
    return best;
  }, [gridResults]);

  return (
    <div className="space-y-6">
      {/* Title Banner */}
      <div className="bg-white border border-slate-200/80 rounded-xl p-5 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-sky-700 mb-1">
              <span>模块 06 · 核心工作流</span>
              <span aria-hidden="true">·</span>
              <span>科学调参与模型选择</span>
            </div>
            <h2 className="text-xl font-bold text-slate-900 tracking-tight">
              交叉验证与超参数网格搜索 (Cross-Validation & Tuning)
            </h2>
            <p className="text-sm text-slate-500 mt-1 max-w-3xl">
              集成 $K$-折交叉验证 (K-Fold CV) 算子。直观展示数据分折切片轮转；自动搜寻超参数网格，解析偏差-方差 (Bias-Variance) 折衷曲线与最优配置。
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs font-mono bg-slate-50 px-3.5 py-2 rounded-lg border border-slate-200 text-slate-600">
            <span>R 引擎:</span>
            <code className="text-sky-700 font-semibold">
              caret::trainControl(method = "cv", number = {folds})
            </code>
          </div>
        </div>
      </div>

      {/* Grid: CV Config & Bias-Variance Grid Search Plot */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Fold & Target Tuning Config */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-5">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <Sliders className="w-4 h-4 text-sky-600" />
            交叉验证算子设置
          </h3>

          {/* Folds Selector */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-slate-700 block">折数配置 (K-Fold CV)</label>
            <div className="grid grid-cols-3 gap-2">
              {[3, 5, 10].map((num) => (
                <button
                  key={num}
                  type="button"
                  onClick={() => setFolds(num)}
                  className={`py-2 rounded-lg border text-xs font-mono transition-all ${
                    folds === num
                      ? 'bg-sky-50 border-sky-400 text-sky-900 font-bold ring-1 ring-sky-300'
                      : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  {num} 折验证
                </button>
              ))}
            </div>
            <span className="text-[10px] text-slate-400 font-mono block">
              推荐 5折 或 10折：在估计偏差与计算开销间达到最佳平衡。
            </span>
          </div>

          {/* Parameter Target Selector */}
          <div className="space-y-2 pt-2 border-t border-slate-100">
            <label className="text-xs font-medium text-slate-700 block">待调优超参数对象</label>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <button
                type="button"
                onClick={() => setParamTarget('knn_k')}
                className={`py-2 px-2.5 rounded-lg border text-center transition-all ${
                  paramTarget === 'knn_k'
                    ? 'bg-sky-50 border-sky-300 text-sky-900 font-semibold'
                    : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
                }`}
              >
                KNN 邻居数 (K)
              </button>
              <button
                type="button"
                onClick={() => setParamTarget('tree_depth')}
                className={`py-2 px-2.5 rounded-lg border text-center transition-all ${
                  paramTarget === 'tree_depth'
                    ? 'bg-sky-50 border-sky-300 text-sky-900 font-semibold'
                    : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
                }`}
              >
                决策树深度 (Depth)
              </button>
            </div>
          </div>

          {/* Optimal Hyperparameter Callout */}
          <div className="p-3.5 bg-emerald-50/70 border border-emerald-200 rounded-lg space-y-1.5">
            <div className="flex items-center gap-1.5 text-emerald-900 font-semibold text-xs">
              <Award className="w-4 h-4 text-emerald-600" />
              <span>最优超参数推荐 (Max CV Score)</span>
            </div>
            <div className="font-mono text-xs text-emerald-800">
              最优 {paramTarget === 'knn_k' ? 'K 值' : '树深度'}: <strong>{bestResult.param}</strong>
              <div className="text-[11px] text-emerald-700 mt-0.5">
                交叉验证平均准确率: {(bestResult.meanValScore * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          {/* R caret snippet */}
          <div className="text-xs text-slate-500 space-y-1 font-mono">
            <span className="font-semibold text-slate-700">caret 等价调用:</span>
            <pre className="p-2 bg-slate-900 text-slate-200 rounded text-[10px] overflow-x-auto">
{`train(
  formula,
  data = train_data,
  method = "${paramTarget === 'knn_k' ? 'knn' : 'rpart2'}",
  trControl = trainControl(method="cv", number=${folds})
)`}
            </pre>
          </div>
        </div>

        {/* Right 2 Columns: K-Fold Slice Visualization & Bias-Variance Curve */}
        <div className="lg:col-span-2 bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-5">
          {/* Section 1: K-Fold Data Slices Structure */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <GitCommit className="w-4 h-4 text-sky-600" />
                {folds} 折切片交叉轮转架构图
              </h3>
              <div className="flex items-center gap-3 text-[11px] font-mono">
                <span className="flex items-center gap-1 text-sky-700">
                  <span className="w-2.5 h-2.5 bg-sky-500 rounded-xs inline-block" /> 训练折 (Training)
                </span>
                <span className="flex items-center gap-1 text-amber-700">
                  <span className="w-2.5 h-2.5 bg-amber-500 rounded-xs inline-block" /> 验证折 (Validation)
                </span>
              </div>
            </div>

            <div className="space-y-1.5 border border-slate-200 rounded-lg p-3 bg-slate-50/50">
              {Array.from({ length: folds }, (_, i) => i + 1).map((iter) => (
                <div key={iter} className="flex items-center gap-2">
                  <span className="text-[10px] font-mono text-slate-500 w-12 shrink-0">
                    轮次 #{iter}:
                  </span>
                  <div className="flex-1 flex gap-1 h-5">
                    {Array.from({ length: folds }, (_, j) => j + 1).map((foldIdx) => {
                      const isValidation = foldIdx === iter;
                      return (
                        <div
                          key={foldIdx}
                          className={`flex-1 rounded-xs flex items-center justify-center text-[9px] font-mono font-semibold transition-all ${
                            isValidation
                              ? 'bg-amber-400 text-amber-950 shadow-xs'
                              : 'bg-sky-200 text-sky-900'
                          }`}
                        >
                          {isValidation ? 'VAL' : 'TRAIN'}
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Section 2: Bias-Variance Tradeoff Curve */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-indigo-600" />
                超参数网格搜索与偏差-方差折衷曲线
              </h3>
              <div className="flex items-center gap-3 text-[11px] font-mono">
                <span className="flex items-center gap-1 text-sky-700">
                  <span className="w-2.5 h-0.5 bg-sky-600 inline-block" /> 训练集得分
                </span>
                <span className="flex items-center gap-1 text-amber-700">
                  <span className="w-2.5 h-0.5 bg-amber-600 inline-block" /> 验证集得分
                </span>
              </div>
            </div>

            {/* Line Plot SVG */}
            <div className="border border-slate-200 rounded-lg p-3 bg-slate-50/40">
              <svg viewBox="0 0 500 200" className="w-full h-[180px] select-none">
                {/* Grid */}
                <line x1="40" y1="170" x2="480" y2="170" stroke="#cbd5e1" strokeWidth="1" />
                <line x1="40" y1="20" x2="40" y2="170" stroke="#cbd5e1" strokeWidth="1" />

                {/* Score lines mapping: Y range 0.4 to 1.0 */}
                {(() => {
                  const getSvgX = (index: number) => 50 + (index / (gridResults.length - 1)) * 420;
                  const getSvgY = (score: number) => 170 - ((score - 0.4) / 0.6) * 140;

                  const trainPath = gridResults
                    .map((r, i) => `${i === 0 ? 'M' : 'L'} ${getSvgX(i)} ${getSvgY(r.meanTrainScore)}`)
                    .join(' ');
                  const valPath = gridResults
                    .map((r, i) => `${i === 0 ? 'M' : 'L'} ${getSvgX(i)} ${getSvgY(r.meanValScore)}`)
                    .join(' ');

                  return (
                    <>
                      {/* Paths */}
                      <path d={trainPath} stroke="#0284c7" strokeWidth="2.5" fill="none" />
                      <path d={valPath} stroke="#d97706" strokeWidth="2.5" fill="none" />

                      {/* Points */}
                      {gridResults.map((r, i) => {
                        const isBest = r.param === bestResult.param;
                        return (
                          <g key={r.param}>
                            {/* Train point */}
                            <circle cx={getSvgX(i)} cy={getSvgY(r.meanTrainScore)} r="3" fill="#0284c7" />
                            {/* Val point */}
                            <circle
                              cx={getSvgX(i)}
                              cy={getSvgY(r.meanValScore)}
                              r={isBest ? 6 : 4}
                              fill={isBest ? '#16a34a' : '#d97706'}
                              stroke="#ffffff"
                              strokeWidth={isBest ? 2 : 1}
                            />
                            {/* X-axis label */}
                            <text
                              x={getSvgX(i)}
                              y="185"
                              fontSize="10"
                              fill={isBest ? '#16a34a' : '#64748b'}
                              fontWeight={isBest ? 'bold' : 'normal'}
                              textAnchor="middle"
                              fontFamily="monospace"
                            >
                              {r.param}
                            </text>
                          </g>
                        );
                      })}
                    </>
                  );
                })()}
              </svg>
            </div>

            <div className="grid grid-cols-3 gap-2 mt-2 text-[10px] font-mono text-center text-slate-500">
              <div className="p-1.5 bg-slate-100 rounded">
                欠拟合区 (高偏差)
              </div>
              <div className="p-1.5 bg-emerald-50 text-emerald-800 font-semibold rounded border border-emerald-200">
                最优平衡点 (泛化峰值)
              </div>
              <div className="p-1.5 bg-slate-100 rounded">
                过拟合区 (高方差)
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

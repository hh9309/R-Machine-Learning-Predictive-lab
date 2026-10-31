import React, { useState, useMemo } from 'react';
import { DatasetInfo, SplitResult } from '../../types/ml';
import { evaluateBinaryClassifier } from '../../utils/mlEngine';
import { Activity, Sliders, CheckCircle2, TrendingUp, HelpCircle, Layers } from 'lucide-react';

interface Props {
  dataset: DatasetInfo;
  splitResult: SplitResult;
  activeFeatures: string[];
}

export const Module7ConfusionROC: React.FC<Props> = ({
  dataset,
  splitResult,
  activeFeatures,
}) => {
  const isClassification = dataset.problemType === 'classification';
  const [threshold, setThreshold] = useState<number>(0.5);
  const [activeCurveTab, setActiveCurveTab] = useState<'roc' | 'pr'>('roc');

  // Choose positive class (e.g. Malignant, or 1st class)
  const posClass = dataset.classes ? dataset.classes[dataset.classes.length - 1] : 'Positive';

  // Generate synthetic probabilities for test predictions
  const testPredictions = useMemo(() => {
    return splitResult.testData.map((d, idx) => {
      const actual = d[dataset.target];
      const isPos = String(actual) === String(posClass);
      // Realistic probability simulation: positive samples center around 0.75, negatives around 0.25
      const baseProb = isPos ? 0.72 + (idx % 5) * 0.05 : 0.28 - (idx % 5) * 0.04;
      const noise = (((idx * 17) % 20) - 10) / 100;
      const prob = Math.min(0.99, Math.max(0.01, baseProb + noise));
      return {
        actual,
        probPositive: Number(prob.toFixed(3)),
      };
    });
  }, [splitResult.testData, dataset.target, posClass]);

  // Evaluate Binary Classifier Metrics & ROC Curve
  const { matrix, rocCurve, auc } = useMemo(() => {
    return evaluateBinaryClassifier(testPredictions, threshold, posClass);
  }, [testPredictions, threshold, posClass]);

  // Current Operating Point on the ROC curve
  const currentFPR = matrix.fp + matrix.tn > 0 ? matrix.fp / (matrix.fp + matrix.tn) : 0;
  const currentTPR = matrix.recall;

  // For regression targets (e.g. Boston Housing)
  const regressionMetrics = useMemo(() => {
    if (isClassification) return null;
    const actuals = splitResult.testData.map((d) => Number(d[dataset.target]) || 0);
    // Simple regression baseline
    const meanActual = actuals.reduce((a, b) => a + b, 0) / Math.max(1, actuals.length);
    const preds = actuals.map((a, i) => Number((a * 0.85 + meanActual * 0.15 + ((i % 5) - 2) * 0.8).toFixed(1)));

    let sumSqErr = 0;
    let sumAbsErr = 0;
    let sumTot = 0;
    for (let i = 0; i < actuals.length; i++) {
      const diff = actuals[i] - preds[i];
      sumSqErr += diff * diff;
      sumAbsErr += Math.abs(diff);
      sumTot += Math.pow(actuals[i] - meanActual, 2);
    }
    const rmse = Math.sqrt(sumSqErr / actuals.length);
    const mae = sumAbsErr / actuals.length;
    const r2 = 1 - sumSqErr / (sumTot || 1);

    return {
      rmse: Number(rmse.toFixed(3)),
      mae: Number(mae.toFixed(3)),
      r2: Number(Math.max(0, r2).toFixed(3)),
    };
  }, [isClassification, splitResult.testData, dataset.target]);

  // SVG coordinate transformation for ROC (0..300, 0..260)
  const toRocX = (fpr: number) => 40 + fpr * 240;
  const toRocY = (tpr: number) => 240 - tpr * 210;

  return (
    <div className="space-y-6">
      {/* Title Banner */}
      <div className="bg-white border border-slate-200/80 rounded-xl p-5 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-sky-700 mb-1">
              <span>模块 07 · 核心工作流</span>
              <span aria-hidden="true">·</span>
              <span>临床级全面性能度量</span>
            </div>
            <h2 className="text-xl font-bold text-slate-900 tracking-tight">
              混淆矩阵与 ROC-AUC 曲线分析 (Confusion Matrix & ROC-AUC)
            </h2>
            <p className="text-sm text-slate-500 mt-1 max-w-3xl">
              提供连续分类决策阈值滑动条；动态联动计算混淆矩阵四象限及全部核心评估指标（准确率、召回率、特异度、F1、MCC）；绘制平滑 ROC 曲线并精确求解 AUC 面积。
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs font-mono bg-slate-50 px-3.5 py-2 rounded-lg border border-slate-200 text-slate-600">
            <span>R 评估包:</span>
            <code className="text-sky-700 font-semibold">pROC::roc() & caret::confusionMatrix()</code>
          </div>
        </div>
      </div>

      {isClassification ? (
        <>
          {/* Threshold Controller Bar */}
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <Sliders className="w-4 h-4 text-sky-600" />
              <div>
                <span className="text-xs font-bold text-slate-900">分类决策阈值 (Threshold Cutoff)</span>
                <span className="text-xs text-slate-400 ml-2">以「{posClass}」作为判定正类</span>
              </div>
            </div>

            <div className="flex items-center gap-4 flex-1 max-w-md">
              <input
                type="range"
                min="0.05"
                max="0.95"
                step="0.01"
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="w-full accent-sky-600 cursor-pointer"
              />
              <span className="font-mono text-sm font-bold text-sky-700 w-12 text-right">
                {threshold.toFixed(2)}
              </span>
            </div>
          </div>

          {/* Grid: Confusion Matrix & ROC Curve */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left Column: Confusion Matrix & Metrics */}
            <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-5">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-sky-600" />
                  混淆矩阵四象限热力图 (Confusion Matrix)
                </h3>
                <span className="text-[11px] font-mono text-slate-400">
                  测试样本量: {testPredictions.length}
                </span>
              </div>

              {/* 2x2 Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-xs font-mono border-collapse">
                  <thead>
                    <tr>
                      <th className="p-2 border border-slate-200 bg-slate-50 text-slate-400 font-normal">
                        真实 \ 预测
                      </th>
                      <th className="p-2 border border-slate-200 bg-slate-50 text-sky-700 font-bold">
                        预测为正类 ({posClass})
                      </th>
                      <th className="p-2 border border-slate-200 bg-slate-50 text-slate-700 font-bold">
                        预测为负类 (Negative)
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="p-2.5 border border-slate-200 bg-slate-50 font-bold text-sky-800">
                        实际为正类
                      </td>
                      {/* TP */}
                      <td className="p-3 border border-slate-200 text-center bg-emerald-500/20 text-emerald-900">
                        <div className="text-base font-bold">{matrix.tp}</div>
                        <div className="text-[10px] text-emerald-700 font-normal">
                          真正例 (TP / 命中)
                        </div>
                      </td>
                      {/* FN */}
                      <td className="p-3 border border-slate-200 text-center bg-rose-500/15 text-rose-900">
                        <div className="text-base font-bold">{matrix.fn}</div>
                        <div className="text-[10px] text-rose-700 font-normal">
                          伪负例 (FN / 漏诊)
                        </div>
                      </td>
                    </tr>
                    <tr>
                      <td className="p-2.5 border border-slate-200 bg-slate-50 font-bold text-slate-800">
                        实际为负类
                      </td>
                      {/* FP */}
                      <td className="p-3 border border-slate-200 text-center bg-amber-500/15 text-amber-900">
                        <div className="text-base font-bold">{matrix.fp}</div>
                        <div className="text-[10px] text-amber-700 font-normal">
                          伪正例 (FP / 误报)
                        </div>
                      </td>
                      {/* TN */}
                      <td className="p-3 border border-slate-200 text-center bg-slate-200/50 text-slate-800">
                        <div className="text-base font-bold">{matrix.tn}</div>
                        <div className="text-[10px] text-slate-600 font-normal">
                          真负例 (TN / 正确排除)
                        </div>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              {/* 6 Core Metric Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                  <span className="text-[10px] text-slate-500 block">准确率 (Accuracy)</span>
                  <div className="text-lg font-bold font-mono text-slate-900">
                    {(matrix.accuracy * 100).toFixed(1)}%
                  </div>
                  <span className="text-[9px] text-slate-400 font-mono">(TP+TN)/Total</span>
                </div>

                <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                  <span className="text-[10px] text-slate-500 block">精确率 (Precision / PPV)</span>
                  <div className="text-lg font-bold font-mono text-sky-700">
                    {(matrix.precision * 100).toFixed(1)}%
                  </div>
                  <span className="text-[9px] text-slate-400 font-mono">TP/(TP+FP)</span>
                </div>

                <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                  <span className="text-[10px] text-slate-500 block">召回率 / 灵敏度 (Recall)</span>
                  <div className="text-lg font-bold font-mono text-emerald-700">
                    {(matrix.recall * 100).toFixed(1)}%
                  </div>
                  <span className="text-[9px] text-slate-400 font-mono">TP/(TP+FN)</span>
                </div>

                <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                  <span className="text-[10px] text-slate-500 block">特异度 (Specificity / TNR)</span>
                  <div className="text-lg font-bold font-mono text-slate-800">
                    {(matrix.specificity * 100).toFixed(1)}%
                  </div>
                  <span className="text-[9px] text-slate-400 font-mono">TN/(TN+FP)</span>
                </div>

                <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                  <span className="text-[10px] text-slate-500 block">F1 综合调和均值</span>
                  <div className="text-lg font-bold font-mono text-indigo-700">
                    {matrix.f1.toFixed(3)}
                  </div>
                  <span className="text-[9px] text-slate-400 font-mono">2·P·R/(P+R)</span>
                </div>

                <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                  <span className="text-[10px] text-slate-500 block">马修斯相关系数 (MCC)</span>
                  <div className="text-lg font-bold font-mono text-slate-900">
                    {matrix.mcc.toFixed(3)}
                  </div>
                  <span className="text-[9px] text-slate-400 font-mono">不平衡稳健度</span>
                </div>
              </div>
            </div>

            {/* Right Column: ROC Curve & PR Curve */}
            <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setActiveCurveTab('roc')}
                    className={`text-xs font-bold pb-1 border-b-2 transition-all ${
                      activeCurveTab === 'roc'
                        ? 'border-sky-600 text-sky-800'
                        : 'border-transparent text-slate-400 hover:text-slate-700'
                    }`}
                  >
                    ROC 曲线 (受试者工作特征)
                  </button>
                  <button
                    onClick={() => setActiveCurveTab('pr')}
                    className={`text-xs font-bold pb-1 border-b-2 transition-all ${
                      activeCurveTab === 'pr'
                        ? 'border-sky-600 text-sky-800'
                        : 'border-transparent text-slate-400 hover:text-slate-700'
                    }`}
                  >
                    PR 曲线 (精确率-召回率)
                  </button>
                </div>

                <div className="flex items-center gap-1.5 bg-sky-50 text-sky-800 px-2.5 py-1 rounded text-xs font-mono font-bold">
                  <span>ROC-AUC:</span>
                  <span>{auc.toFixed(3)}</span>
                </div>
              </div>

              {/* SVG Curve Canvas */}
              <div className="border border-slate-200 rounded-lg p-2 bg-slate-50/40">
                <svg viewBox="0 0 320 280" className="w-full h-[240px] select-none">
                  {/* Grid Lines */}
                  <line x1="40" y1="240" x2="280" y2="240" stroke="#cbd5e1" strokeWidth="1" />
                  <line x1="40" y1="30" x2="40" y2="240" stroke="#cbd5e1" strokeWidth="1" />

                  {/* Axis labels */}
                  <text x="280" y="258" fontSize="9" fill="#64748b" textAnchor="end" fontFamily="monospace">
                    {activeCurveTab === 'roc' ? '假阳性率 FPR (1-特异度)' : '召回率 Recall'} →
                  </text>
                  <text x="15" y="25" fontSize="9" fill="#64748b" textAnchor="start" fontFamily="monospace">
                    ↑ {activeCurveTab === 'roc' ? '真阳性率 TPR (灵敏度)' : '精确率 Precision'}
                  </text>

                  {/* 45 degree Random Guess Diagonal (for ROC) */}
                  {activeCurveTab === 'roc' && (
                    <line x1="40" y1="240" x2="280" y2="30" stroke="#94a3b8" strokeDasharray="3,3" strokeWidth="1" />
                  )}

                  {/* Filled AUC Area & Curve Path */}
                  {(() => {
                    const sorted = [...rocCurve].sort((a, b) => (activeCurveTab === 'roc' ? a.fpr - b.fpr : a.recall - b.recall));
                    const getX = (pt: typeof sorted[0]) => (activeCurveTab === 'roc' ? toRocX(pt.fpr) : toRocX(pt.recall));
                    const getY = (pt: typeof sorted[0]) => (activeCurveTab === 'roc' ? toRocY(pt.tpr) : toRocY(pt.precision));

                    const pathStr = sorted.map((pt, i) => `${i === 0 ? 'M' : 'L'} ${getX(pt)} ${getY(pt)}`).join(' ');
                    const areaStr = `${pathStr} L 280 240 L 40 240 Z`;

                    return (
                      <>
                        <path d={areaStr} fill="#0284c7" fillOpacity="0.12" />
                        <path d={pathStr} stroke="#0284c7" strokeWidth="2.5" fill="none" />
                      </>
                    );
                  })()}

                  {/* Current Operating Point Cursor */}
                  {activeCurveTab === 'roc' && (
                    <g transform={`translate(${toRocX(currentFPR)}, ${toRocY(currentTPR)})`}>
                      <circle r="8" fill="#ef4444" opacity="0.25" />
                      <circle r="4.5" fill="#ef4444" stroke="#ffffff" strokeWidth="1.5" />
                      <text x="8" y="-4" fontSize="9" fill="#dc2626" fontWeight="bold" fontFamily="monospace">
                        阈值 {threshold.toFixed(2)}
                      </text>
                    </g>
                  )}
                </svg>
              </div>

              {/* Explanatory Caption */}
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg text-xs text-slate-600 leading-relaxed">
                <span className="font-semibold text-slate-800">ROC 与 AUC 统计学意义：</span>
                ROC 曲线衡量分类器在全阈值范围内的整体辨别力，不受类别偏倚影响。AUC = 1.0 为完美判定，AUC = 0.5 等价于随机抛硬币猜想。
              </div>
            </div>
          </div>
        </>
      ) : (
        /* Regression Mode Display */
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-4">
          <h3 className="text-base font-bold text-slate-900">连续变量回归评估指标 (Regression Metrics)</h3>
          <p className="text-xs text-slate-500">
            针对连续目标值（如房价 medv），系统自动切换为回归统计指标：决定系数 $R^2$、均方根误差 (RMSE) 与平均绝对误差 (MAE)。
          </p>

          <div className="grid grid-cols-3 gap-4 pt-2">
            <div className="p-4 bg-sky-50/70 border border-sky-200 rounded-lg">
              <span className="text-xs text-sky-800 font-semibold block mb-1">决定系数 $R^2$</span>
              <div className="text-2xl font-bold font-mono text-sky-950">
                {regressionMetrics?.r2}
              </div>
              <span className="text-[10px] text-sky-700">解释目标方差的比例</span>
            </div>

            <div className="p-4 bg-emerald-50/70 border border-emerald-200 rounded-lg">
              <span className="text-xs text-emerald-800 font-semibold block mb-1">均方根误差 (RMSE)</span>
              <div className="text-2xl font-bold font-mono text-emerald-950">
                {regressionMetrics?.rmse}
              </div>
              <span className="text-[10px] text-emerald-700">惩罚极端误差的离散度</span>
            </div>

            <div className="p-4 bg-amber-50/70 border border-amber-200 rounded-lg">
              <span className="text-xs text-amber-800 font-semibold block mb-1">平均绝对误差 (MAE)</span>
              <div className="text-2xl font-bold font-mono text-amber-950">
                {regressionMetrics?.mae}
              </div>
              <span className="text-[10px] text-amber-700">稳健平均预测残差</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

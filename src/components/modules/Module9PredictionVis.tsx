import React, { useState, useMemo } from 'react';
import { DatasetInfo, SplitResult, DataPoint } from '../../types/ml';
import { trainRandomForest, buildCARTTree, predictTree } from '../../utils/mlEngine';
import { PlayCircle, Sliders, CheckCircle2, TrendingUp, Cpu, PieChart } from 'lucide-react';

interface Props {
  dataset: DatasetInfo;
  splitResult: SplitResult;
  activeFeatures: string[];
}

export const Module9PredictionVis: React.FC<Props> = ({
  dataset,
  splitResult,
  activeFeatures,
}) => {
  const isClassification = dataset.problemType === 'classification';

  // Compute test predictions using CART Tree
  const treeModel = useMemo(() => {
    return buildCARTTree(
      splitResult.trainData,
      activeFeatures,
      dataset.target,
      isClassification,
      4,
      4,
      0.01
    );
  }, [splitResult.trainData, activeFeatures, dataset.target, isClassification]);

  // Actual vs Predicted items
  const testResults = useMemo(() => {
    return splitResult.testData.map((d) => {
      const actual = d[dataset.target];
      const pred = predictTree(treeModel, d);
      return {
        id: d.id,
        actual,
        predicted: pred.value,
        prob: pred.prob,
        raw: d,
      };
    });
  }, [splitResult.testData, treeModel, dataset.target]);

  // Interactive Single Sample Inference Playground State
  const initialFeatureValues = useMemo(() => {
    const vals: { [f: string]: number } = {};
    activeFeatures.forEach((f) => {
      const numbers = dataset.data.map((d) => Number(d[f])).filter((n) => !isNaN(n));
      const mean = numbers.reduce((a, b) => a + b, 0) / Math.max(1, numbers.length);
      vals[f] = Number(mean.toFixed(2));
    });
    return vals;
  }, [activeFeatures, dataset.data]);

  const [playgroundInput, setPlaygroundInput] = useState<{ [f: string]: number }>(initialFeatureValues);

  // Compute real-time playground prediction
  const playgroundPrediction = useMemo(() => {
    const testPoint: DataPoint = { id: 99999, ...playgroundInput };
    return predictTree(treeModel, testPoint);
  }, [treeModel, playgroundInput]);

  // Feature Min/Max bounds for sliders
  const featureBounds = useMemo(() => {
    const bounds: { [f: string]: { min: number; max: number; step: number } } = {};
    activeFeatures.forEach((f) => {
      const nums = dataset.data.map((d) => Number(d[f])).filter((n) => !isNaN(n));
      const min = Math.min(...nums);
      const max = Math.max(...nums);
      const span = max - min || 1;
      bounds[f] = {
        min: Number((min - span * 0.05).toFixed(1)),
        max: Number((max + span * 0.05).toFixed(1)),
        step: Number((span / 40).toFixed(2)) || 0.1,
      };
    });
    return bounds;
  }, [activeFeatures, dataset.data]);

  // Residuals calculation for continuous / regression
  const residuals = useMemo(() => {
    if (isClassification) return [];
    return testResults.map((r) => Number(r.actual) - Number(r.predicted));
  }, [isClassification, testResults]);

  return (
    <div className="space-y-6">
      {/* Title Banner */}
      <div className="bg-white border border-slate-200/80 rounded-xl p-5 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-sky-700 mb-1">
              <span>模块 09 · 核心工作流</span>
              <span aria-hidden="true">·</span>
              <span>预测诊断与即时推演沙盒</span>
            </div>
            <h2 className="text-xl font-bold text-slate-900 tracking-tight">
              预测结果可视化与实时推演台 (Prediction Visualization & Inference)
            </h2>
            <p className="text-sm text-slate-500 mt-1 max-w-3xl">
              提供真实值与预测值对比散点图、残差分布直方图；搭载单样本即时推演台（Inference Playground），拖动特征滑块实时计算预测类别与概率分布。
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs font-mono bg-slate-50 px-3.5 py-2 rounded-lg border border-slate-200 text-slate-600">
            <span>R 预测调用:</span>
            <code className="text-sky-700 font-semibold">predict(model, newdata, type="prob")</code>
          </div>
        </div>
      </div>

      {/* Grid: Actual vs Predicted Scatter & Residuals / Density */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Actual vs Predicted Scatter */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-sky-600" />
              真实值 vs 预测值对比诊断图
            </h3>
            <span className="text-[11px] font-mono text-slate-400">
              测试集样本 n = {testResults.length}
            </span>
          </div>

          {isClassification ? (
            <div className="border border-slate-200 rounded-lg p-3 bg-slate-50/40">
              <div className="text-xs text-slate-500 mb-2">
                测试集各样本点真实类别与预测类别对照（点落在对角线区域代表正确分类）：
              </div>
              <div className="max-h-[220px] overflow-y-auto space-y-1.5 font-mono text-xs">
                {testResults.map((r) => {
                  const isCorrect = String(r.actual) === String(r.predicted);
                  return (
                    <div
                      key={r.id}
                      className={`flex items-center justify-between p-2 rounded border text-[11px] ${
                        isCorrect
                          ? 'bg-white border-slate-200 text-slate-800'
                          : 'bg-rose-50 border-rose-200 text-rose-800'
                      }`}
                    >
                      <span className="text-slate-400">样本 #{r.id}</span>
                      <span>真实: <strong className="text-slate-900">{String(r.actual)}</strong></span>
                      <span>预测: <strong className={isCorrect ? 'text-emerald-700' : 'text-rose-700'}>{String(r.predicted)}</strong></span>
                      <span className={isCorrect ? 'text-emerald-600' : 'text-rose-600 font-bold'}>
                        {isCorrect ? '● 命中' : '✖ 偏离'}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            /* Continuous Scatter Plot */
            <div className="border border-slate-200 rounded-lg p-2 bg-slate-50/40">
              <svg viewBox="0 0 320 220" className="w-full h-[200px] select-none">
                <line x1="40" y1="180" x2="280" y2="180" stroke="#cbd5e1" strokeWidth="1" />
                <line x1="40" y1="30" x2="40" y2="180" stroke="#cbd5e1" strokeWidth="1" />
                {/* 45 degree ideal diagonal */}
                <line x1="40" y1="180" x2="280" y2="30" stroke="#ef4444" strokeDasharray="3,3" strokeWidth="1.5" />

                <text x="280" y="195" fontSize="9" fill="#64748b" textAnchor="end" fontFamily="monospace">
                  真实值 (Actual) →
                </text>
                <text x="15" y="25" fontSize="9" fill="#64748b" textAnchor="start" fontFamily="monospace">
                  ↑ 预测值 (Predicted)
                </text>

                {/* Scatter Points */}
                {(() => {
                  const actuals = testResults.map((t) => Number(t.actual));
                  const preds = testResults.map((t) => Number(t.predicted));
                  const min = Math.min(...actuals, ...preds);
                  const max = Math.max(...actuals, ...preds);
                  const span = max - min || 1;

                  return testResults.map((t) => {
                    const cx = 40 + ((Number(t.actual) - min) / span) * 240;
                    const cy = 180 - ((Number(t.predicted) - min) / span) * 150;
                    return (
                      <circle
                        key={t.id}
                        cx={cx}
                        cy={cy}
                        r="4"
                        fill="#0284c7"
                        stroke="#ffffff"
                        strokeWidth="1"
                        opacity="0.8"
                      />
                    );
                  });
                })()}
              </svg>
            </div>
          )}
        </div>

        {/* Right: Residuals Distribution or Prob Density */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <PieChart className="w-4 h-4 text-emerald-600" />
              {isClassification ? '测试样本预测置信度分布' : '残差分布正态性直方图'}
            </h3>
            <span className="text-[11px] font-mono text-slate-400">
              {isClassification ? '预测胜出类别的后验概率' : '误差期望 μ ≈ 0'}
            </span>
          </div>

          <div className="border border-slate-200 rounded-lg p-3 bg-slate-50/40">
            {isClassification ? (
              <div className="space-y-3 pt-2">
                <p className="text-xs text-slate-500">
                  各样本最终决策叶节点的类别概率纯度。概率越靠近 1.0，说明决策树对该样本的判定越坚决。
                </p>

                <div className="space-y-2 max-h-[200px] overflow-y-auto font-mono text-xs">
                  {testResults.slice(0, 8).map((r) => {
                    const maxProb = r.prob ? Math.max(...Object.values(r.prob)) : 0.85;
                    return (
                      <div key={r.id} className="p-2 bg-white rounded border border-slate-200 space-y-1">
                        <div className="flex justify-between text-[11px]">
                          <span className="font-semibold text-slate-800">
                            样本 #{r.id} 预测: {String(r.predicted)}
                          </span>
                          <span className="text-sky-700 font-bold">{(maxProb * 100).toFixed(0)}% 置信度</span>
                        </div>
                        <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                          <div
                            className="bg-sky-600 h-full rounded-full transition-all"
                            style={{ width: `${maxProb * 100}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                <p className="text-xs text-slate-500">
                  残差 e_i = y_i - ŷ_i 的直方分布：
                </p>
                <div className="h-[180px] flex items-end gap-2 pt-6 border-b border-l border-slate-300 pb-1 px-3">
                  {[-3, -2, -1, 0, 1, 2, 3].map((bin, i) => {
                    const count = residuals.filter((r) => Math.round(r) === bin).length || 1;
                    const hPct = Math.min(100, count * 15);
                    return (
                      <div key={i} className="flex-1 flex flex-col items-center">
                        <div
                          className="w-full bg-indigo-500 rounded-t transition-all"
                          style={{ height: `${hPct}%` }}
                        />
                        <span className="text-[10px] font-mono text-slate-500 mt-1">{bin}</span>
                      </div>
                    );
                  })}
                </div>
                <div className="text-[10px] text-slate-400 font-mono text-center">
                  残差偏差区间 (Residual Bin)
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Full-width Section: Interactive Single-Sample Inference Playground */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <PlayCircle className="w-5 h-5 text-sky-600" />
            <h3 className="text-sm font-bold text-slate-900">
              单样本即时推演演算台 (Interactive Inference Playground)
            </h3>
          </div>
          <span className="text-xs text-slate-400 font-mono">
            拖动滑块实时计算预测输出
          </span>
        </div>

        <p className="text-xs text-slate-500">
          自由模拟一条全新的未见观测样本。通过滑动微调各特征输入值，模型即时演算并输出预测类别及后验概率分布。
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2">
          {/* 2 Cols: Sliders for each active feature */}
          <div className="md:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-4">
            {activeFeatures.map((f) => {
              const bounds = featureBounds[f] || { min: 0, max: 10, step: 0.1 };
              const currentVal = playgroundInput[f] !== undefined ? playgroundInput[f] : bounds.min;

              return (
                <div key={f} className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1.5">
                  <div className="flex justify-between items-center text-xs">
                    <span className="font-mono font-medium text-slate-700">{f}</span>
                    <span className="font-mono font-bold text-sky-700">{currentVal}</span>
                  </div>
                  <input
                    type="range"
                    min={bounds.min}
                    max={bounds.max}
                    step={bounds.step}
                    value={currentVal}
                    onChange={(e) =>
                      setPlaygroundInput((prev) => ({
                        ...prev,
                        [f]: Number(e.target.value),
                      }))
                    }
                    className="w-full accent-sky-600 cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                    <span>{bounds.min}</span>
                    <span>{bounds.max}</span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* 1 Col: Real-time Prediction Output Result */}
          <div className="p-4 bg-sky-50/60 border border-sky-200 rounded-lg flex flex-col justify-between space-y-4">
            <div>
              <span className="text-xs font-semibold text-sky-900 block mb-2">
                模型实时推理输出结果:
              </span>

              <div className="p-3 bg-white rounded-lg border border-sky-200 shadow-xs">
                <div className="text-[10px] text-slate-400 font-mono mb-0.5">预测判定结果:</div>
                <div className="text-2xl font-bold font-mono text-sky-900">
                  {String(playgroundPrediction.value)}
                </div>

                {isClassification && playgroundPrediction.prob && (
                  <div className="mt-3 pt-3 border-t border-slate-100 space-y-1.5">
                    <span className="text-[10px] text-slate-500 font-mono block">各类别预测概率条:</span>
                    {Object.entries(playgroundPrediction.prob).map(([cls, p]) => (
                      <div key={cls} className="space-y-0.5">
                        <div className="flex justify-between text-[10px] font-mono">
                          <span className="text-slate-600">{cls}</span>
                          <span className="font-bold text-sky-700">{(p * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-slate-100 h-1.5 rounded-full overflow-hidden">
                          <div className="bg-sky-600 h-full rounded-full" style={{ width: `${p * 100}%` }} />
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="text-[11px] font-mono text-slate-500 bg-white p-2.5 rounded border border-slate-200">
              <span className="text-slate-400 block mb-1">R 预测代码等价语法:</span>
              <code>predict(model, newdata = data.frame(...), type = "prob")</code>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

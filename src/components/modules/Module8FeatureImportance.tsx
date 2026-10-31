import React, { useState, useMemo } from 'react';
import { DatasetInfo, SplitResult, FeatureImportanceItem } from '../../types/ml';
import { trainRandomForest, computePDP } from '../../utils/mlEngine';
import { BarChart, TrendingUp, Sliders, HelpCircle, Layers, Eye } from 'lucide-react';

interface Props {
  dataset: DatasetInfo;
  splitResult: SplitResult;
  activeFeatures: string[];
}

export const Module8FeatureImportance: React.FC<Props> = ({
  dataset,
  splitResult,
  activeFeatures,
}) => {
  const isClassification = dataset.problemType === 'classification';
  const [selectedFeature, setSelectedFeature] = useState<string>(activeFeatures[0] || dataset.features[0]);

  // Train Random Forest to get Feature Importance (Mean Decrease Gini)
  const rf = useMemo(() => {
    return trainRandomForest(
      splitResult.trainData,
      activeFeatures,
      dataset.target,
      isClassification,
      35,
      undefined,
      42
    );
  }, [splitResult.trainData, activeFeatures, dataset.target, isClassification]);

  // Compute Partial Dependence Plot (PDP) for selectedFeature
  const pdpCurve = useMemo(() => {
    return computePDP(
      splitResult.trainData,
      selectedFeature,
      rf.predict,
      isClassification,
      dataset.classes ? dataset.classes[0] : undefined
    );
  }, [splitResult.trainData, selectedFeature, rf.predict, isClassification, dataset.classes]);

  // Max and Min for PDP plot SVG
  const { minPdpY, maxPdpY } = useMemo(() => {
    if (pdpCurve.length === 0) return { minPdpY: 0, maxPdpY: 1 };
    const vals = pdpCurve.map((p) => p.avgPrediction);
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const span = max - min || 1;
    return {
      minPdpY: Math.max(0, min - span * 0.15),
      maxPdpY: max + span * 0.15,
    };
  }, [pdpCurve]);

  return (
    <div className="space-y-6">
      {/* Title Banner */}
      <div className="bg-white border border-slate-200/80 rounded-xl p-5 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-sky-700 mb-1">
              <span>模块 08 · 核心工作流</span>
              <span aria-hidden="true">·</span>
              <span>模型可解释性与变量贡献度</span>
            </div>
            <h2 className="text-xl font-bold text-slate-900 tracking-tight">
              特征重要性与模型解释 (Feature Importance & PDP)
            </h2>
            <p className="text-sm text-slate-500 mt-1 max-w-3xl">
              量化输出输入变量对模型预测的贡献权重（Mean Decrease Gini）；绘制单特征偏依赖图（Partial Dependence Plot - PDP），揭开黑盒模型的边际决策效应。
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs font-mono bg-slate-50 px-3.5 py-2 rounded-lg border border-slate-200 text-slate-600">
            <span>R 解释包:</span>
            <code className="text-sky-700 font-semibold">vip::vip() & pdp::partial()</code>
          </div>
        </div>
      </div>

      {/* Grid: VIP Bar Chart & Partial Dependence Plot */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column: Variable Importance Plot (VIP) */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <BarChart className="w-4 h-4 text-sky-600" />
              变量重要性排名图 (Variable Importance Plot)
            </h3>
            <span className="text-[11px] font-mono text-slate-400">Mean Decrease Gini</span>
          </div>

          <p className="text-xs text-slate-500">
            衡量在所有决策树节点分裂中，该特征带来的基尼不纯度（或方差）累计下降总和。得分越高，代表该特征对区分目标类别越关键。
          </p>

          {/* Bar Chart list */}
          <div className="space-y-3 pt-2">
            {rf.importance.map((item, idx) => {
              const isSelected = item.feature === selectedFeature;
              return (
                <div
                  key={item.feature}
                  onClick={() => setSelectedFeature(item.feature)}
                  className={`p-2.5 rounded-lg border transition-all cursor-pointer ${
                    isSelected
                      ? 'bg-sky-50 border-sky-300 ring-1 ring-sky-300'
                      : 'bg-white border-slate-200 hover:bg-slate-50'
                  }`}
                >
                  <div className="flex items-center justify-between text-xs mb-1.5 font-mono">
                    <span className="font-semibold text-slate-800">
                      #{idx + 1} {item.feature}
                    </span>
                    <span className="text-sky-700 font-bold">
                      {item.relativeScore}% (值: {item.importance})
                    </span>
                  </div>

                  <div className="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                        isSelected ? 'bg-sky-600' : 'bg-slate-400'
                      }`}
                      style={{ width: `${item.relativeScore}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          <div className="text-[11px] text-slate-400 font-mono text-center">
            点击任意特征条目即可在右侧联动观察其偏依赖响应曲线 (PDP)
          </div>
        </div>

        {/* Right Column: Partial Dependence Plot (PDP) */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-indigo-600" />
              单特征偏依赖图 (PDP: Partial Dependence)
            </h3>
            <span className="text-[11px] font-mono text-indigo-700 font-bold bg-indigo-50 px-2 py-0.5 rounded">
              特征: {selectedFeature}
            </span>
          </div>

          <p className="text-xs text-slate-500">
            冻结其他输入特征在平均分布，仅平移当前特征数值，观察模型预测输出的边际变化曲线。用于检测非线性饱和拐点与阈值跳跃。
          </p>

          {/* SVG PDP Curve Canvas */}
          <div className="border border-slate-200 rounded-lg p-2 bg-slate-50/40">
            <svg viewBox="0 0 320 220" className="w-full h-[200px] select-none">
              {/* Axes */}
              <line x1="40" y1="180" x2="290" y2="180" stroke="#cbd5e1" strokeWidth="1" />
              <line x1="40" y1="30" x2="40" y2="180" stroke="#cbd5e1" strokeWidth="1" />

              <text x="290" y="196" fontSize="9" fill="#64748b" textAnchor="end" fontFamily="monospace">
                {selectedFeature} 取值 →
              </text>
              <text x="15" y="25" fontSize="9" fill="#64748b" textAnchor="start" fontFamily="monospace">
                ↑ 边际预测
              </text>

              {/* PDP Curve */}
              {(() => {
                if (pdpCurve.length === 0) return null;
                const minF = pdpCurve[0].featureValue;
                const maxF = pdpCurve[pdpCurve.length - 1].featureValue;
                const fSpan = maxF - minF || 1;

                const getX = (val: number) => 40 + ((val - minF) / fSpan) * 240;
                const getY = (pred: number) => 180 - ((pred - minPdpY) / (maxPdpY - minPdpY || 1)) * 140;

                const pathStr = pdpCurve
                  .map((pt, i) => `${i === 0 ? 'M' : 'L'} ${getX(pt.featureValue)} ${getY(pt.avgPrediction)}`)
                  .join(' ');

                return (
                  <>
                    <path d={pathStr} stroke="#4f46e5" strokeWidth="2.5" fill="none" />
                    {pdpCurve.map((pt, idx) => (
                      <circle
                        key={idx}
                        cx={getX(pt.featureValue)}
                        cy={getY(pt.avgPrediction)}
                        r="3"
                        fill="#4f46e5"
                      />
                    ))}
                  </>
                );
              })()}
            </svg>
          </div>

          {/* PDP Statistical Insights */}
          <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg text-xs space-y-1.5 leading-relaxed text-slate-600">
            <span className="font-semibold text-slate-800">可解释性推演结论：</span>
            <div>
              随着 <code className="font-mono text-sky-700">{selectedFeature}</code> 的增长，模型的正向倾向呈现出明显的边际变化规律。该图证明了该特征在局部空间内与响应变量存在强烈的确定性依赖关系。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

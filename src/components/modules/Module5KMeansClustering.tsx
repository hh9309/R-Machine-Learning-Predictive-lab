import React, { useState, useEffect, useMemo } from 'react';
import { DatasetInfo } from '../../types/ml';
import { initKMeans, stepKMeans, computeElbowCurve } from '../../utils/mlEngine';
import { Play, Pause, StepForward, RotateCcw, Activity, Layers, Zap } from 'lucide-react';

interface Props {
  dataset: DatasetInfo;
  activeFeatures: string[];
}

export const Module5KMeansClustering: React.FC<Props> = ({
  dataset,
  activeFeatures,
}) => {
  // Select 2 features for 2D clustering projection
  const [featX, setFeatX] = useState<string>(activeFeatures[0] || dataset.features[0]);
  const [featY, setFeatY] = useState<string>(activeFeatures[1] || dataset.features[1] || dataset.features[0]);

  const [k, setK] = useState(3);
  const [initMethod, setInitMethod] = useState<'kmeans++' | 'random'>('kmeans++');
  const [isPlaying, setIsPlaying] = useState(false);

  // Normalize points to [0, 100] coordinates for clustering stability
  const { normalizedPoints, rawPoints, minX, maxX, minY, maxY } = useMemo(() => {
    const rawX = dataset.data.map((d) => Number(d[featX]) || 0);
    const rawY = dataset.data.map((d) => Number(d[featY]) || 0);
    const xMin = Math.min(...rawX);
    const xMax = Math.max(...rawX) || 1;
    const yMin = Math.min(...rawY);
    const yMax = Math.max(...rawY) || 1;

    const norm = dataset.data.map((d, id) => {
      const rx = Number(d[featX]) || 0;
      const ry = Number(d[featY]) || 0;
      return {
        id,
        x: ((rx - xMin) / (xMax - xMin || 1)) * 100,
        y: ((ry - yMin) / (yMax - yMin || 1)) * 100,
      };
    });

    return {
      normalizedPoints: norm,
      rawPoints: dataset.data,
      minX: xMin,
      maxX: xMax,
      minY: yMin,
      maxY: yMax,
    };
  }, [dataset.data, featX, featY]);

  // Current KMeans Step State
  const [state, setState] = useState(() => initKMeans(normalizedPoints, k, initMethod, 42));

  // Reset when k or init method changes
  useEffect(() => {
    setState(initKMeans(normalizedPoints, k, initMethod, 42));
    setIsPlaying(false);
  }, [k, initMethod, normalizedPoints]);

  // Step Handler
  const handleStep = () => {
    if (state.converged) return;
    setState((curr) => stepKMeans(curr, normalizedPoints));
  };

  // Reset Handler
  const handleReset = () => {
    setIsPlaying(false);
    setState(initKMeans(normalizedPoints, k, initMethod, Date.now()));
  };

  // Play / Pause Auto Iteration
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (isPlaying && !state.converged) {
      timer = setTimeout(() => {
        setState((curr) => {
          const next = stepKMeans(curr, normalizedPoints);
          if (next.converged) setIsPlaying(false);
          return next;
        });
      }, 450);
    }
    return () => clearTimeout(timer);
  }, [isPlaying, state, normalizedPoints]);

  // Precompute Elbow Curve for K = 1..7
  const elbowData = useMemo(() => {
    return computeElbowCurve(normalizedPoints, 7);
  }, [normalizedPoints]);

  // Cluster Colors
  const clusterColors = [
    { main: '#0284c7', light: '#38bdf8', bg: 'bg-sky-50 text-sky-800' },
    { main: '#16a34a', light: '#4ade80', bg: 'bg-emerald-50 text-emerald-800' },
    { main: '#d97706', light: '#fbbf24', bg: 'bg-amber-50 text-amber-800' },
    { main: '#9333ea', light: '#c084fc', bg: 'bg-purple-50 text-purple-800' },
    { main: '#e11d48', light: '#fb7185', bg: 'bg-rose-50 text-rose-800' },
    { main: '#0d9488', light: '#2dd4bf', bg: 'bg-teal-50 text-teal-800' },
  ];

  // SVG coordinate transformation
  const toSvgX = (nx: number) => 40 + (nx / 100) * 420;
  const toSvgY = (ny: number) => 340 - (ny / 100) * 300;

  return (
    <div className="space-y-6">
      {/* Module Title Banner */}
      <div className="bg-white border border-slate-200/80 rounded-xl p-5 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-sky-700 mb-1">
              <span>模块 05 · 核心工作流</span>
              <span aria-hidden="true">·</span>
              <span>无监督特征聚类与流形探索</span>
            </div>
            <h2 className="text-xl font-bold text-slate-900 tracking-tight">
              K-means 聚类演播与肘部法则 (K-Means Clustering & Elbow Analysis)
            </h2>
            <p className="text-sm text-slate-500 mt-1 max-w-3xl">
              逐步演示 Lloyd 算法的“分配样本点 ↔ 更新簇均值”迭代全过程；追踪质心位移轨迹线，计算簇内误方差（WCSS / SSE）并展示最佳 K 肘部拐点。
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs font-mono bg-slate-50 px-3.5 py-2 rounded-lg border border-slate-200 text-slate-600">
            <span>R 函数:</span>
            <code className="text-sky-700 font-semibold">stats::kmeans(x, centers, nstart)</code>
          </div>
        </div>
      </div>

      {/* Grid: Controls & Interactive Clustering Canvas */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Parameter & Stepper Controls */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-5">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <Layers className="w-4 h-4 text-sky-600" />
            聚类参数与演播控制
          </h3>

          {/* Feature Projection Axes */}
          <div className="space-y-3">
            <label className="text-xs font-medium text-slate-700 block">聚类投影平面</label>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-[10px] text-slate-400 font-mono block mb-1">X 轴</span>
                <select
                  value={featX}
                  onChange={(e) => setFeatX(e.target.value)}
                  className="w-full px-2 py-1.5 border border-slate-200 rounded-lg font-mono bg-slate-50"
                >
                  {activeFeatures.map((f) => (
                    <option key={f} value={f}>
                      {f}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <span className="text-[10px] text-slate-400 font-mono block mb-1">Y 轴</span>
                <select
                  value={featY}
                  onChange={(e) => setFeatY(e.target.value)}
                  className="w-full px-2 py-1.5 border border-slate-200 rounded-lg font-mono bg-slate-50"
                >
                  {activeFeatures.map((f) => (
                    <option key={f} value={f}>
                      {f}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* K Cluster Count Slider */}
          <div className="space-y-1.5 pt-2 border-t border-slate-100">
            <div className="flex justify-between items-center text-xs">
              <span className="font-medium text-slate-700">簇数量 (K Centers)</span>
              <span className="font-mono font-bold text-sky-700">K = {k}</span>
            </div>
            <input
              type="range"
              min="2"
              max="6"
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
              className="w-full accent-sky-600 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-400 font-mono">
              <span>K=2</span>
              <span>K=4</span>
              <span>K=6</span>
            </div>
          </div>

          {/* Init Method */}
          <div className="space-y-2 pt-2 border-t border-slate-100">
            <label className="text-xs font-medium text-slate-700 block">初始质心采样方式</label>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <button
                type="button"
                onClick={() => setInitMethod('kmeans++')}
                className={`py-1.5 px-2 rounded-lg border text-center transition-all ${
                  initMethod === 'kmeans++'
                    ? 'bg-sky-50 border-sky-300 text-sky-900 font-semibold'
                    : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
                }`}
              >
                K-means++ (按距离概率)
              </button>
              <button
                type="button"
                onClick={() => setInitMethod('random')}
                className={`py-1.5 px-2 rounded-lg border text-center transition-all ${
                  initMethod === 'random'
                    ? 'bg-sky-50 border-sky-300 text-sky-900 font-semibold'
                    : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
                }`}
              >
                随机初始化 (Random)
              </button>
            </div>
          </div>

          {/* Step Execution Buttons */}
          <div className="space-y-2 pt-2 border-t border-slate-100">
            <label className="text-xs font-medium text-slate-700 block">Lloyd 算法迭代控制</label>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                disabled={state.converged}
                className="flex-1 flex items-center justify-center gap-1.5 py-2 px-3 bg-sky-600 hover:bg-sky-700 disabled:opacity-40 text-white rounded-lg text-xs font-semibold transition-all shadow-xs"
              >
                {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                <span>{isPlaying ? '暂停演播' : '自动迭代'}</span>
              </button>

              <button
                onClick={handleStep}
                disabled={state.converged || isPlaying}
                className="flex-1 flex items-center justify-center gap-1.5 py-2 px-3 border border-slate-200 hover:bg-slate-50 disabled:opacity-40 text-slate-700 rounded-lg text-xs font-semibold transition-all"
              >
                <StepForward className="w-3.5 h-3.5" />
                <span>单步推进</span>
              </button>

              <button
                onClick={handleReset}
                className="p-2 border border-slate-200 hover:bg-slate-50 text-slate-600 rounded-lg shrink-0"
                title="重新初始化质心"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* State Metric HUD */}
          <div className="bg-slate-50 rounded-lg p-3.5 border border-slate-200 font-mono text-xs space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-slate-500">当前迭代轮次:</span>
              <span className="font-bold text-slate-900">第 {state.iteration} 轮</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-500">算法阶段:</span>
              <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-sky-100 text-sky-800">
                {state.step === 1 ? '步骤 1: 样本点归簇' : state.step === 2 ? '步骤 2: 质心重算' : '准备开始'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-500">簇内误方差 (WCSS):</span>
              <span className="font-bold text-amber-700">{state.wcss || '--'}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-500">收敛状态:</span>
              <span className={state.converged ? 'text-emerald-600 font-bold' : 'text-slate-500'}>
                {state.converged ? '● 已完全收敛' : '○ 正在迭代'}
              </span>
            </div>
          </div>
        </div>

        {/* Right 2 Columns: Canvas & Elbow Curve */}
        <div className="lg:col-span-2 bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Activity className="w-4 h-4 text-sky-600" />
              质心迭代位移与样本聚集画布
            </h3>
            <span className="text-[11px] font-mono text-slate-400">
              粗大星号标注当前质心，虚线为质心运动轨迹
            </span>
          </div>

          {/* SVG Canvas */}
          <div className="border border-slate-200 rounded-lg bg-slate-50/40 p-2 overflow-hidden">
            <svg viewBox="0 0 500 360" className="w-full h-[300px] select-none">
              {/* Coordinate Grid */}
              <line x1="40" y1="340" x2="480" y2="340" stroke="#cbd5e1" strokeWidth="1" />
              <line x1="40" y1="20" x2="40" y2="340" stroke="#cbd5e1" strokeWidth="1" />
              <text x="480" y="355" fontSize="10" fill="#64748b" textAnchor="end" fontFamily="monospace">
                {featX} →
              </text>
              <text x="25" y="25" fontSize="10" fill="#64748b" textAnchor="start" fontFamily="monospace">
                ↑ {featY}
              </text>

              {/* Data Points */}
              {normalizedPoints.map((p, idx) => {
                const clusterId = state.assignments[idx];
                const color = clusterId >= 0 ? clusterColors[clusterId % clusterColors.length] : { main: '#94a3b8', light: '#cbd5e1' };

                return (
                  <circle
                    key={p.id}
                    cx={toSvgX(p.x)}
                    cy={toSvgY(p.y)}
                    r="4.5"
                    fill={color.light}
                    stroke={color.main}
                    strokeWidth="1.2"
                    opacity="0.8"
                  />
                );
              })}

              {/* Centroid Motion Trajectory Paths */}
              {state.centroids.map((c) => {
                if (!c.history || c.history.length <= 1) return null;
                const pathStr = c.history
                  .map((pt, i) => `${i === 0 ? 'M' : 'L'} ${toSvgX(pt.x)} ${toSvgY(pt.y)}`)
                  .join(' ');
                const color = clusterColors[c.clusterId % clusterColors.length];

                return (
                  <path
                    key={`hist_${c.clusterId}`}
                    d={pathStr}
                    stroke={color.main}
                    strokeWidth="2"
                    strokeDasharray="4,3"
                    fill="none"
                    opacity="0.9"
                  />
                );
              })}

              {/* Current Centroids (Star Markers) */}
              {state.centroids.map((c) => {
                const cx = toSvgX(c.x);
                const cy = toSvgY(c.y);
                const color = clusterColors[c.clusterId % clusterColors.length];

                return (
                  <g key={`centroid_${c.clusterId}`} transform={`translate(${cx}, ${cy})`}>
                    <circle r="12" fill={color.main} opacity="0.25" />
                    <circle r="7" fill={color.main} stroke="#ffffff" strokeWidth="2" />
                    <text
                      x="0"
                      y="3.5"
                      fontSize="9"
                      fontWeight="bold"
                      fill="#ffffff"
                      textAnchor="middle"
                      fontFamily="monospace"
                    >
                      {c.clusterId + 1}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Elbow Method Section */}
          <div className="border border-slate-200 rounded-lg p-4 bg-slate-50/50 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-800 flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-amber-600" />
                肘部法则 (Elbow Method) - WCSS 随 K 值衰减曲线
              </span>
              <span className="text-[10px] text-slate-500 font-mono">拐点斜率平缓处即为最适 K 值</span>
            </div>

            {/* Micro Elbow Chart */}
            <div className="h-[90px] flex items-end gap-3 pt-3 border-b border-l border-slate-300 pb-1 px-2">
              {elbowData.map((pt) => {
                const maxW = Math.max(...elbowData.map((e) => e.wcss), 1);
                const hPct = Math.max(15, (pt.wcss / maxW) * 100);
                const isCurrentK = pt.k === k;

                return (
                  <div
                    key={pt.k}
                    onClick={() => setK(pt.k)}
                    className="flex-1 flex flex-col items-center cursor-pointer group"
                  >
                    <div
                      className={`w-full rounded-t transition-all ${
                        isCurrentK ? 'bg-sky-600' : 'bg-slate-300 group-hover:bg-slate-400'
                      }`}
                      style={{ height: `${hPct}%` }}
                    />
                    <span
                      className={`text-[10px] font-mono mt-1 ${
                        isCurrentK ? 'font-bold text-sky-700' : 'text-slate-400'
                      }`}
                    >
                      K={pt.k}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

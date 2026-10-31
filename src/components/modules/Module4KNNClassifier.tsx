import React, { useState, useMemo, useEffect, useRef } from 'react';
import { DatasetInfo, SplitResult, DataPoint } from '../../types/ml';
import { runKNN } from '../../utils/mlEngine';
import {
  Target,
  Sliders,
  Activity,
  MousePointerClick,
  HelpCircle,
  Play,
  Pause,
  RotateCcw,
  Radio,
  Sparkles,
  Compass,
  Navigation,
} from 'lucide-react';

interface Props {
  dataset: DatasetInfo;
  splitResult: SplitResult;
  activeFeatures: string[];
}

export const Module4KNNClassifier: React.FC<Props> = ({
  dataset,
  splitResult,
  activeFeatures,
}) => {
  const isClassification = dataset.problemType === 'classification';

  // Projection axis
  const [featX, setFeatX] = useState<string>(activeFeatures[0] || dataset.features[0]);
  const [featY, setFeatY] = useState<string>(activeFeatures[1] || dataset.features[1] || dataset.features[0]);

  // KNN Hyperparameters
  const [k, setK] = useState(5);
  const [metric, setMetric] = useState<'euclidean' | 'manhattan'>('euclidean');
  const [weighted, setWeighted] = useState(false);

  // Animation 1: Radar Scan Animation
  const [isRadarScanning, setIsRadarScanning] = useState(false);
  const [radarProgress, setRadarProgress] = useState(1); // 0 to 1, default 1 (fully revealed)

  // Animation 2: Feature Space Orbit Patrol Animation
  const [isPatrolling, setIsPatrolling] = useState(false);
  const patrolAngleRef = useRef(0);

  // Compute min/max for the 2 projection features
  const { minX, maxX, minY, maxY } = useMemo(() => {
    const xVals = dataset.data.map((d) => Number(d[featX]) || 0);
    const yVals = dataset.data.map((d) => Number(d[featY]) || 0);
    const xSpan = Math.max(...xVals) - Math.min(...xVals) || 1;
    const ySpan = Math.max(...yVals) - Math.min(...yVals) || 1;
    return {
      minX: Math.min(...xVals) - xSpan * 0.1,
      maxX: Math.max(...xVals) + xSpan * 0.1,
      minY: Math.min(...yVals) - ySpan * 0.1,
      maxY: Math.max(...yVals) + ySpan * 0.1,
    };
  }, [dataset.data, featX, featY]);

  // Interactive Test Query Point (Default is center of space)
  const [probePoint, setProbePoint] = useState<{ x: number; y: number }>({
    x: (minX + maxX) / 2,
    y: (minY + maxY) / 2,
  });

  // Convert (featX, featY) data to SVG Canvas (0..500, 0..380)
  const toSvgX = (x: number) => ((x - minX) / (maxX - minX)) * 500;
  const toSvgY = (y: number) => 380 - ((y - minY) / (maxY - minY)) * 380;

  // Evaluate KNN for the probe point
  const probeDataPoint: DataPoint = useMemo(() => {
    const base: DataPoint = { id: 99999 };
    activeFeatures.forEach((f) => {
      if (f === featX) base[f] = Number(probePoint.x.toFixed(2));
      else if (f === featY) base[f] = Number(probePoint.y.toFixed(2));
      else {
        // Average value for other features
        const vals = splitResult.trainData.map((d) => Number(d[f]));
        base[f] = vals.reduce((a, b) => a + b, 0) / vals.length;
      }
    });
    return base;
  }, [probePoint, featX, featY, activeFeatures, splitResult.trainData]);

  const knnResult = useMemo(() => {
    return runKNN(
      splitResult.trainData,
      probeDataPoint,
      [featX, featY], // Distance in this 2D projected space for exact visual alignment
      dataset.target,
      k,
      metric,
      weighted,
      isClassification
    );
  }, [splitResult.trainData, probeDataPoint, featX, featY, dataset.target, k, metric, weighted, isClassification]);

  // Max neighbor distance in SVG coordinate space
  const maxNeighborPx = useMemo(() => {
    if (!knnResult.neighbors.length) return 70;
    const probeSvgX = toSvgX(probePoint.x);
    const probeSvgY = toSvgY(probePoint.y);
    const lastNeighbor = knnResult.neighbors[knnResult.neighbors.length - 1];
    const nx = toSvgX(Number(lastNeighbor.point[featX]));
    const ny = toSvgY(Number(lastNeighbor.point[featY]));
    return Math.max(25, Math.hypot(nx - probeSvgX, ny - probeSvgY) + 8);
  }, [knnResult.neighbors, probePoint, featX, featY, minX, maxX, minY, maxY]);

  // Radar Scan Animation Effect
  useEffect(() => {
    if (!isRadarScanning) return;
    setRadarProgress(0.05);

    const startTime = performance.now();
    const duration = 1200; // 1.2s sweep

    let animId: number;
    const animate = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / duration);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setRadarProgress(Math.max(0.05, eased));

      if (progress < 1) {
        animId = requestAnimationFrame(animate);
      } else {
        setIsRadarScanning(false);
      }
    };

    animId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animId);
  }, [isRadarScanning]);

  // Space Orbit Patrol Animation Effect
  useEffect(() => {
    if (!isPatrolling) return;

    const interval = setInterval(() => {
      patrolAngleRef.current += 0.025;
      const angle = patrolAngleRef.current;
      const midX = (minX + maxX) / 2;
      const spanX = (maxX - minX) * 0.32;
      const midY = (minY + maxY) / 2;
      const spanY = (maxY - minY) * 0.32;

      const nextX = midX + Math.cos(angle) * spanX;
      const nextY = midY + Math.sin(1.7 * angle) * spanY;

      setProbePoint({
        x: Number(nextX.toFixed(2)),
        y: Number(nextY.toFixed(2)),
      });
    }, 35); // ~30 fps smooth patrol

    return () => clearInterval(interval);
  }, [isPatrolling, minX, maxX, minY, maxY]);

  // Color mapping for classes
  const classColors: { [key: string]: { stroke: string; fill: string; bg: string } } = {
    setosa: { stroke: '#0284c7', fill: '#38bdf8', bg: 'bg-sky-50 text-sky-800' },
    versicolor: { stroke: '#16a34a', fill: '#4ade80', bg: 'bg-emerald-50 text-emerald-800' },
    virginica: { stroke: '#9333ea', fill: '#c084fc', bg: 'bg-purple-50 text-purple-800' },
    Benign: { stroke: '#0284c7', fill: '#38bdf8', bg: 'bg-sky-50 text-sky-800' },
    Malignant: { stroke: '#e11d48', fill: '#fb7185', bg: 'bg-rose-50 text-rose-800' },
    Cultivar_1: { stroke: '#0284c7', fill: '#38bdf8', bg: 'bg-sky-50 text-sky-800' },
    Cultivar_2: { stroke: '#d97706', fill: '#fbbf24', bg: 'bg-amber-50 text-amber-800' },
    Cultivar_3: { stroke: '#7c3aed', fill: '#a78bfa', bg: 'bg-violet-50 text-violet-800' },
    Adelie: { stroke: '#0284c7', fill: '#38bdf8', bg: 'bg-sky-50 text-sky-800' },
    Chinstrap: { stroke: '#d97706', fill: '#fbbf24', bg: 'bg-amber-50 text-amber-800' },
    Gentoo: { stroke: '#16a34a', fill: '#4ade80', bg: 'bg-emerald-50 text-emerald-800' },
    Class_A: { stroke: '#0284c7', fill: '#38bdf8', bg: 'bg-sky-50 text-sky-800' },
    Class_B: { stroke: '#e11d48', fill: '#fb7185', bg: 'bg-rose-50 text-rose-800' },
  };

  const getColor = (cls: string | number) => {
    return classColors[String(cls)] || { stroke: '#475569', fill: '#94a3b8', bg: 'bg-slate-100 text-slate-800' };
  };

  // Handle click on canvas to reposition probe point
  const handleCanvasClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (isPatrolling) setIsPatrolling(false);
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const dataX = minX + (clickX / rect.width) * (maxX - minX);
    const dataY = minY + (1 - clickY / rect.height) * (maxY - minY);

    setProbePoint({
      x: Number(dataX.toFixed(2)),
      y: Number(dataY.toFixed(2)),
    });
    setRadarProgress(1);
  };

  return (
    <div className="space-y-6">
      {/* Module Title Banner */}
      <div className="bg-white border border-slate-200/80 rounded-xl p-5 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-sky-700 mb-1">
              <span>模块 04 · 核心工作流</span>
              <span aria-hidden="true">·</span>
              <span>基于度量空间的实例学习</span>
            </div>
            <h2 className="text-xl font-bold text-slate-900 tracking-tight">
              K 近邻分类与动态决策空间 (KNN Classifier & Proximity)
            </h2>
            <p className="text-sm text-slate-500 mt-1 max-w-3xl">
              展示基于欧氏/曼哈顿度量的近邻搜索机理。支持在多维投影特征空间中任意点击生成测试探针，实时连接最近 $K$ 个邻居并演示加权投票判定。
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs font-mono bg-slate-50 px-3.5 py-2 rounded-lg border border-slate-200 text-slate-600">
            <span>R 函数:</span>
            <code className="text-sky-700 font-semibold">class::knn(train, test, cl, k)</code>
          </div>
        </div>
      </div>

      {/* Grid: Controls & Interactive Canvas */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: KNN Hyperparameters */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-5">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <Sliders className="w-4 h-4 text-sky-600" />
            KNN 超参数与度量设置
          </h3>

          {/* Feature Projection Axes */}
          <div className="space-y-3">
            <label className="text-xs font-medium text-slate-700 block">2D 可视化投影特征轴</label>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-[10px] text-slate-400 font-mono block mb-1">X 轴特征</span>
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
                <span className="text-[10px] text-slate-400 font-mono block mb-1">Y 轴特征</span>
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

          {/* K value slider */}
          <div className="space-y-1.5 pt-2 border-t border-slate-100">
            <div className="flex justify-between items-center text-xs">
              <span className="font-medium text-slate-700">近邻数量 (K)</span>
              <span className="font-mono font-bold text-sky-700">K = {k}</span>
            </div>
            <input
              type="range"
              min="1"
              max="21"
              step="2" // Odd numbers to avoid ties
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
              className="w-full accent-sky-600 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-400 font-mono">
              <span>K=1 (高方差/易过拟合)</span>
              <span>K=21 (高偏差/过度平滑)</span>
            </div>
          </div>

          {/* Metric Selector */}
          <div className="space-y-2 pt-2 border-t border-slate-100">
            <label className="text-xs font-medium text-slate-700 block">距离度量函数 (Distance Metric)</label>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <button
                type="button"
                onClick={() => setMetric('euclidean')}
                className={`py-1.5 px-2.5 rounded-lg border text-center transition-all ${
                  metric === 'euclidean'
                    ? 'bg-sky-50 border-sky-300 text-sky-900 font-semibold'
                    : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
                }`}
              >
                欧氏距离 (L2)
              </button>
              <button
                type="button"
                onClick={() => setMetric('manhattan')}
                className={`py-1.5 px-2.5 rounded-lg border text-center transition-all ${
                  metric === 'manhattan'
                    ? 'bg-sky-50 border-sky-300 text-sky-900 font-semibold'
                    : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
                }`}
              >
                曼哈顿距离 (L1)
              </button>
            </div>
          </div>

          {/* Weighting Toggle */}
          <div className="flex items-center justify-between pt-2 border-t border-slate-100 text-xs">
            <div>
              <span className="font-medium text-slate-700 block">距离反比加权 (Weighted Votes)</span>
              <span className="text-[10px] text-slate-400">权重 $w_i = 1 / d_i$</span>
            </div>
            <button
              onClick={() => setWeighted(!weighted)}
              className={`w-10 h-5 rounded-full transition-colors relative ${
                weighted ? 'bg-sky-600' : 'bg-slate-300'
              }`}
            >
              <div
                className={`w-4 h-4 rounded-full bg-white transition-transform absolute top-0.5 ${
                  weighted ? 'left-5.5' : 'left-0.5'
                }`}
              />
            </button>
          </div>

          {/* Probe Point Coordinates HUD */}
          <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg text-xs space-y-2">
            <div className="flex items-center justify-between text-slate-600">
              <span className="font-medium flex items-center gap-1">
                <MousePointerClick className="w-3.5 h-3.5 text-sky-600" />
                当前探针坐标 (Probe Point)
              </span>
              <span className="text-[10px] text-slate-400">点击画布任意放置</span>
            </div>
            <div className="font-mono text-xs flex justify-between bg-white p-2 rounded border border-slate-200">
              <span>
                {featX}: <strong className="text-slate-900">{probePoint.x.toFixed(2)}</strong>
              </span>
              <span>
                {featY}: <strong className="text-slate-900">{probePoint.y.toFixed(2)}</strong>
              </span>
            </div>
          </div>
        </div>

        {/* Right 2 Columns: Interactive 2D Feature Space Canvas & Neighbor Details */}
        <div className="lg:col-span-2 bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Target className="w-4 h-4 text-sky-600" />
              交互式度量空间与动态决策边界 (KNN Proximity Space)
            </h3>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 font-mono">实时判定:</span>
              <span className="text-xs font-mono text-sky-700 font-bold bg-sky-50 px-3 py-1 rounded-full border border-sky-200 shadow-2xs">
                {String(knnResult.predictedValue)}
              </span>
            </div>
          </div>

          {/* Animation Toolbar for KNN */}
          <div className="flex flex-wrap items-center justify-between gap-2.5 p-3 bg-slate-50 rounded-lg border border-slate-200">
            <div className="flex flex-wrap items-center gap-2">
              {/* Radar Wave Scan Button */}
              <button
                type="button"
                onClick={() => {
                  if (isPatrolling) setIsPatrolling(false);
                  setIsRadarScanning(true);
                }}
                disabled={isRadarScanning}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-sky-600 hover:bg-sky-700 disabled:bg-sky-400 text-white transition-colors cursor-pointer shadow-2xs"
                title="发射度量雷达波逐步捕捉最近K个邻居"
              >
                <Radio className={`w-3.5 h-3.5 ${isRadarScanning ? 'animate-spin' : ''}`} />
                <span>{isRadarScanning ? '雷达波探测中...' : '启动近邻雷达搜寻'}</span>
              </button>

              {/* Orbit Patrol Button */}
              <button
                type="button"
                onClick={() => {
                  if (isRadarScanning) setIsRadarScanning(false);
                  setIsPatrolling(!isPatrolling);
                }}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors cursor-pointer shadow-2xs ${
                  isPatrolling
                    ? 'bg-amber-600 hover:bg-amber-700 text-white'
                    : 'bg-emerald-600 hover:bg-emerald-700 text-white'
                }`}
                title="让探针自动沿空间轨道平滑漫游，动态演示分类决策边界变化"
              >
                {isPatrolling ? <Pause className="w-3.5 h-3.5" /> : <Navigation className="w-3.5 h-3.5" />}
                <span>{isPatrolling ? '暂停空间漫游' : '启动边界漫游巡航'}</span>
              </button>

              {/* Reset Probe */}
              <button
                type="button"
                onClick={() => {
                  setIsPatrolling(false);
                  setIsRadarScanning(false);
                  setProbePoint({
                    x: (minX + maxX) / 2,
                    y: (minY + maxY) / 2,
                  });
                  setRadarProgress(1);
                }}
                className="p-1.5 rounded-lg text-slate-600 hover:bg-slate-200 border border-slate-200 cursor-pointer"
                title="重置探针至空间几何中心"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* Radar Scope Telemetry */}
            <div className="flex items-center gap-2 text-[11px] font-mono text-slate-600">
              <span className="bg-white px-2 py-0.5 rounded border border-slate-200">
                度量半径: <strong>{(maxNeighborPx * radarProgress).toFixed(1)} px</strong>
              </span>
              <span className="bg-white px-2 py-0.5 rounded border border-slate-200 text-sky-700 font-bold">
                已锁定: {Math.min(k, Math.round(k * radarProgress))} / {k} 邻居
              </span>
            </div>
          </div>

          {/* SVG Canvas for Training Points, Radar Waves & Probe */}
          <div className="relative border border-slate-200 rounded-lg bg-slate-50/50 p-2 overflow-hidden shadow-inner">
            <svg
              viewBox="0 0 500 380"
              onClick={handleCanvasClick}
              className="w-full h-[330px] cursor-crosshair select-none"
            >
              <defs>
                <radialGradient id="radarGlow" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.25" />
                  <stop offset="85%" stopColor="#0284c7" stopOpacity="0.1" />
                  <stop offset="100%" stopColor="#0284c7" stopOpacity="0.35" />
                </radialGradient>
              </defs>

              {/* Grid Lines */}
              <line x1="50" y1="340" x2="480" y2="340" stroke="#cbd5e1" strokeWidth="1" />
              <line x1="50" y1="30" x2="50" y2="340" stroke="#cbd5e1" strokeWidth="1" />

              {/* Axis Labels */}
              <text x="480" y="360" fontSize="10" fill="#64748b" textAnchor="end" fontFamily="monospace">
                {featX} →
              </text>
              <text x="35" y="30" fontSize="10" fill="#64748b" textAnchor="start" fontFamily="monospace">
                ↑ {featY}
              </text>

              {/* Radar Radiating Wave Front */}
              {radarProgress > 0 && (
                metric === 'euclidean' ? (
                  <circle
                    cx={toSvgX(probePoint.x)}
                    cy={toSvgY(probePoint.y)}
                    r={maxNeighborPx * radarProgress}
                    fill="url(#radarGlow)"
                    stroke="#0284c7"
                    strokeWidth="2"
                    strokeDasharray={isRadarScanning ? '4,4' : 'none'}
                    className={isRadarScanning ? 'animate-pulse' : ''}
                  />
                ) : (
                  <polygon
                    points={`
                      ${toSvgX(probePoint.x)},${toSvgY(probePoint.y) - maxNeighborPx * radarProgress}
                      ${toSvgX(probePoint.x) + maxNeighborPx * radarProgress},${toSvgY(probePoint.y)}
                      ${toSvgX(probePoint.x)},${toSvgY(probePoint.y) + maxNeighborPx * radarProgress}
                      ${toSvgX(probePoint.x) - maxNeighborPx * radarProgress},${toSvgY(probePoint.y)}
                    `}
                    fill="url(#radarGlow)"
                    stroke="#0284c7"
                    strokeWidth="2"
                    strokeDasharray={isRadarScanning ? '4,4' : 'none'}
                  />
                )
              )}

              {/* Vector Rays from Probe to K Nearest Neighbors */}
              {knnResult.neighbors.map((neighbor) => {
                const nx = Number(neighbor.point[featX]);
                const ny = Number(neighbor.point[featY]);
                const distPx = Math.hypot(toSvgX(nx) - toSvgX(probePoint.x), toSvgY(ny) - toSvgY(probePoint.y));
                const isCaptured = distPx <= maxNeighborPx * radarProgress;

                if (!isCaptured) return null;

                return (
                  <g key={`ray_${neighbor.index}`}>
                    <line
                      x1={toSvgX(probePoint.x)}
                      y1={toSvgY(probePoint.y)}
                      x2={toSvgX(nx)}
                      y2={toSvgY(ny)}
                      stroke="#0284c7"
                      strokeWidth="1.8"
                      strokeDasharray="3,3"
                      opacity="0.9"
                    />
                    {/* Pulsing Capture Ring */}
                    <circle
                      cx={toSvgX(nx)}
                      cy={toSvgY(ny)}
                      r="9"
                      fill="none"
                      stroke="#0284c7"
                      strokeWidth="1.5"
                      className="animate-ping opacity-75"
                    />
                  </g>
                );
              })}

              {/* Training Points */}
              {splitResult.trainData.map((d) => {
                const px = toSvgX(Number(d[featX]));
                const py = toSvgY(Number(d[featY]));
                const c = getColor(d[dataset.target]);
                const isNeighbor = knnResult.neighbors.some((n) => n.index === d.id);
                const distPx = Math.hypot(px - toSvgX(probePoint.x), py - toSvgY(probePoint.y));
                const isCaptured = isNeighbor && distPx <= maxNeighborPx * radarProgress;

                return (
                  <circle
                    key={d.id}
                    cx={px}
                    cy={py}
                    r={isCaptured ? 6.5 : isNeighbor ? 5.5 : 4}
                    fill={c.fill}
                    stroke={isCaptured ? '#0284c7' : c.stroke}
                    strokeWidth={isCaptured ? 3 : isNeighbor ? 2 : 1}
                    opacity={isCaptured ? 1.0 : isNeighbor ? 0.85 : 0.6}
                  />
                );
              })}

              {/* Interactive Query Probe Point */}
              <g transform={`translate(${toSvgX(probePoint.x)}, ${toSvgY(probePoint.y)})`}>
                {/* Target crosshair rings */}
                <circle
                  r="16"
                  fill="none"
                  stroke="#ef4444"
                  strokeWidth="1.5"
                  strokeDasharray="4,4"
                  className={isPatrolling ? 'animate-spin' : ''}
                />
                <circle r="6" fill="#ef4444" stroke="#ffffff" strokeWidth="2" className="shadow-md" />
                <line x1="-20" y1="0" x2="20" y2="0" stroke="#ef4444" strokeWidth="1.5" opacity="0.6" />
                <line x1="0" y1="-20" x2="0" y2="20" stroke="#ef4444" strokeWidth="1.5" opacity="0.6" />
                <text x="12" y="-10" fontSize="10" fill="#dc2626" fontWeight="bold" fontFamily="monospace">
                  {isPatrolling ? '🛰️ 漫游探针' : '🎯 测试探针'}
                </text>
              </g>
            </svg>
          </div>

          {/* Voting Breakdown & Nearest Neighbors Table */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Votes Tally */}
            <div className="border border-slate-200 rounded-lg p-3.5 bg-slate-50/50 space-y-2.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-800">
                  最近 {k} 个邻居投票统计 (Voting Tally)
                </span>
                <span className="text-[10px] font-mono text-slate-500">
                  {weighted ? '依据反比权重累加' : '等权重多数决'}
                </span>
              </div>

              <div className="space-y-2 font-mono text-xs">
                {Object.entries(knnResult.votes).map(([cls, count]) => {
                  const c = getColor(cls);
                  const isWinner = cls === String(knnResult.predictedValue);
                  const totalVotes = Object.values(knnResult.votes).reduce((a, b) => a + b, 0) || 1;
                  const pct = ((count / totalVotes) * 100).toFixed(1);

                  return (
                    <div key={cls} className="space-y-1">
                      <div className="flex items-center justify-between bg-white px-2.5 py-1.5 rounded border border-slate-200">
                        <span className="font-semibold flex items-center gap-1.5" style={{ color: c.stroke }}>
                          {cls} {isWinner && '👑'}
                        </span>
                        <span className="text-slate-700 text-xs">
                          {weighted ? `权重分: ${count.toFixed(2)}` : `${count} 票 (${pct}%)`}
                        </span>
                      </div>
                      <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-300"
                          style={{
                            width: `${pct}%`,
                            backgroundColor: c.stroke,
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Top Nearest Neighbors Ranking List */}
            <div className="border border-slate-200 rounded-lg p-3.5 bg-slate-50/50 space-y-2">
              <span className="text-xs font-semibold text-slate-800 block">
                Top 近邻度量空间距离拓扑表
              </span>
              <div className="max-h-[140px] overflow-y-auto space-y-1 text-xs font-mono">
                {knnResult.neighbors.map((n, idx) => {
                  const c = getColor(n.targetValue);
                  return (
                    <div
                      key={idx}
                      className="flex items-center justify-between p-1.5 bg-white rounded border border-slate-100 text-[11px]"
                    >
                      <span className="text-slate-400">#{idx + 1} ID:{n.point.id}</span>
                      <span className="font-semibold px-2 py-0.5 rounded text-[10px]" style={{ backgroundColor: `${c.fill}33`, color: c.stroke }}>
                        {String(n.targetValue)}
                      </span>
                      <span className="text-slate-600 font-mono">d = {n.distance}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

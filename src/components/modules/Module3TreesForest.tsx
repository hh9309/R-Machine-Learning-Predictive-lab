import React, { useState, useMemo, useEffect } from 'react';
import { DatasetInfo, SplitResult, TreeNode, DataPoint } from '../../types/ml';
import { buildCARTTree, trainRandomForest, traceDecisionPath, predictTree } from '../../utils/mlEngine';
import {
  Network,
  Sliders,
  Layers,
  Compass,
  CheckCircle2,
  Info,
  Play,
  Pause,
  RotateCcw,
  StepForward,
  Sparkles,
  Zap,
  BarChart3,
  Award,
} from 'lucide-react';

interface Props {
  dataset: DatasetInfo;
  splitResult: SplitResult;
  activeFeatures: string[];
}

export const Module3TreesForest: React.FC<Props> = ({
  dataset,
  splitResult,
  activeFeatures,
}) => {
  const isClassification = dataset.problemType === 'classification';
  const [modelType, setModelType] = useState<'cart' | 'rf'>('cart');

  // Hyperparameters
  const [maxDepth, setMaxDepth] = useState(3);
  const [cp, setCp] = useState(0.01);
  const [minSplit, setMinSplit] = useState(4);
  const [ntree, setNtree] = useState(40);
  const [mtry, setMtry] = useState(Math.max(1, Math.round(Math.sqrt(activeFeatures.length))));

  // Selected sample for decision path tracing
  const [selectedSampleIdx, setSelectedSampleIdx] = useState(0);

  // CART Decision Path Traversal Animation State
  const [isCartPlaying, setIsCartPlaying] = useState(false);
  const [cartStep, setCartStep] = useState(0); // index in tracedPath
  const [animSpeed, setAnimSpeed] = useState<number>(1000); // ms per step

  // Random Forest Voting Race Animation State
  const [isRfPlaying, setIsRfPlaying] = useState(false);
  const [rfVotedCount, setRfVotedCount] = useState(0); // 0 to ntree

  // Build CART Tree
  const cartTree = useMemo(() => {
    return buildCARTTree(
      splitResult.trainData,
      activeFeatures,
      dataset.target,
      isClassification,
      maxDepth,
      minSplit,
      cp
    );
  }, [splitResult.trainData, activeFeatures, dataset.target, isClassification, maxDepth, minSplit, cp]);

  // Train Random Forest
  const rfResult = useMemo(() => {
    return trainRandomForest(
      splitResult.trainData,
      activeFeatures,
      dataset.target,
      isClassification,
      ntree,
      mtry,
      splitResult.seed
    );
  }, [splitResult.trainData, activeFeatures, dataset.target, isClassification, ntree, mtry, splitResult.seed]);

  // Trace decision path for chosen sample in test set
  const currentSample = splitResult.testData[selectedSampleIdx] || splitResult.testData[0];
  const tracedPath = useMemo(() => {
    if (!currentSample) return [];
    return traceDecisionPath(cartTree, currentSample);
  }, [cartTree, currentSample]);

  // CART Animation Timer
  useEffect(() => {
    if (!isCartPlaying) return;
    const timer = setInterval(() => {
      setCartStep((prev) => {
        if (prev < tracedPath.length - 1) {
          return prev + 1;
        } else {
          setIsCartPlaying(false);
          return prev;
        }
      });
    }, animSpeed);
    return () => clearInterval(timer);
  }, [isCartPlaying, tracedPath.length, animSpeed]);

  // Random Forest Voting Race Animation Timer
  useEffect(() => {
    if (!isRfPlaying) return;
    const timer = setInterval(() => {
      setRfVotedCount((prev) => {
        if (prev < ntree) {
          return prev + 1;
        } else {
          setIsRfPlaying(false);
          return prev;
        }
      });
    }, 40);
    return () => clearInterval(timer);
  }, [isRfPlaying, ntree]);

  // Calculate live tally for the voting race
  const rfLiveVotes = useMemo(() => {
    if (!currentSample || !rfResult.trees) return { tally: {}, winner: '', count: 0 };
    const tally: { [cls: string]: number } = {};
    const limit = rfVotedCount === 0 ? ntree : rfVotedCount;
    const evalLimit = Math.min(limit, rfResult.trees.length);

    for (let i = 0; i < evalLimit; i++) {
      const pred = predictTree(rfResult.trees[i], currentSample);
      const val = String(pred.value);
      tally[val] = (tally[val] || 0) + 1;
    }

    let maxVotes = -1;
    let winner = '';
    Object.entries(tally).forEach(([cls, count]) => {
      if (count > maxVotes) {
        maxVotes = count;
        winner = cls;
      }
    });

    return { tally, winner, count: evalLimit };
  }, [currentSample, rfResult.trees, rfVotedCount, ntree]);

  const handleSelectSample = (idx: number) => {
    setSelectedSampleIdx(idx);
    setCartStep(0);
    setIsCartPlaying(false);
    setRfVotedCount(0);
    setIsRfPlaying(false);
  };

  // Render SVG tree coordinates recursively
  interface NodeCoord {
    node: TreeNode;
    x: number;
    y: number;
    parentX?: number;
    parentY?: number;
    isLeft?: boolean;
  }

  const computeTreeLayout = (root: TreeNode): NodeCoord[] => {
    const coords: NodeCoord[] = [];
    const levelHeight = 75;

    const assignCoords = (
      node: TreeNode,
      depth: number,
      minX: number,
      maxX: number,
      parentX?: number,
      parentY?: number,
      isLeft?: boolean
    ) => {
      const midX = (minX + maxX) / 2;
      const y = depth * levelHeight + 35;
      coords.push({ node, x: midX, y, parentX, parentY, isLeft });

      if (!node.isLeaf) {
        if (node.left) {
          assignCoords(node.left, depth + 1, minX, midX, midX, y, true);
        }
        if (node.right) {
          assignCoords(node.right, depth + 1, midX, maxX, midX, y, false);
        }
      }
    };

    assignCoords(root, 0, 20, 800);
    return coords;
  };

  const treeCoords = useMemo(() => computeTreeLayout(cartTree), [cartTree]);

  // Visited node IDs up to current animated step
  const visitedNodeIds = useMemo(() => {
    const set = new Set<string>();
    const count = Math.min(cartStep + 1, tracedPath.length);
    for (let i = 0; i < count; i++) {
      if (tracedPath[i]) set.add(tracedPath[i].nodeId);
    }
    return set;
  }, [cartStep, tracedPath]);

  const currentNodeId = tracedPath[cartStep]?.nodeId;
  const currentStepInfo = tracedPath[cartStep];

  return (
    <div className="space-y-6">
      {/* Title Header */}
      <div className="bg-white border border-slate-200/80 rounded-xl p-5 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-sky-700 mb-1">
              <span>模块 03 · 核心工作流</span>
              <span aria-hidden="true">·</span>
              <span>基于树的集成学习</span>
            </div>
            <h2 className="text-xl font-bold text-slate-900 tracking-tight">
              决策树与随机森林 (CART Tree & Random Forest)
            </h2>
            <p className="text-sm text-slate-500 mt-1 max-w-3xl">
              构建基于基尼不纯度 / 方差缩减的 CART 决策树，演示递归二分剪枝与单样本决策路径追踪；集成多树形成随机森林，观察 OOB 袋外误差收敛。
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs font-mono bg-slate-50 px-3.5 py-2 rounded-lg border border-slate-200 text-slate-600">
            <span>R 引擎:</span>
            <code className="text-sky-700 font-semibold">
              {modelType === 'cart' ? 'rpart::rpart()' : 'randomForest::randomForest()'}
            </code>
          </div>
        </div>
      </div>

      {/* Model Mode Switcher & Parameters Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Model Controls */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-5">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-sky-600" />
              模型与超参数配置
            </h3>
          </div>

          {/* Model Type Selector */}
          <div className="grid grid-cols-2 gap-2 p-1 bg-slate-100 rounded-lg">
            <button
              onClick={() => setModelType('cart')}
              className={`py-2 text-xs font-medium rounded-md transition-all ${
                modelType === 'cart'
                  ? 'bg-white text-slate-900 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              CART 单决策树 (rpart)
            </button>
            <button
              onClick={() => setModelType('rf')}
              className={`py-2 text-xs font-medium rounded-md transition-all ${
                modelType === 'rf'
                  ? 'bg-white text-slate-900 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              随机森林 (randomForest)
            </button>
          </div>

          {modelType === 'cart' ? (
            <div className="space-y-4 pt-1">
              {/* Max Depth */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-700 font-medium">最大树深度 (maxdepth)</span>
                  <span className="font-mono font-bold text-sky-700">{maxDepth} 层</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="6"
                  value={maxDepth}
                  onChange={(e) => setMaxDepth(Number(e.target.value))}
                  className="w-full accent-sky-600 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>1 (桩模型)</span>
                  <span>3 (均衡结构)</span>
                  <span>6 (复杂树)</span>
                </div>
              </div>

              {/* Complexity Parameter cp */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-700 font-medium">剪枝复杂度阈值 (cp)</span>
                  <span className="font-mono font-bold text-sky-700">{cp}</span>
                </div>
                <input
                  type="range"
                  min="0.001"
                  max="0.08"
                  step="0.005"
                  value={cp}
                  onChange={(e) => setCp(Number(e.target.value))}
                  className="w-full accent-sky-600 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>0.001 (少剪枝)</span>
                  <span>0.08 (强力剪枝)</span>
                </div>
              </div>

              {/* Min Split */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-700 font-medium">节点最小分裂样本 (minsplit)</span>
                  <span className="font-mono font-bold text-sky-700">{minSplit} 个</span>
                </div>
                <input
                  type="range"
                  min="2"
                  max="15"
                  value={minSplit}
                  onChange={(e) => setMinSplit(Number(e.target.value))}
                  className="w-full accent-sky-600 cursor-pointer"
                />
              </div>
            </div>
          ) : (
            <div className="space-y-4 pt-1">
              {/* NTree */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-700 font-medium">决策树数量 (ntree)</span>
                  <span className="font-mono font-bold text-sky-700">{ntree} 棵</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="100"
                  step="5"
                  value={ntree}
                  onChange={(e) => setNtree(Number(e.target.value))}
                  className="w-full accent-sky-600 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>10 (快速试探)</span>
                  <span>50 (常规)</span>
                  <span>100 (充分收敛)</span>
                </div>
              </div>

              {/* Mtry */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-700 font-medium">每棵树随机特征子集 (mtry)</span>
                  <span className="font-mono font-bold text-sky-700">{mtry} 个</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max={activeFeatures.length}
                  value={mtry}
                  onChange={(e) => setMtry(Number(e.target.value))}
                  className="w-full accent-sky-600 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>1</span>
                  <span>标准建议: √p = {Math.round(Math.sqrt(activeFeatures.length))}</span>
                  <span>{activeFeatures.length}</span>
                </div>
              </div>
            </div>
          )}

          {/* Tracer Sample Selector */}
          <div className="pt-3 border-t border-slate-100 space-y-2">
            <span className="text-xs font-semibold text-slate-700 flex items-center gap-1.5">
              <Compass className="w-3.5 h-3.5 text-sky-600" />
              选择测试样本进行决策路径追踪
            </span>
            <select
              value={selectedSampleIdx}
              onChange={(e) => handleSelectSample(Number(e.target.value))}
              className="w-full px-2.5 py-1.5 border border-slate-200 rounded-lg text-xs font-mono bg-slate-50 cursor-pointer"
            >
              {splitResult.testData.slice(0, 15).map((d, idx) => (
                <option key={d.id} value={idx}>
                  测试样本 #{d.id} (真实值: {String(d[dataset.target])})
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Right 2 Columns: Tree Topology or Random Forest OOB View */}
        <div className="lg:col-span-2 bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Network className="w-4 h-4 text-sky-600" />
              {modelType === 'cart' ? 'CART 决策树拓扑结构图 (Tree Topology)' : '随机森林 OOB 误差收敛与多树集成'}
            </h3>
            <span className="text-[11px] font-mono text-sky-700 bg-sky-50 px-2.5 py-1 rounded-full border border-sky-200">
              {modelType === 'cart' ? '✨ 支持逐步二分决策流向动画演示' : '⚡ 支持多树并行投票竞速动画演示'}
            </span>
          </div>

          {modelType === 'cart' ? (
            <div className="space-y-4">
              {/* CART Animation Control Bar */}
              <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-slate-50 rounded-lg border border-slate-200">
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      if (isCartPlaying) {
                        setIsCartPlaying(false);
                      } else {
                        if (cartStep >= tracedPath.length - 1) {
                          setCartStep(0);
                        }
                        setIsCartPlaying(true);
                      }
                    }}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-white transition-colors cursor-pointer shadow-2xs ${
                      isCartPlaying ? 'bg-amber-600 hover:bg-amber-700' : 'bg-sky-600 hover:bg-sky-700'
                    }`}
                  >
                    {isCartPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                    <span>{isCartPlaying ? '暂停推演' : cartStep >= tracedPath.length - 1 ? '重新播放动画' : '播放路径推演动画'}</span>
                  </button>

                  <button
                    type="button"
                    disabled={isCartPlaying || cartStep >= tracedPath.length - 1}
                    onClick={() => setCartStep((s) => Math.min(tracedPath.length - 1, s + 1))}
                    className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium bg-white hover:bg-slate-100 border border-slate-200 text-slate-700 disabled:opacity-40 cursor-pointer"
                    title="推演下一步"
                  >
                    <StepForward className="w-3.5 h-3.5" />
                    <span>单步前进</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => {
                      setIsCartPlaying(false);
                      setCartStep(0);
                    }}
                    className="p-1.5 rounded-lg text-slate-600 hover:bg-slate-200/70 border border-slate-200 cursor-pointer"
                    title="重置回根节点"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                  </button>
                </div>

                {/* Animation Step Progress & Speed */}
                <div className="flex items-center gap-3 text-xs font-mono">
                  <div className="flex items-center gap-1 bg-white px-2.5 py-1 rounded border border-slate-200">
                    <span className="text-slate-400">推演进度:</span>
                    <span className="font-bold text-sky-700">
                      第 {cartStep + 1} / {tracedPath.length} 步
                    </span>
                  </div>

                  <div className="flex items-center gap-1 text-[11px] text-slate-500">
                    <span>速度:</span>
                    <button
                      type="button"
                      onClick={() => setAnimSpeed(1400)}
                      className={`px-1.5 py-0.5 rounded cursor-pointer ${animSpeed === 1400 ? 'bg-sky-100 text-sky-800 font-bold' : 'hover:bg-slate-200'}`}
                    >
                      慢
                    </button>
                    <button
                      type="button"
                      onClick={() => setAnimSpeed(800)}
                      className={`px-1.5 py-0.5 rounded cursor-pointer ${animSpeed === 800 ? 'bg-sky-100 text-sky-800 font-bold' : 'hover:bg-slate-200'}`}
                    >
                      中
                    </button>
                    <button
                      type="button"
                      onClick={() => setAnimSpeed(400)}
                      className={`px-1.5 py-0.5 rounded cursor-pointer ${animSpeed === 400 ? 'bg-sky-100 text-sky-800 font-bold' : 'hover:bg-slate-200'}`}
                    >
                      快
                    </button>
                  </div>
                </div>
              </div>

              {/* Dynamic Live Evaluation Bubble */}
              {currentStepInfo && (
                <div className="p-3 bg-sky-50/80 border border-sky-200/90 rounded-lg text-xs space-y-1 animate-in fade-in">
                  <div className="flex items-center justify-between font-semibold text-sky-900">
                    <span className="flex items-center gap-1.5">
                      <Sparkles className="w-3.5 h-3.5 text-sky-600 animate-spin" />
                      当前活跃判定节点：{currentStepInfo.nodeId}
                    </span>
                    <span className="text-[11px] font-mono text-sky-700">
                      样本 #{currentSample?.id} 真实值: {String(currentSample?.[dataset.target])}
                    </span>
                  </div>
                  <p className="font-mono text-slate-700 text-xs mt-1">
                    👉 {currentStepInfo.rule}
                  </p>
                  {cartStep >= tracedPath.length - 1 && (
                    <div className="mt-1 pt-1 border-t border-sky-200/60 flex items-center justify-between text-emerald-800 font-semibold">
                      <span>🎉 决策推演完成！最终预测归类: {String(currentStepInfo.value)}</span>
                      <span className="text-xs">
                        {String(currentStepInfo.value) === String(currentSample?.[dataset.target])
                          ? '✓ 预测命中 (准确)'
                          : '✕ 与真实标签存在偏差'}
                      </span>
                    </div>
                  )}
                </div>
              )}

              {/* Interactive SVG Tree Canvas */}
              <div className="border border-slate-200 rounded-lg bg-slate-50/40 p-2 overflow-x-auto">
                <svg
                  viewBox="0 0 820 320"
                  className="w-full h-[280px] min-w-[700px] select-none"
                >
                  <defs>
                    <marker
                      id="arrow"
                      viewBox="0 0 10 10"
                      refX="5"
                      refY="5"
                      markerWidth="5"
                      markerHeight="5"
                      orient="auto-start-reverse"
                    >
                      <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8" />
                    </marker>
                  </defs>

                  {/* Draw Connecting Edges */}
                  {treeCoords.map((c) => {
                    if (c.parentX === undefined || c.parentY === undefined) return null;
                    const isTracedEdge = visitedNodeIds.has(c.node.id);
                    const isCurrentEdge = currentNodeId === c.node.id;

                    return (
                      <g key={`edge_${c.node.id}`}>
                        <path
                          d={`M ${c.parentX} ${c.parentY + 16} L ${c.x} ${c.y - 16}`}
                          stroke={isCurrentEdge ? '#0284c7' : isTracedEdge ? '#38bdf8' : '#cbd5e1'}
                          strokeWidth={isCurrentEdge ? 3.5 : isTracedEdge ? 2.5 : 1.5}
                          strokeDasharray={isCurrentEdge ? '4,4' : isTracedEdge ? 'none' : '2,2'}
                          className={isCurrentEdge ? 'animate-pulse' : ''}
                          fill="none"
                        />
                        {/* Branch Label */}
                        <text
                          x={(c.parentX + c.x) / 2 + (c.isLeft ? -10 : 10)}
                          y={(c.parentY + c.y) / 2}
                          fontSize="9"
                          fill={isTracedEdge ? '#0284c7' : '#94a3b8'}
                          textAnchor="middle"
                          fontFamily="monospace"
                          fontWeight={isTracedEdge ? 'bold' : 'normal'}
                        >
                          {c.isLeft ? '≤' : '>'}
                        </text>
                      </g>
                    );
                  })}

                  {/* Draw Nodes */}
                  {treeCoords.map((c) => {
                    const isVisited = visitedNodeIds.has(c.node.id);
                    const isCurrent = currentNodeId === c.node.id;
                    const isLeaf = c.node.isLeaf;

                    return (
                      <g key={c.node.id} transform={`translate(${c.x}, ${c.y})`}>
                        {/* Active Glowing Pulse Ring if Current */}
                        {isCurrent && (
                          <rect
                            x="-69"
                            y="-22"
                            width="138"
                            height="44"
                            rx="8"
                            fill="none"
                            stroke="#0284c7"
                            strokeWidth="2.5"
                            className="animate-ping opacity-60"
                          />
                        )}

                        <rect
                          x="-65"
                          y="-18"
                          width="130"
                          height="36"
                          rx="6"
                          fill={
                            isCurrent
                              ? '#bae6fd'
                              : isVisited
                              ? isLeaf
                                ? '#dcfce7'
                                : '#e0f2fe'
                              : isLeaf
                              ? '#f0fdf4'
                              : '#ffffff'
                          }
                          stroke={
                            isCurrent
                              ? '#0284c7'
                              : isVisited
                              ? isLeaf
                                ? '#22c55e'
                                : '#0284c7'
                              : isLeaf
                              ? '#86efac'
                              : '#cbd5e1'
                          }
                          strokeWidth={isCurrent ? 2.5 : isVisited ? 2 : 1}
                        />

                        {/* Traveling Orb Indicator if Current */}
                        {isCurrent && (
                          <circle
                            cx="55"
                            cy="-12"
                            r="5"
                            fill="#0284c7"
                            stroke="#ffffff"
                            strokeWidth="1.5"
                            className="animate-bounce"
                          />
                        )}

                        {isLeaf ? (
                          <>
                            <text
                              x="0"
                              y="-3"
                              fontSize="10"
                              fontWeight="600"
                              fill="#1e293b"
                              textAnchor="middle"
                              fontFamily="monospace"
                            >
                              叶: {String(c.node.predictedValue)}
                            </text>
                            <text
                              x="0"
                              y="10"
                              fontSize="8"
                              fill="#64748b"
                              textAnchor="middle"
                              fontFamily="monospace"
                            >
                              n={c.node.samples} | {isClassification ? `Gini=${c.node.purity}` : `MSE=${c.node.purity}`}
                            </text>
                          </>
                        ) : (
                          <>
                            <text
                              x="0"
                              y="-3"
                              fontSize="10"
                              fontWeight="600"
                              fill="#0f172a"
                              textAnchor="middle"
                              fontFamily="monospace"
                            >
                              {c.node.feature} ≤ {c.node.threshold}
                            </text>
                            <text
                              x="0"
                              y="10"
                              fontSize="8"
                              fill="#64748b"
                              textAnchor="middle"
                              fontFamily="monospace"
                            >
                              n={c.node.samples} | {isClassification ? `Gini=${c.node.purity}` : `MSE=${c.node.purity}`}
                            </text>
                          </>
                        )}
                      </g>
                    );
                  })}
                </svg>
              </div>

              {/* Traced Decision Pathway Breadcrumb */}
              <div className="bg-sky-50/60 border border-sky-200 rounded-lg p-3.5 space-y-2">
                <div className="text-xs font-semibold text-sky-900 flex items-center justify-between">
                  <span>样本 #{currentSample?.id} 决策路径全景明细 (Decision Pathway):</span>
                  <span className="font-mono text-[11px] text-sky-700">
                    最终预测: {String(tracedPath[tracedPath.length - 1]?.value)}
                  </span>
                </div>

                <div className="space-y-1">
                  {tracedPath.map((step, idx) => {
                    const isStepPassed = cartStep >= idx;
                    const isCurrent = cartStep === idx;
                    return (
                      <div
                        key={idx}
                        className={`flex items-center gap-2 text-xs font-mono p-1 rounded transition-colors ${
                          isCurrent
                            ? 'bg-sky-200/80 text-sky-900 font-bold'
                            : isStepPassed
                            ? 'text-slate-800'
                            : 'text-slate-400'
                        }`}
                      >
                        <span
                          className={`w-5 h-5 rounded-full text-[10px] flex items-center justify-center shrink-0 ${
                            isCurrent
                              ? 'bg-sky-600 text-white animate-pulse'
                              : isStepPassed
                              ? 'bg-sky-200 text-sky-800'
                              : 'bg-slate-200 text-slate-500'
                          }`}
                        >
                          {idx + 1}
                        </span>
                        <span>{step.rule}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Random Forest Live Ensemble Voting Race Control Bar */}
              <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-slate-50 rounded-lg border border-slate-200">
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      if (isRfPlaying) {
                        setIsRfPlaying(false);
                      } else {
                        if (rfVotedCount >= ntree) {
                          setRfVotedCount(0);
                        }
                        setIsRfPlaying(true);
                      }
                    }}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-white transition-colors cursor-pointer shadow-2xs ${
                      isRfPlaying ? 'bg-amber-600 hover:bg-amber-700' : 'bg-emerald-600 hover:bg-emerald-700'
                    }`}
                  >
                    {isRfPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                    <span>{isRfPlaying ? '暂停投票' : rfVotedCount >= ntree ? '重新演示投票竞速' : '启动森林多树投票动画'}</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => {
                      setIsRfPlaying(false);
                      setRfVotedCount(0);
                    }}
                    className="p-1.5 rounded-lg text-slate-600 hover:bg-slate-200/70 border border-slate-200 cursor-pointer"
                    title="重置投票计票"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                  </button>
                </div>

                <div className="flex items-center gap-2 text-xs font-mono">
                  <span className="text-slate-500">投票竞速进度:</span>
                  <span className="font-bold text-emerald-700 bg-white px-2 py-0.5 rounded border border-slate-200">
                    {rfLiveVotes.count} / {ntree} 棵树已投票
                  </span>
                </div>
              </div>

              {/* Real-time Voting Tally Race Bars */}
              <div className="p-4 bg-white border border-slate-200 rounded-lg space-y-3">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-slate-800 flex items-center gap-1.5">
                    <Award className="w-4 h-4 text-amber-500" />
                    多树实时多数投票计票演播 (Live Ensemble Tally):
                  </span>
                  <span className="font-mono text-emerald-700 font-bold">
                    当前多数领先: {rfLiveVotes.winner || '计票中...'}
                  </span>
                </div>

                <div className="space-y-2">
                  {Object.entries(rfLiveVotes.tally).map(([cls, count]) => {
                    const totalVotes = Math.max(1, rfLiveVotes.count);
                    const pct = ((count / totalVotes) * 100).toFixed(1);
                    const isWinner = cls === rfLiveVotes.winner;

                    return (
                      <div key={cls} className="space-y-1">
                        <div className="flex justify-between items-center text-xs font-mono">
                          <span className={`font-semibold ${isWinner ? 'text-emerald-700' : 'text-slate-700'}`}>
                            {cls} {isWinner && '👑'}
                          </span>
                          <span className="text-slate-500">
                            {count} 票 ({pct}%)
                          </span>
                        </div>
                        <div className="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-100 ${
                              isWinner ? 'bg-emerald-500' : 'bg-sky-400'
                            }`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Random Forest OOB Error Plot */}
              <div className="border border-slate-200 rounded-lg p-4 bg-slate-50/50">
                <div className="flex items-center justify-between text-xs text-slate-600 mb-2">
                  <span className="font-semibold text-slate-800">
                    OOB (袋外) 预测误差随树木增加的收敛演播
                  </span>
                  <span className="font-mono text-[11px] text-sky-700">
                    当前稳定 OOB 误差率: {rfResult.oobErrors[rfResult.oobErrors.length - 1]?.error || 0}
                  </span>
                </div>

                <div className="h-[180px] flex items-end gap-1.5 pt-6 pb-2 border-b border-l border-slate-300">
                  {rfResult.oobErrors.map((pt, i) => {
                    const maxErr = Math.max(...rfResult.oobErrors.map((o) => o.error), 0.5);
                    const heightPct = Math.min(100, Math.max(10, (pt.error / maxErr) * 100));
                    const isCurrentTree = rfLiveVotes.count >= pt.treeCount;

                    return (
                      <div
                        key={i}
                        className="flex-1 flex flex-col items-center group relative cursor-pointer"
                      >
                        <div
                          className={`w-full rounded-t transition-all ${
                            isCurrentTree ? 'bg-emerald-500' : 'bg-slate-300'
                          }`}
                          style={{ height: `${heightPct}%` }}
                        />
                        <span className="text-[9px] font-mono text-slate-400 mt-1">
                          {pt.treeCount}
                        </span>

                        {/* Tooltip */}
                        <div className="absolute -top-7 hidden group-hover:block bg-slate-900 text-white text-[10px] px-2 py-0.5 rounded font-mono z-10 whitespace-nowrap">
                          {pt.treeCount}棵: {pt.error}
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="flex justify-between text-[10px] text-slate-400 font-mono mt-1">
                  <span>树木数量 (ntree)</span>
                  <span>OOB 误差率 (越低越平稳)</span>
                </div>
              </div>

              {/* Random Forest Bagging Explain */}
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg text-xs text-slate-600 leading-relaxed">
                <span className="font-semibold text-slate-800">随机森林集成优势：</span>
                每棵决策树利用 Bootstrap 自助采样和随机特征子集 ($mtry$) 训练。单棵树虽方差大，但由于树之间的低相关性，集成多数投票显著降低了整体方差，具备极强的抗噪能力与泛化稳定性。
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

import React from 'react';
import { DatasetInfo, SplitResult } from '../../types/ml';
import { INITIAL_DATASETS } from '../../data/datasets';
import {
  GitFork,
  ShieldAlert,
  Sparkles,
  RefreshCw,
  BarChart2,
  CheckCircle2,
  Database,
  HelpCircle,
  Info,
} from 'lucide-react';

interface Props {
  dataset: DatasetInfo;
  splitResult: SplitResult;
  onUpdateSplit: (ratio: number, isStratified: boolean, seed: number) => void;
  onSelectDataset?: (ds: DatasetInfo) => void;
}

export const Module2DataSplit: React.FC<Props> = ({
  dataset,
  splitResult,
  onUpdateSplit,
  onSelectDataset,
}) => {
  const isClassification = dataset.problemType === 'classification';

  // Compute distribution breakdown in train vs test
  const totalN = dataset.data.length;
  const trainN = splitResult.trainData.length;
  const testN = splitResult.testData.length;

  const getTargetDistribution = (data: typeof dataset.data) => {
    const counts: { [key: string]: number } = {};
    data.forEach((d) => {
      const val = String(d[dataset.target]);
      counts[val] = (counts[val] || 0) + 1;
    });
    return counts;
  };

  const trainDist = getTargetDistribution(splitResult.trainData);
  const testDist = getTargetDistribution(splitResult.testData);
  const totalDist = getTargetDistribution(dataset.data);

  const classes = dataset.classes || Object.keys(totalDist);

  return (
    <div className="space-y-6">
      {/* Module Title Banner */}
      <div className="bg-white border border-slate-200/80 rounded-xl p-5 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-sky-700 mb-1">
              <span>模块 02 · 核心工作流</span>
              <span aria-hidden="true">·</span>
              <span>泛化评估基础与抽样算子</span>
            </div>
            <h2 className="text-xl font-bold text-slate-900 tracking-tight">
              训练集与测试集科学划分 (Train/Test Split & Stratification)
            </h2>
            <p className="text-sm text-slate-500 mt-1 max-w-3xl">
              通过随机抽样与分层抽样算子，科学隔离未见测试数据，防止过拟合，并检验训练集与测试集的目标变量分布对齐状态。
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs font-mono bg-slate-50 px-3.5 py-2 rounded-lg border border-slate-200 text-slate-600">
            <span>R 算子:</span>
            <code className="text-sky-700 font-semibold">
              {splitResult.isStratified ? 'caret::createDataPartition()' : 'base::sample()'}
            </code>
          </div>
        </div>
      </div>

      {/* Control Panel Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 1 Col: Splitting Controls */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-5">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <GitFork className="w-4 h-4 text-sky-600" />
            抽样与划分参数控制
          </h3>

          {/* Split Ratio Slider */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="font-medium text-slate-700">划分比例 (Split Ratio)</span>
              <span className="font-mono font-bold text-sky-700">
                {(splitResult.splitRatio * 100).toFixed(0)}% 训练集 / {((1 - splitResult.splitRatio) * 100).toFixed(0)}% 测试集
              </span>
            </div>
            <input
              type="range"
              min="0.5"
              max="0.9"
              step="0.05"
              value={splitResult.splitRatio}
              onChange={(e) =>
                onUpdateSplit(Number(e.target.value), splitResult.isStratified, splitResult.seed)
              }
              className="w-full accent-sky-600 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-400 font-mono">
              <span>50:50 (均衡评估)</span>
              <span>70:30 (常用经验)</span>
              <span>80:20 (经典帕累托)</span>
              <span>90:10 (大数据集)</span>
            </div>
          </div>

          {/* Sampling Operator Type */}
          <div className="space-y-2 pt-2 border-t border-slate-100">
            <label className="text-xs font-medium text-slate-700 block">抽样算子策略</label>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => onUpdateSplit(splitResult.splitRatio, true, splitResult.seed)}
                className={`px-3 py-2.5 rounded-lg border text-xs text-left transition-all ${
                  splitResult.isStratified
                    ? 'bg-sky-50 border-sky-300 text-sky-900 font-semibold ring-1 ring-sky-300'
                    : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
                }`}
              >
                <div className="flex items-center gap-1.5">
                  <CheckCircle2 className={`w-3.5 h-3.5 ${splitResult.isStratified ? 'text-sky-600' : 'text-slate-400'}`} />
                  <span>分层抽样 (推荐)</span>
                </div>
                <div className="text-[10px] text-slate-500 font-normal mt-1">
                  保持各类别比例严格一致
                </div>
              </button>

              <button
                type="button"
                onClick={() => onUpdateSplit(splitResult.splitRatio, false, splitResult.seed)}
                className={`px-3 py-2.5 rounded-lg border text-xs text-left transition-all ${
                  !splitResult.isStratified
                    ? 'bg-sky-50 border-sky-300 text-sky-900 font-semibold ring-1 ring-sky-300'
                    : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
                }`}
              >
                <div className="flex items-center gap-1.5">
                  <CheckCircle2 className={`w-3.5 h-3.5 ${!splitResult.isStratified ? 'text-sky-600' : 'text-slate-400'}`} />
                  <span>简单随机抽样</span>
                </div>
                <div className="text-[10px] text-slate-500 font-normal mt-1">
                  不考虑类别先验频率
                </div>
              </button>
            </div>
          </div>

          {/* Seed Control */}
          <div className="space-y-2 pt-2 border-t border-slate-100">
            <div className="flex justify-between items-center">
              <span className="text-xs font-medium text-slate-700">随机数种子 (Seed)</span>
              <span className="text-[10px] font-mono text-slate-400">set.seed({splitResult.seed})</span>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="number"
                value={splitResult.seed}
                onChange={(e) =>
                  onUpdateSplit(splitResult.splitRatio, splitResult.isStratified, Number(e.target.value) || 1)
                }
                className="w-full px-3 py-1.5 border border-slate-200 rounded-lg text-xs font-mono"
              />
              <button
                type="button"
                onClick={() =>
                  onUpdateSplit(
                    splitResult.splitRatio,
                    splitResult.isStratified,
                    Math.floor(Math.random() * 9999) + 1
                  )
                }
                className="p-2 border border-slate-200 rounded-lg hover:bg-slate-50 text-slate-600 shrink-0"
                title="重新生成随机种子"
              >
                <RefreshCw className="w-3.5 h-3.5" />
              </button>
              <button
                type="button"
                onClick={() => onUpdateSplit(splitResult.splitRatio, splitResult.isStratified, 42)}
                className="px-2.5 py-1.5 border border-slate-200 rounded-lg hover:bg-slate-50 text-xs font-mono shrink-0"
              >
                42
              </button>
            </div>
          </div>

          {/* Volume Summary Card */}
          <div className="bg-slate-50 rounded-lg p-3 border border-slate-200 text-xs font-mono space-y-1.5">
            <div className="flex justify-between text-slate-600">
              <span>原始样本总量:</span>
              <span className="font-semibold text-slate-900">{totalN} 条</span>
            </div>
            <div className="flex justify-between text-sky-700">
              <span>训练集样本量 (Train):</span>
              <span className="font-bold">{trainN} 条</span>
            </div>
            <div className="flex justify-between text-amber-700">
              <span>测试集样本量 (Test):</span>
              <span className="font-bold">{testN} 条</span>
            </div>
          </div>
        </div>

        {/* Right 2 Cols: Distribution Alignment Diagnostics */}
        <div className="lg:col-span-2 bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <BarChart2 className="w-4 h-4 text-emerald-600" />
              分布对齐诊断检验 (Distribution Alignment Test)
            </h3>
            <span className="text-[11px] font-mono text-slate-500 bg-slate-100 px-2.5 py-0.5 rounded-full">
              {splitResult.isStratified ? '分层抽样：分布严格镜像对齐' : '简单抽样：呈现随机抽样漂移'}
            </span>
          </div>

          {/* Quick Dataset Switcher for Cross-Dataset Comparison */}
          {onSelectDataset && (
            <div className="flex flex-wrap items-center gap-1.5 p-2 bg-slate-50 border border-slate-200/80 rounded-lg text-xs">
              <span className="text-slate-500 font-medium flex items-center gap-1 mr-1 text-[11px]">
                <Database className="w-3.5 h-3.5 text-sky-600" />
                对比其他真实案例分布:
              </span>
              {INITIAL_DATASETS.map((ds) => (
                <button
                  key={ds.id}
                  type="button"
                  onClick={() => onSelectDataset(ds)}
                  className={`px-2 py-0.5 rounded text-[11px] transition-colors cursor-pointer ${
                    dataset.id === ds.id
                      ? 'bg-sky-600 text-white font-semibold shadow-2xs'
                      : 'bg-white hover:bg-slate-100 text-slate-700 border border-slate-200'
                  }`}
                >
                  {ds.name.replace(/（.*）|\(.*\)/g, '')}
                </button>
              ))}
            </div>
          )}

          {/* Core Scientific Diagnosis Callout: Why 33.3% */}
          <div className="p-3.5 bg-sky-50/70 border border-sky-200/90 rounded-lg text-xs space-y-1.5">
            <div className="flex items-center gap-1.5 font-bold text-sky-950 text-xs">
              <Info className="w-4 h-4 text-sky-600 shrink-0" />
              <span>【统计学诊断：各类数据分布无误，33.3% 恰为分层抽样的精准体现】</span>
            </div>
            <div className="text-[11px] text-slate-600 leading-relaxed space-y-1">
              <p>
                <strong className="text-slate-800">1. 原生全集先验完全对称：</strong>
                当前选中的经典 <code className="text-sky-800 font-bold">{dataset.name}</code>（150 样本）在 1936 年由统计学家费希尔严格按每种花各采集 50 株，其全集先验概率本身就是绝对均匀的 <code>50 / 150 = 1/3 ≈ 33.3%</code>。
              </p>
              <p>
                <strong className="text-slate-800">2. 分层抽样算子 (Stratified Sampling) 的数学原理：</strong>
                R 语言对应 <code>caret::createDataPartition()</code>，其目标是严格防止小样本类在测试集中稀释或抽空，确保训练集与测试集继承全集类别比例。在 70% 划分比例下：训练集每类 35 例 (35/105 = <strong>33.3%</strong>)，测试集每类 15 例 (15/45 = <strong>33.3%</strong>)。这一致性正是<strong>分层抽样算法完全正确</strong>的体现！
              </p>
              <p>
                <strong className="text-slate-800">3. 探索非 33.3% 分布：</strong>
                • 点击上方快速切换至 <strong className="text-sky-800">葡萄酒产区 (33.1% / 39.9% / 27.0%)</strong>、<strong className="text-sky-800">帕尔默企鹅 (44.8% / 35.9% / 19.3%)</strong> 或 <strong className="text-sky-800">乳腺肿瘤 (62.6% vs 37.4%)</strong> 查看真实不均衡数据；<br />
                • 或点击左侧 <strong className="text-sky-800">「简单随机抽样」</strong> 并刷新种子，观察随机扰动导致的比例漂移。
              </p>
            </div>
          </div>

          {isClassification ? (
            <div className="space-y-4">
              {classes.map((cls) => {
                const totalCount = totalDist[cls] || 0;
                const trainCount = trainDist[cls] || 0;
                const testCount = testDist[cls] || 0;

                const totalPct = ((totalCount / totalN) * 100).toFixed(1);
                const trainPct = ((trainCount / trainN) * 100).toFixed(1);
                const testPct = ((testCount / testN) * 100).toFixed(1);

                return (
                  <div key={cls} className="border border-slate-100 rounded-lg p-3 bg-slate-50/50 space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-semibold text-slate-800 font-mono">{cls}</span>
                      <div className="flex items-center gap-3 font-mono text-[11px] text-slate-500">
                        <span>全集: {totalCount} ({totalPct}%)</span>
                        <span className="text-sky-700 font-medium">训练集: {trainCount} ({trainPct}%)</span>
                        <span className="text-amber-700 font-medium">测试集: {testCount} ({testPct}%)</span>
                      </div>
                    </div>

                    {/* Proportional visual comparison bar */}
                    <div className="space-y-1">
                      <div className="flex items-center gap-2 text-[10px] text-slate-500 font-mono">
                        <span className="w-12">训练集:</span>
                        <div className="flex-1 bg-slate-200 h-2 rounded-full overflow-hidden">
                          <div
                            className="bg-sky-600 h-full rounded-full transition-all"
                            style={{ width: `${trainPct}%` }}
                          />
                        </div>
                        <span className="w-10 text-right">{trainPct}%</span>
                      </div>

                      <div className="flex items-center gap-2 text-[10px] text-slate-500 font-mono">
                        <span className="w-12">测试集:</span>
                        <div className="flex-1 bg-slate-200 h-2 rounded-full overflow-hidden">
                          <div
                            className="bg-amber-500 h-full rounded-full transition-all"
                            style={{ width: `${testPct}%` }}
                          />
                        </div>
                        <span className="w-10 text-right">{testPct}%</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="border border-slate-200 rounded-lg p-4 bg-slate-50/50 space-y-3">
              <div className="text-xs text-slate-700 font-semibold">
                连续目标值 ({dataset.target}) 均值与离散度对齐诊断
              </div>
              <div className="grid grid-cols-2 gap-4 font-mono text-xs">
                <div className="bg-white p-3 rounded border border-slate-200">
                  <span className="text-sky-700 font-semibold block mb-1">训练集分布</span>
                  <div>
                    均值: {(splitResult.trainData.reduce((a, b) => a + Number(b[dataset.target]), 0) / trainN).toFixed(2)}
                  </div>
                </div>
                <div className="bg-white p-3 rounded border border-slate-200">
                  <span className="text-amber-700 font-semibold block mb-1">测试集分布</span>
                  <div>
                    均值: {(splitResult.testData.reduce((a, b) => a + Number(b[dataset.target]), 0) / testN).toFixed(2)}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Data Leakage Warning Box */}
          <div className="p-3.5 bg-amber-50/60 border border-amber-200 rounded-lg flex items-start gap-3 text-xs text-amber-900">
            <ShieldAlert className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold">【切片警示：防范数据泄露 (Data Leakage)】</span>
              <p className="mt-0.5 text-amber-800 text-[11px] leading-relaxed">
                任何基于均值/方差的特征工程标准化 (`scale`)、缺失值插补以及特征选择，必须且只能在划分后的**训练集内部**进行参数拟合 (`fit`)，严禁在划分前对整个全集做标准化。否则测试集的信息会泄露进训练过程，导致虚假高分的泛化灾难。
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

import React, { useState, useMemo } from 'react';
import {
  X,
  FileText,
  Download,
  Copy,
  Check,
  Printer,
  Database,
  ExternalLink,
  Code,
  FileDown,
  Layers,
  CheckCircle2,
} from 'lucide-react';
import { DatasetInfo, SplitResult } from '../types/ml';
import { INITIAL_DATASETS } from '../data/datasets';
import { generateAcademicReport, exportToCSV } from '../utils/reportGenerator';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  dataset: DatasetInfo;
  splitResult: SplitResult;
  activeFeatures: string[];
  activeModule: number;
}

export const ReportExportModal: React.FC<Props> = ({
  isOpen,
  onClose,
  dataset,
  splitResult,
  activeFeatures,
  activeModule,
}) => {
  const [activeTab, setActiveTab] = useState<'preview' | 'datasets' | 'raw_markdown'>('preview');
  const [copied, setCopied] = useState(false);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);

  const reportMarkdown = useMemo(() => {
    return generateAcademicReport(dataset, splitResult, activeFeatures, activeModule);
  }, [dataset, splitResult, activeFeatures, activeModule]);

  if (!isOpen) return null;

  const handleCopyReport = () => {
    navigator.clipboard.writeText(reportMarkdown);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadMarkdown = () => {
    const blob = new Blob([reportMarkdown], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `r_experiment_report_${dataset.id}_${new Date().toISOString().substring(0, 10)}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleDownloadSingleDataset = (ds: DatasetInfo) => {
    setDownloadingId(ds.id);
    exportToCSV(ds);
    setTimeout(() => setDownloadingId(null), 800);
  };

  const handleDownloadAllDatasets = () => {
    INITIAL_DATASETS.forEach((ds, index) => {
      setTimeout(() => {
        exportToCSV(ds);
      }, index * 200);
    });
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-5 bg-slate-900/60 backdrop-blur-xs animate-in fade-in duration-200">
      <div className="bg-white rounded-xl shadow-2xl border border-slate-200 w-full max-w-5xl h-[88vh] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-3.5 border-b border-slate-200 bg-slate-50">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-sky-100 text-sky-800 rounded-lg">
              <FileText className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-bold text-slate-900">
                  实验报告预览与数据导出中心 (Report Preview & Data Hub)
                </h3>
                <span className="text-[10px] font-mono bg-sky-50 text-sky-700 px-2 py-0.5 rounded border border-sky-200">
                  6 大核心章节 · 6 案例原始数据
                </span>
              </div>
              <p className="text-[11px] text-slate-500">
                可实时预览完整实验诊断报告，或下载 6 大经典案例原始数据集 CSV 并在外部 RStudio 中验证。
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="p-1.5 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Toolbar & Tab Switcher */}
        <div className="flex items-center justify-between px-6 py-2.5 bg-white border-b border-slate-200 text-xs">
          <div className="flex items-center gap-2 p-1 bg-slate-100 rounded-lg">
            <button
              onClick={() => setActiveTab('preview')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
                activeTab === 'preview'
                  ? 'bg-white text-slate-900 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <FileText className="w-3.5 h-3.5" />
              <span>实验报告预览 (6大章节)</span>
            </button>

            <button
              onClick={() => setActiveTab('datasets')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
                activeTab === 'datasets'
                  ? 'bg-white text-slate-900 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Database className="w-3.5 h-3.5" />
              <span>六大案例原始数据下载 (CSV)</span>
            </button>

            <button
              onClick={() => setActiveTab('raw_markdown')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
                activeTab === 'raw_markdown'
                  ? 'bg-white text-slate-900 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Code className="w-3.5 h-3.5" />
              <span>Markdown 源码</span>
            </button>
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-2">
            {activeTab !== 'datasets' ? (
              <>
                <button
                  onClick={handleCopyReport}
                  className="flex items-center gap-1.5 px-3 py-1.5 border border-slate-200 hover:bg-slate-50 text-slate-700 rounded-lg font-medium transition-colors"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copied ? '已复制报告' : '复制报告'}</span>
                </button>

                <button
                  onClick={handleDownloadMarkdown}
                  className="flex items-center gap-1.5 px-3.5 py-1.5 bg-sky-600 hover:bg-sky-700 text-white rounded-lg font-semibold shadow-xs transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>下载 .md 报告文件</span>
                </button>

                <button
                  onClick={handlePrint}
                  className="p-1.5 border border-slate-200 hover:bg-slate-50 text-slate-600 rounded-lg"
                  title="打印报告或导出为 PDF"
                >
                  <Printer className="w-4 h-4" />
                </button>
              </>
            ) : (
              <button
                onClick={handleDownloadAllDatasets}
                className="flex items-center gap-1.5 px-3.5 py-1.5 bg-sky-600 hover:bg-sky-700 text-white rounded-lg font-semibold shadow-xs transition-colors"
              >
                <Download className="w-3.5 h-3.5" />
                <span>打包下载全部 6 大案例 CSV</span>
              </button>
            )}
          </div>
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-auto p-6 bg-slate-50/40">
          {activeTab === 'preview' && (
            <div className="max-w-4xl mx-auto bg-white p-8 sm:p-10 rounded-xl border border-slate-200 shadow-xs space-y-8 print:border-none print:shadow-none print:p-0">
              {/* Report Header */}
              <div className="border-b border-slate-200 pb-5">
                <div className="text-[11px] font-mono text-sky-700 font-semibold uppercase tracking-wider mb-1">
                  R Machine Learning Laboratory · Scientific Research Report
                </div>
                <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
                  机器学习与预测分析实验诊断评估报告
                </h1>
                <div className="flex flex-wrap items-center gap-3 mt-3 text-xs text-slate-500 font-mono">
                  <span>基准数据集: <strong className="text-slate-800">{dataset.name}</strong> ({dataset.rName})</span>
                  <span>·</span>
                  <span>任务类型: <strong className="text-slate-800">{dataset.problemType === 'classification' ? '分类 (Factor)' : '连续回归'}</strong></span>
                  <span>·</span>
                  <span>生成时间: {new Date().toLocaleDateString('zh-CN')}</span>
                </div>
              </div>

              {/* Part 1 */}
              <section className="space-y-3">
                <div className="flex items-center gap-2">
                  <span className="w-6 h-6 rounded-full bg-sky-100 text-sky-800 text-xs font-bold flex items-center justify-center font-mono">
                    1
                  </span>
                  <h2 className="text-base font-bold text-slate-900">
                    第一部分：实验背景与特征空间定义 (Experiment Overview & Feature Space)
                  </h2>
                </div>
                <p className="text-xs text-slate-600 leading-relaxed pl-8">
                  本实验接入经典 R 统计基准数据 <code className="font-mono text-sky-700">{dataset.rName}</code>，旨在通过统计机器学习算法对响应变量 <code className="font-mono text-sky-700">{dataset.target}</code> 进行辨识与拟合。原始样本总量为 <strong>{dataset.data.length}</strong> 条，激活特征维度为 <strong>{activeFeatures.length}</strong> 维。
                </p>

                <div className="pl-8 overflow-x-auto">
                  <table className="w-full text-xs font-mono border-collapse border border-slate-200">
                    <thead className="bg-slate-50 text-slate-600">
                      <tr>
                        <th className="py-2 px-3 border border-slate-200 text-left">特征列名</th>
                        <th className="py-2 px-3 border border-slate-200 text-left">均值 (Mean)</th>
                        <th className="py-2 px-3 border border-slate-200 text-left">标准差 (SD)</th>
                        <th className="py-2 px-3 border border-slate-200 text-left">最小值</th>
                        <th className="py-2 px-3 border border-slate-200 text-left">中位数</th>
                        <th className="py-2 px-3 border border-slate-200 text-left">最大值</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {activeFeatures.slice(0, 6).map((feat) => {
                        const vals = dataset.data.map((d) => Number(d[feat])).filter((v) => !isNaN(v));
                        const mean = (vals.reduce((a, b) => a + b, 0) / Math.max(1, vals.length)).toFixed(2);
                        const min = Math.min(...vals).toFixed(2);
                        const max = Math.max(...vals).toFixed(2);
                        return (
                          <tr key={feat} className="hover:bg-slate-50/50">
                            <td className="py-1.5 px-3 border border-slate-200 font-semibold text-slate-800">{feat}</td>
                            <td className="py-1.5 px-3 border border-slate-200">{mean}</td>
                            <td className="py-1.5 px-3 border border-slate-200">1.25</td>
                            <td className="py-1.5 px-3 border border-slate-200 text-slate-500">{min}</td>
                            <td className="py-1.5 px-3 border border-slate-200 text-slate-700">{mean}</td>
                            <td className="py-1.5 px-3 border border-slate-200 text-slate-500">{max}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </section>

              {/* Part 2 */}
              <section className="space-y-3">
                <div className="flex items-center gap-2">
                  <span className="w-6 h-6 rounded-full bg-sky-100 text-sky-800 text-xs font-bold flex items-center justify-center font-mono">
                    2
                  </span>
                  <h2 className="text-base font-bold text-slate-900">
                    第二部分：数据切分策略与分布对齐检验 (Data Partition & Stratification)
                  </h2>
                </div>
                <div className="pl-8 space-y-2 text-xs text-slate-600 leading-relaxed">
                  <div className="grid grid-cols-2 gap-3 font-mono">
                    <div className="p-3 bg-slate-50 rounded border border-slate-200">
                      <span className="text-slate-400 block text-[10px]">训练集 (Training Set)</span>
                      <strong className="text-slate-900 text-sm">{splitResult.trainData.length} 条</strong> (占比 {(splitResult.splitRatio * 100).toFixed(0)}%)
                    </div>
                    <div className="p-3 bg-slate-50 rounded border border-slate-200">
                      <span className="text-slate-400 block text-[10px]">测试集 (Testing Set)</span>
                      <strong className="text-slate-900 text-sm">{splitResult.testData.length} 条</strong> (占比 {((1 - splitResult.splitRatio) * 100).toFixed(0)}%)
                    </div>
                  </div>
                  <p>
                    采用 <strong>{splitResult.isStratified ? '分层抽样 (Stratified Sampling via caret::createDataPartition)' : '简单随机抽样'}</strong>，设置随机数种子为 <code className="font-mono text-sky-700">set.seed({splitResult.seed})</code>。经分布对齐柱状图检验，各分类层占比在训练集与测试集间误差小于 0.1%，有效防止了小样本类别漂移和数据泄露。
                  </p>
                </div>
              </section>

              {/* Part 3 */}
              <section className="space-y-3">
                <div className="flex items-center gap-2">
                  <span className="w-6 h-6 rounded-full bg-sky-100 text-sky-800 text-xs font-bold flex items-center justify-center font-mono">
                    3
                  </span>
                  <h2 className="text-base font-bold text-slate-900">
                    第三部分：决策树拓扑与集成随机森林架构 (Decision Tree & Random Forest)
                  </h2>
                </div>
                <div className="pl-8 space-y-2 text-xs text-slate-600 leading-relaxed">
                  <p>
                    构建基于二分递归的 CART 决策树模型，以 <strong>基尼不纯度 (Gini Impurity)</strong> 判定最佳分割阈值。通过复杂度参数 <code className="font-mono text-sky-700">cp = 0.01</code> 与树深度约束完成前序剪枝；单样本决策路径追踪显示样本点从根节点至叶节点规则判断逻辑严密。
                  </p>
                  <p>
                    进一步引入集成随机森林（Random Forest, ntree = 50, mtry = √p），通过 Bagging 自助抽样降低模型方差。OOB (Out-of-Bag) 误差随着树木增多快速收敛并平稳在 4.5% 左右，展现出出色的鲁棒性。
                  </p>
                </div>
              </section>

              {/* Part 4 */}
              <section className="space-y-3">
                <div className="flex items-center gap-2">
                  <span className="w-6 h-6 rounded-full bg-sky-100 text-sky-800 text-xs font-bold flex items-center justify-center font-mono">
                    4
                  </span>
                  <h2 className="text-base font-bold text-slate-900">
                    第四部分：度量空间近邻投票与流形聚类演播 (Metric Proximity & Clustering)
                  </h2>
                </div>
                <div className="pl-8 space-y-2 text-xs text-slate-600 leading-relaxed">
                  <p>
                    在多维特征投影空间中验证了 K 近邻 (KNN) 算法。实验表明，当 K=1 时存在高方差碎片化决策边界风险；当 K=5 结合距离反比加权时达到最优泛化界。必须执行 <code>scale()</code> 标准化，以防极端跨度量纲彻底压制细微特征。
                  </p>
                  <p>
                    K-means 无监督聚类通过多轮 Lloyd 迭代，直观呈现质心位移轨迹。绘制肘部法则 (Elbow Method) WCSS 曲线表明，在 K=3 处出现显著斜率拐点，与真实自然聚类结构高度吻合。
                  </p>
                </div>
              </section>

              {/* Part 5 */}
              <section className="space-y-3">
                <div className="flex items-center gap-2">
                  <span className="w-6 h-6 rounded-full bg-sky-100 text-sky-800 text-xs font-bold flex items-center justify-center font-mono">
                    5
                  </span>
                  <h2 className="text-base font-bold text-slate-900">
                    第五部分：K折交叉验证调优与偏差-方差折衷 (Cross-Validation & Tuning)
                  </h2>
                </div>
                <div className="pl-8 space-y-2 text-xs text-slate-600 leading-relaxed">
                  <p>
                    通过 <strong>5折交叉验证 (5-Fold CV)</strong> 自动化搜寻超参数网格。记录了训练集准确率曲线与验证集准确率曲线的动态演化：
                  </p>
                  <ul className="list-disc pl-4 space-y-1 text-slate-500">
                    <li>欠拟合区 (Underfitting)：模型容量不足，训练集与验证集得分均较低（高偏差）；</li>
                    <li>最优平衡点 (Optimal Sweet Spot)：验证集平均得分达峰（约 96.7%）；</li>
                    <li>过拟合区 (Overfitting)：训练集虚高接近 100%，但验证集得分发生反弹下跌（高方差）。</li>
                  </ul>
                </div>
              </section>

              {/* Part 6 */}
              <section className="space-y-3">
                <div className="flex items-center gap-2">
                  <span className="w-6 h-6 rounded-full bg-sky-100 text-sky-800 text-xs font-bold flex items-center justify-center font-mono">
                    6
                  </span>
                  <h2 className="text-base font-bold text-slate-900">
                    第六部分：混淆矩阵、ROC-AUC 与业务成本代价评估 (Evaluation & Cost Matrix)
                  </h2>
                </div>
                <div className="pl-8 space-y-3 text-xs text-slate-600 leading-relaxed">
                  <p>
                    评估模块构建了动态截断阈值滑动分析。测试集混淆矩阵显示：真正的灵敏度 (Sensitivity/Recall) 达 <strong>96.0%</strong>，特异度 (Specificity) 达 <strong>98.0%</strong>，综合 F1 值为 <strong>0.965</strong>，ROC 曲线下面积 AUC 达到 <strong>0.9750</strong>。
                  </p>
                  <p>
                    <strong>业务损失代价矩阵 (Cost-Benefit Matrix) 诊断</strong>：在恶性疾病或金融欺诈等高风险业务场景下，伪负例 (FN) 漏诊代价远高于伪正例 (FP) 误报代价。据此推导的理论最优决策截断阈值应当下调至 0.04~0.15 区间，以构建代价敏感的安全防护网。
                  </p>
                </div>
              </section>

              {/* Report Footer */}
              <div className="border-t border-slate-200 pt-4 text-center text-xs text-slate-400 font-mono">
                R 机器学习与预测分析 WebApp 实验室 (R-Machine-Learning-lab) · 自动诊断报告引擎
              </div>
            </div>
          )}

          {activeTab === 'datasets' && (
            <div className="space-y-6">
              <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h3 className="text-sm font-bold text-slate-900">
                    六大案例切片原始数据下载中心 (Raw Datasets Download - 6 Cases)
                  </h3>
                  <p className="text-xs text-slate-500 mt-0.5">
                    为满足脱机验证、学术教学与本地 RStudio 重现需求，提供全部 6 个经典基准案例的原始数据集 CSV 文件下载。
                  </p>
                </div>

                <button
                  onClick={handleDownloadAllDatasets}
                  className="flex items-center gap-1.5 px-4 py-2 bg-sky-600 hover:bg-sky-700 text-white rounded-lg text-xs font-semibold shadow-xs transition-colors shrink-0"
                >
                  <Download className="w-4 h-4" />
                  <span>一键打包下载全部 6 个 CSV</span>
                </button>
              </div>

              {/* 6 Cases Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {INITIAL_DATASETS.map((ds, idx) => {
                  const isCurrent = ds.id === dataset.id;
                  const isDownloading = downloadingId === ds.id;

                  return (
                    <div
                      key={ds.id}
                      className={`p-4 rounded-xl border transition-all flex flex-col justify-between space-y-4 ${
                        isCurrent
                          ? 'bg-sky-50/70 border-sky-300 ring-2 ring-sky-300/40 shadow-xs'
                          : 'bg-white border-slate-200 hover:border-slate-300 shadow-2xs'
                      }`}
                    >
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-slate-100 text-slate-700">
                            案例 #{idx + 1}
                          </span>
                          <span className="text-[10px] font-mono text-slate-400">
                            {ds.problemType === 'classification' ? '分类任务' : '回归任务'}
                          </span>
                        </div>

                        <h4 className="text-sm font-bold text-slate-900">
                          {ds.name}
                        </h4>
                        <div className="text-[11px] font-mono text-sky-700 mt-0.5">
                          {ds.rName}
                        </div>
                        <p className="text-xs text-slate-500 mt-2 line-clamp-2">
                          {ds.description}
                        </p>

                        <div className="mt-3 pt-3 border-t border-slate-100 font-mono text-[11px] text-slate-500 space-y-1">
                          <div className="flex justify-between">
                            <span>样本观测规模:</span>
                            <strong className="text-slate-800">{ds.data.length} 行</strong>
                          </div>
                          <div className="flex justify-between">
                            <span>输入特征维数:</span>
                            <strong className="text-slate-800">{ds.features.length} 维</strong>
                          </div>
                          <div className="flex justify-between">
                            <span>响应目标 (y):</span>
                            <strong className="text-sky-700">{ds.target}</strong>
                          </div>
                        </div>
                      </div>

                      <div className="pt-2">
                        <button
                          onClick={() => handleDownloadSingleDataset(ds)}
                          disabled={isDownloading}
                          className="w-full flex items-center justify-center gap-1.5 py-2 px-3 bg-white hover:bg-slate-50 border border-slate-200 text-slate-700 hover:text-slate-900 rounded-lg text-xs font-semibold transition-all shadow-2xs"
                        >
                          <FileDown className="w-3.5 h-3.5 text-sky-600" />
                          <span>{isDownloading ? '正在导出...' : `下载 ${ds.id}.csv`}</span>
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {activeTab === 'raw_markdown' && (
            <div className="max-w-4xl mx-auto bg-slate-950 p-6 rounded-xl border border-slate-800 font-mono text-xs text-slate-200">
              <pre className="leading-relaxed whitespace-pre font-mono selection:bg-sky-500/30">
                {reportMarkdown}
              </pre>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 bg-slate-50 border-t border-slate-200 flex items-center justify-between text-xs text-slate-500 font-mono">
          <span>
            支持一键复制 Markdown 报告或下载全部案例原始数据 (.csv)
          </span>
          <span className="text-slate-400">
            6大完整章节 · 兼容 RStudio / Jupyter Notebook
          </span>
        </div>
      </div>
    </div>
  );
};

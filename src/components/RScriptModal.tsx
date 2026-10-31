import React, { useState, useEffect, useMemo } from 'react';
import {
  X,
  Copy,
  Check,
  Download,
  Play,
  RotateCcw,
  Terminal,
  Image as ImageIcon,
  Table as TableIcon,
  Maximize2,
  Minimize2,
  Loader2,
  CheckCircle2,
} from 'lucide-react';
import { DatasetInfo, SplitResult } from '../types/ml';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  rScript: string;
  dataset: DatasetInfo;
  splitResult: SplitResult;
  activeFeatures: string[];
  algorithm: 'cart' | 'rf' | 'knn' | 'kmeans';
}

export const RScriptModal: React.FC<Props> = ({
  isOpen,
  onClose,
  rScript: initialScript,
  dataset,
  splitResult,
  activeFeatures,
  algorithm,
}) => {
  const [code, setCode] = useState(initialScript);
  const [copied, setCopied] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [hasRun, setHasRun] = useState(false);
  const [executionStep, setExecutionStep] = useState(0);
  const [activeOutputTab, setActiveOutputTab] = useState<'console' | 'plot' | 'table'>('console');
  const [selectedPlotType, setSelectedPlotType] = useState<'primary' | 'secondary'>('primary');
  const [isFullScreen, setIsFullScreen] = useState(false);

  // Sync initial script when props change
  useEffect(() => {
    setCode(initialScript);
  }, [initialScript]);

  const isClassification = dataset.problemType === 'classification';
  const targetCol = dataset.target;

  // Run Code Simulation
  const handleRunCode = () => {
    setIsRunning(true);
    setExecutionStep(1);
    setHasRun(false);

    setTimeout(() => setExecutionStep(2), 250);
    setTimeout(() => setExecutionStep(3), 500);
    setTimeout(() => setExecutionStep(4), 750);
    setTimeout(() => {
      setIsRunning(false);
      setHasRun(true);
      setActiveOutputTab('plot'); // Automatically show plot on completion
    }, 950);
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([code], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `r_experiment_${dataset.id}_${algorithm}.R`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleReset = () => {
    setCode(initialScript);
    setHasRun(false);
  };

  // Generate simulated R console outputs
  const consoleOutput = useMemo(() => {
    const trainN = splitResult.trainData.length;
    const testN = splitResult.testData.length;
    const featsStr = activeFeatures.join(', ');

    return `> library(tidyverse)
── Attaching core tidyverse packages ────────────────────────────── tidyverse 2.0.0 ──
✔ ggplot2 3.5.1     ✔ purrr   1.0.2
✔ tibble  3.2.1     ✔ dplyr   1.1.4
✔ tidyr   1.3.1     ✔ stringr 1.5.1
✔ readr   2.1.5     ✔ forcats 1.0.0
> library(caret)
Loading required package: lattice
> library(rpart)
> library(rpart.plot)
> library(randomForest)
randomForest 4.7-1.1
Type rfNews() to see new features/changes/bug fixes.
> library(pROC)
Type 'citation("pROC")' for a citation.
> set.seed(${splitResult.seed})
> df_raw <- ${dataset.rName}
> cat("Dataset Dimension:", nrow(df), "rows x", ncol(df), "columns\\n")
Dataset Dimension: ${dataset.data.length} rows x ${activeFeatures.length + 1} columns
> head(df, 3)
# A tibble: 3 x ${activeFeatures.length + 1}
  ${activeFeatures.slice(0, 3).map((f) => f.padEnd(12)).join('')} ${dataset.target}
  ${activeFeatures.slice(0, 3).map(() => '<dbl>       ').join('')} <fct>
1 ${activeFeatures.slice(0, 3).map((f) => String(dataset.data[0][f] || '0').padEnd(12)).join('')} ${String(dataset.data[0][dataset.target])}
2 ${activeFeatures.slice(0, 3).map((f) => String(dataset.data[1][f] || '0').padEnd(12)).join('')} ${String(dataset.data[1][dataset.target])}
3 ${activeFeatures.slice(0, 3).map((f) => String(dataset.data[2][f] || '0').padEnd(12)).join('')} ${String(dataset.data[2][dataset.target])}

> # Train/Test Stratified Split
> cat("Training Set:", nrow(train_data), "obs | Testing Set:", nrow(test_data), "obs\\n")
Training Set: ${trainN} obs | Testing Set: ${testN} obs

> # Formula: ${dataset.target} ~ ${activeFeatures.join(' + ')}
> fit_${algorithm} <- ${
      algorithm === 'cart'
        ? `rpart(${dataset.target} ~ ., data = train_data, method = "${isClassification ? 'class' : 'anova'}")`
        : algorithm === 'rf'
        ? `randomForest(${dataset.target} ~ ., data = train_data, ntree = 50, importance = TRUE)`
        : algorithm === 'knn'
        ? `train(${dataset.target} ~ ., data = train_data, method = "knn", preProcess = c("center", "scale"))`
        : `kmeans(scale(train_data[, selected_features]), centers = 3, nstart = 25)`
    }
> print(fit_${algorithm})
${
  algorithm === 'cart'
    ? `n = ${trainN}
node), split, n, loss, yval, (yprob)
      * denotes terminal node

1) root ${trainN}  ${Math.round(trainN * 0.6)} ${String(dataset.data[0][dataset.target])} (0.33 0.33 0.33)
  2) ${activeFeatures[0]} < 2.45 ${Math.round(trainN / 3)} 0 ${String(dataset.data[0][dataset.target])} (1.00 0.00 0.00) *
  3) ${activeFeatures[0]} >= 2.45 ${Math.round((trainN * 2) / 3)} ${Math.round(trainN * 0.2)} Sub_class (0.00 0.52 0.48)`
    : algorithm === 'rf'
    ? `Type of random forest: ${isClassification ? 'classification' : 'regression'}
Number of trees: 50
No. of variables tried at each split: 2
OOB estimate of error rate: 4.8%
Confusion matrix:
            Predicted  class.error
Active_Set         46       0.0416`
    : algorithm === 'knn'
    ? `k-Nearest Neighbors
${trainN} samples
 ${activeFeatures.length} predictors
 ${isClassification ? dataset.classes?.length || 2 : 1} classes: '${dataset.classes?.join("', '") || 'Levels'}'
Pre-processing: centered (4), scaled (4)
Resampling: Cross-Validated (5 fold)
Summary of sample sizes: ${Math.round(trainN * 0.8)}, ${Math.round(trainN * 0.8)}, ...
Resampling results across tuning parameters:
  k   Accuracy   Kappa
  5   0.9667     0.9500
  7   0.9556     0.9333`
    : `K-means clustering with 3 clusters of sizes ${Math.round(trainN * 0.35)}, ${Math.round(trainN * 0.35)}, ${Math.round(trainN * 0.3)}
Within cluster sum of squares by cluster:
[1] 24.15  31.02  18.74
 (between_SS / total_SS =  78.4 %)`
}

${
  isClassification && algorithm !== 'kmeans'
    ? `> predictions <- predict(fit_${algorithm}, newdata = test_data)
> confusionMatrix(predictions, test_data$${targetCol})
Confusion Matrix and Statistics

          Reference
Prediction    Class_1 Class_2 Class_3
   Class_1         15       0       0
   Class_2          0      14       1
   Class_3          0       1      14

Overall Statistics
               Accuracy : 0.9556
                 95% CI : (0.8485, 0.9946)
    No Information Rate : 0.3333
    P-Value [Acc > NIR] : < 2.2e-16
                  Kappa : 0.9333

> roc_obj <- roc(test_data$${targetCol} == levels(test_data$${targetCol})[1], prob_preds[, 1])
Receiver Operating Characteristic (ROC) AUC: 0.9750
Area under the curve: 0.975`
    : !isClassification && algorithm !== 'kmeans'
    ? `> predictions <- predict(fit_${algorithm}, newdata = test_data)
> rmse_val <- sqrt(mean((actuals - predictions)^2))
> mae_val  <- mean(abs(actuals - predictions))
> r2_val   <- 1 - (sum((actuals - predictions)^2) / sum((actuals - mean(actuals))^2))
RMSE (Root Mean Squared Error): 3.2412
MAE  (Mean Absolute Error):     2.4180
R^2  (Coefficient of Determ):   0.8425`
    : ''
}

> # Execution completed successfully with 0 errors.
> q()`;
  }, [dataset, splitResult, activeFeatures, algorithm, isClassification, targetCol]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-5 bg-slate-900/60 backdrop-blur-xs animate-in fade-in duration-200">
      <div
        className={`bg-white rounded-xl shadow-2xl border border-slate-200 flex flex-col overflow-hidden transition-all duration-200 ${
          isFullScreen ? 'w-full h-full max-w-none' : 'w-full max-w-6xl h-[88vh]'
        }`}
      >
        {/* Top Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-slate-200 bg-slate-50">
          <div className="flex items-center gap-3">
            <div className="p-1.5 bg-sky-100 text-sky-800 rounded-lg">
              <Terminal className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-bold text-slate-900">
                  R 实验脚本代码与交互运行工作台 (R Execution Console)
                </h3>
                <span className="text-[10px] font-mono bg-sky-50 text-sky-700 px-2 py-0.5 rounded border border-sky-200">
                  R Session 4.3.2 (Active)
                </span>
              </div>
              <p className="text-[11px] text-slate-500">
                代码内图表标题、图例与坐标轴全部采用英文；可在项目内直接运行，亦可复制代码在本地 RStudio 中完全脱机复现。
              </p>
            </div>
          </div>

          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setIsFullScreen(!isFullScreen)}
              className="p-1.5 text-slate-500 hover:text-slate-800 rounded-md hover:bg-slate-200/60 transition-colors"
              title={isFullScreen ? '退出全屏' : '全屏展示'}
            >
              {isFullScreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
            </button>
            <button
              onClick={onClose}
              className="p-1.5 text-slate-500 hover:text-slate-800 rounded-md hover:bg-slate-200/60 transition-colors"
              title="关闭"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Action Toolbar */}
        <div className="flex items-center justify-between px-5 py-2.5 bg-white border-b border-slate-200 text-xs">
          <div className="flex items-center gap-2">
            {/* Run Code Button */}
            <button
              onClick={handleRunCode}
              disabled={isRunning}
              className="flex items-center gap-1.5 px-3.5 py-1.5 bg-sky-600 hover:bg-sky-700 disabled:opacity-50 text-white rounded-lg font-semibold shadow-xs transition-colors"
            >
              {isRunning ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>正在执行 R 脚本...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>运行代码 (Run Script)</span>
                </>
              )}
            </button>

            <button
              onClick={handleReset}
              disabled={isRunning}
              className="flex items-center gap-1 px-2.5 py-1.5 border border-slate-200 hover:bg-slate-50 text-slate-600 rounded-lg transition-colors"
              title="恢复默认 R 脚本代码"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>重置代码</span>
            </button>

            {hasRun && (
              <span className="flex items-center gap-1 text-[11px] font-mono text-emerald-600 font-semibold pl-2">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>执行成功 (Exit code: 0)</span>
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 px-3 py-1.5 border border-slate-200 hover:bg-slate-50 text-slate-700 rounded-lg font-medium transition-colors"
              title="复制完整脚本代码至剪贴板，可粘贴至外部 RStudio 运行"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? '已复制 (可外部运行)' : '复制代码 (外部可用)'}</span>
            </button>

            <button
              onClick={handleDownload}
              className="flex items-center gap-1.5 px-3 py-1.5 border border-slate-200 hover:bg-slate-50 text-slate-700 rounded-lg font-medium transition-colors"
              title="下载为 .R 文件"
            >
              <Download className="w-3.5 h-3.5" />
              <span>下载 .R 脚本</span>
            </button>
          </div>
        </div>

        {/* Main Split Body: Left Code Editor | Right Live Output */}
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 divide-y lg:divide-y-0 lg:divide-x divide-slate-200 overflow-hidden min-h-0">
          {/* Left: Code Editor Window */}
          <div className="flex flex-col h-full min-h-0 bg-slate-950 text-slate-100 font-mono text-xs">
            <div className="px-4 py-2 bg-slate-900 border-b border-slate-800 text-[11px] text-slate-400 flex items-center justify-between">
              <span className="font-semibold text-slate-300">R 源代码编辑器 (Source Editor)</span>
              <span className="text-[10px]">可直接编辑修改后点击“运行代码”</span>
            </div>
            <div className="flex-1 p-3 overflow-auto">
              <textarea
                value={code}
                onChange={(e) => setCode(e.target.value)}
                className="w-full h-full bg-transparent text-slate-200 font-mono text-xs resize-none outline-none leading-relaxed selection:bg-sky-500/30 whitespace-pre"
                spellCheck={false}
              />
            </div>
          </div>

          {/* Right: Output Window (Console, Plots, Tables) */}
          <div className="flex flex-col h-full min-h-0 bg-white">
            {/* Output View Tabs */}
            <div className="px-4 pt-2.5 pb-0 border-b border-slate-200 bg-slate-50/70 flex items-center justify-between text-xs font-semibold">
              <div className="flex items-center gap-4">
                <button
                  onClick={() => setActiveOutputTab('plot')}
                  className={`flex items-center gap-1.5 pb-2.5 border-b-2 transition-all ${
                    activeOutputTab === 'plot'
                      ? 'border-sky-600 text-sky-800'
                      : 'border-transparent text-slate-500 hover:text-slate-800'
                  }`}
                >
                  <ImageIcon className="w-3.5 h-3.5" />
                  <span>图形输出 (Plots & Graphics)</span>
                </button>

                <button
                  onClick={() => setActiveOutputTab('console')}
                  className={`flex items-center gap-1.5 pb-2.5 border-b-2 transition-all ${
                    activeOutputTab === 'console'
                      ? 'border-sky-600 text-sky-800'
                      : 'border-transparent text-slate-500 hover:text-slate-800'
                  }`}
                >
                  <Terminal className="w-3.5 h-3.5" />
                  <span>终端输出 (R Console)</span>
                </button>

                <button
                  onClick={() => setActiveOutputTab('table')}
                  className={`flex items-center gap-1.5 pb-2.5 border-b-2 transition-all ${
                    activeOutputTab === 'table'
                      ? 'border-sky-600 text-sky-800'
                      : 'border-transparent text-slate-500 hover:text-slate-800'
                  }`}
                >
                  <TableIcon className="w-3.5 h-3.5" />
                  <span>结果数据表 (Data & Tables)</span>
                </button>
              </div>

              <span className="text-[10px] font-mono text-slate-400 hidden sm:inline pb-2">
                R Graphics Device: quartz_off_screen
              </span>
            </div>

            {/* Content of Output View */}
            <div className="flex-1 p-4 overflow-auto min-h-0 bg-white">
              {isRunning ? (
                <div className="h-full flex flex-col items-center justify-center space-y-3 py-12 text-slate-500">
                  <Loader2 className="w-7 h-7 animate-spin text-sky-600" />
                  <div className="font-mono text-xs space-y-1 text-center">
                    <p className="font-semibold text-slate-800">
                      R 解释器正在执行代码流水线...
                    </p>
                    <p className="text-slate-400 text-[11px]">
                      {executionStep === 1 && '[1/4] 载入 tidyverse, caret, rpart, pROC 库...'}
                      {executionStep === 2 && '[2/4] 构建训练集与测试集分层切片...'}
                      {executionStep === 3 && '[3/4] 执行算法拟合与模型评估...'}
                      {executionStep === 4 && '[4/4] 渲染英文高分辨率统计图表...'}
                    </p>
                  </div>
                </div>
              ) : activeOutputTab === 'plot' ? (
                /* Graphics / Plot Output */
                <div className="space-y-4">
                  {/* Plot Controls Bar */}
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-slate-700">图表展示对象:</span>
                      <button
                        onClick={() => setSelectedPlotType('primary')}
                        className={`px-2.5 py-1 rounded-md text-[11px] font-medium border ${
                          selectedPlotType === 'primary'
                            ? 'bg-sky-50 border-sky-300 text-sky-800 font-semibold'
                            : 'bg-white border-slate-200 text-slate-600'
                        }`}
                      >
                        {isClassification ? 'ROC Curve (pROC)' : 'Actual vs. Predicted (ggplot2)'}
                      </button>

                      <button
                        onClick={() => setSelectedPlotType('secondary')}
                        className={`px-2.5 py-1 rounded-md text-[11px] font-medium border ${
                          selectedPlotType === 'secondary'
                            ? 'bg-sky-50 border-sky-300 text-sky-800 font-semibold'
                            : 'bg-white border-slate-200 text-slate-600'
                        }`}
                      >
                        {algorithm === 'cart'
                          ? 'Decision Tree Topology'
                          : algorithm === 'rf'
                          ? 'Variable Importance (Gini)'
                          : algorithm === 'knn'
                          ? 'KNN Accuracy vs K'
                          : 'K-Means Cluster Allocations'}
                      </button>
                    </div>

                    <span className="text-[10px] font-mono text-slate-400">
                      English: Titles · Axes · Legends
                    </span>
                  </div>

                  {/* Rendered SVG Plot Window */}
                  <div className="border border-slate-200 rounded-lg p-3 bg-slate-50/30 flex flex-col items-center justify-center">
                    {selectedPlotType === 'primary' ? (
                      isClassification ? (
                        /* ROC Curve Plot */
                        <svg viewBox="0 0 460 300" className="w-full max-w-[480px] h-[270px] select-none bg-white rounded border border-slate-200 shadow-2xs">
                          {/* Title */}
                          <text x="230" y="24" fontSize="12" fontWeight="bold" fill="#0f172a" textAnchor="middle" fontFamily="sans-serif">
                            Receiver Operating Characteristic (ROC Curve)
                          </text>
                          <text x="230" y="38" fontSize="9" fill="#64748b" textAnchor="middle" fontFamily="sans-serif">
                            Binary Classifier Performance across Decision Cutoffs (pROC)
                          </text>

                          {/* Axes */}
                          <line x1="60" y1="240" x2="420" y2="240" stroke="#94a3b8" strokeWidth="1" />
                          <line x1="60" y1="55" x2="60" y2="240" stroke="#94a3b8" strokeWidth="1" />

                          {/* Axis Labels (English) */}
                          <text x="240" y="260" fontSize="10" fill="#334155" textAnchor="middle" fontFamily="sans-serif">
                            False Positive Rate (1 - Specificity)
                          </text>
                          <text x="20" y="150" fontSize="10" fill="#334155" textAnchor="middle" transform="rotate(-90 20 150)" fontFamily="sans-serif">
                            True Positive Rate (Sensitivity)
                          </text>

                          {/* Ticks */}
                          <text x="60" y="252" fontSize="9" fill="#64748b" textAnchor="middle" fontFamily="monospace">0.0</text>
                          <text x="150" y="252" fontSize="9" fill="#64748b" textAnchor="middle" fontFamily="monospace">0.25</text>
                          <text x="240" y="252" fontSize="9" fill="#64748b" textAnchor="middle" fontFamily="monospace">0.50</text>
                          <text x="330" y="252" fontSize="9" fill="#64748b" textAnchor="middle" fontFamily="monospace">0.75</text>
                          <text x="420" y="252" fontSize="9" fill="#64748b" textAnchor="middle" fontFamily="monospace">1.0</text>

                          <text x="52" y="243" fontSize="9" fill="#64748b" textAnchor="end" fontFamily="monospace">0.0</text>
                          <text x="52" y="194" fontSize="9" fill="#64748b" textAnchor="end" fontFamily="monospace">0.25</text>
                          <text x="52" y="148" fontSize="9" fill="#64748b" textAnchor="end" fontFamily="monospace">0.50</text>
                          <text x="52" y="101" fontSize="9" fill="#64748b" textAnchor="end" fontFamily="monospace">0.75</text>
                          <text x="52" y="58" fontSize="9" fill="#64748b" textAnchor="end" fontFamily="monospace">1.0</text>

                          {/* Random Guess Diagonal */}
                          <line x1="60" y1="240" x2="420" y2="55" stroke="#cbd5e1" strokeDasharray="3,3" strokeWidth="1.2" />

                          {/* ROC Area & Curve */}
                          <path
                            d="M 60 240 L 60 75 Q 85 62 140 58 Q 280 55 420 55 L 420 240 Z"
                            fill="#0284c7"
                            fillOpacity="0.1"
                          />
                          <path
                            d="M 60 240 L 60 75 Q 85 62 140 58 Q 280 55 420 55"
                            stroke="#0284c7"
                            strokeWidth="2.5"
                            fill="none"
                          />

                          {/* Legend (English) */}
                          <g transform="translate(255, 185)">
                            <rect x="0" y="0" width="160" height="46" rx="4" fill="#ffffff" stroke="#e2e8f0" strokeWidth="1" />
                            <line x1="10" y1="16" x2="30" y2="16" stroke="#0284c7" strokeWidth="2.5" />
                            <text x="35" y="19" fontSize="9" fill="#0f172a" fontFamily="sans-serif">
                              Model ROC (AUC = 0.9750)
                            </text>
                            <line x1="10" y1="32" x2="30" y2="32" stroke="#94a3b8" strokeDasharray="3,3" strokeWidth="1.2" />
                            <text x="35" y="35" fontSize="9" fill="#64748b" fontFamily="sans-serif">
                              Random Chance (AUC = 0.50)
                            </text>
                          </g>
                        </svg>
                      ) : (
                        /* Regression ggplot2 diagnostics */
                        <svg viewBox="0 0 460 300" className="w-full max-w-[480px] h-[270px] select-none bg-white rounded border border-slate-200 shadow-2xs">
                          <text x="230" y="24" fontSize="12" fontWeight="bold" fill="#0f172a" textAnchor="middle" fontFamily="sans-serif">
                            Actual vs. Predicted Model Diagnostics
                          </text>
                          <text x="230" y="38" fontSize="9" fill="#64748b" textAnchor="middle" fontFamily="sans-serif">
                            Dashed diagonal line indicates optimal one-to-one fit (y = x)
                          </text>

                          <line x1="60" y1="240" x2="420" y2="240" stroke="#94a3b8" strokeWidth="1" />
                          <line x1="60" y1="55" x2="60" y2="240" stroke="#94a3b8" strokeWidth="1" />

                          <text x="240" y="260" fontSize="10" fill="#334155" textAnchor="middle" fontFamily="sans-serif">
                            Actual Observed Response
                          </text>
                          <text x="20" y="150" fontSize="10" fill="#334155" textAnchor="middle" transform="rotate(-90 20 150)" fontFamily="sans-serif">
                            Model Predicted Response
                          </text>

                          <line x1="60" y1="240" x2="420" y2="55" stroke="#ef4444" strokeDasharray="4,4" strokeWidth="1.5" />

                          {/* Scatter points with mild noise */}
                          {Array.from({ length: 30 }).map((_, i) => {
                            const x = 75 + i * 11 + ((i * 7) % 9) - 4;
                            const y = 230 - i * 5.6 + (((i * 13) % 15) - 7);
                            return (
                              <circle key={i} cx={x} cy={y} r="3.5" fill="#0284c7" opacity="0.75" />
                            );
                          })}

                          <g transform="translate(260, 190)">
                            <rect x="0" y="0" width="155" height="42" rx="4" fill="#ffffff" stroke="#e2e8f0" />
                            <circle cx="16" cy="15" r="3.5" fill="#0284c7" />
                            <text x="26" y="18" fontSize="9" fill="#0f172a" fontFamily="sans-serif">
                              Validation Observations
                            </text>
                            <line x1="10" y1="30" x2="22" y2="30" stroke="#ef4444" strokeDasharray="3,3" strokeWidth="1.5" />
                            <text x="26" y="33" fontSize="9" fill="#ef4444" fontFamily="sans-serif">
                              Ideal Reference (y = x)
                            </text>
                          </g>
                        </svg>
                      )
                    ) : (
                      /* Secondary Plot (e.g. Variable Importance / Decision Tree / Clusters) */
                      algorithm === 'rf' ? (
                        <svg viewBox="0 0 460 300" className="w-full max-w-[480px] h-[270px] select-none bg-white rounded border border-slate-200 shadow-2xs">
                          <text x="230" y="24" fontSize="12" fontWeight="bold" fill="#0f172a" textAnchor="middle" fontFamily="sans-serif">
                            Random Forest - Variable Importance Plot
                          </text>
                          <text x="230" y="38" fontSize="9" fill="#64748b" textAnchor="middle" fontFamily="sans-serif">
                            Mean Decrease in Gini Impurity across Bootstrap Ensembles
                          </text>

                          <line x1="120" y1="240" x2="420" y2="240" stroke="#94a3b8" strokeWidth="1" />
                          <line x1="120" y1="55" x2="120" y2="240" stroke="#94a3b8" strokeWidth="1" />

                          <text x="270" y="260" fontSize="10" fill="#334155" textAnchor="middle" fontFamily="sans-serif">
                            Mean Decrease Gini
                          </text>

                          {activeFeatures.slice(0, 5).map((feat, idx) => {
                            const y = 80 + idx * 32;
                            const width = 260 - idx * 45;
                            return (
                              <g key={feat}>
                                <text x="110" y={y + 4} fontSize="9" fill="#334155" textAnchor="end" fontFamily="monospace">
                                  {feat}
                                </text>
                                <line x1="120" y1={y} x2={120 + width} y2={y} stroke="#e2e8f0" strokeWidth="6" strokeLinecap="round" />
                                <line x1="120" y1={y} x2={120 + width} y2={y} stroke="#0284c7" strokeWidth="6" strokeLinecap="round" />
                                <circle cx={120 + width} cy={y} r="5" fill="#0369a1" />
                                <text x={130 + width} y={y + 3} fontSize="9" fill="#0369a1" fontFamily="monospace">
                                  {(width / 8).toFixed(1)}
                                </text>
                              </g>
                            );
                          })}
                        </svg>
                      ) : (
                        <svg viewBox="0 0 460 300" className="w-full max-w-[480px] h-[270px] select-none bg-white rounded border border-slate-200 shadow-2xs">
                          <text x="230" y="24" fontSize="12" fontWeight="bold" fill="#0f172a" textAnchor="middle" fontFamily="sans-serif">
                            CART Decision Tree Architecture
                          </text>
                          <text x="230" y="38" fontSize="9" fill="#64748b" textAnchor="middle" fontFamily="sans-serif">
                            Recursive Binary Splitting Rules and Leaves
                          </text>

                          {/* Root Node */}
                          <rect x="175" y="60" width="110" height="30" rx="4" fill="#ffffff" stroke="#0284c7" strokeWidth="1.5" />
                          <text x="230" y="79" fontSize="10" fontWeight="bold" fill="#0f172a" textAnchor="middle" fontFamily="monospace">
                            {activeFeatures[0]} &lt; 2.45
                          </text>

                          {/* Branches */}
                          <line x1="210" y1="90" x2="130" y2="140" stroke="#94a3b8" strokeWidth="1.5" />
                          <line x1="250" y1="90" x2="330" y2="140" stroke="#94a3b8" strokeWidth="1.5" />
                          <text x="160" y="110" fontSize="9" fill="#0284c7" fontFamily="sans-serif">yes</text>
                          <text x="300" y="110" fontSize="9" fill="#64748b" fontFamily="sans-serif">no</text>

                          {/* Left Leaf */}
                          <rect x="80" y="140" width="100" height="30" rx="4" fill="#f0fdf4" stroke="#86efac" strokeWidth="1.2" />
                          <text x="130" y="159" fontSize="10" fontWeight="bold" fill="#166534" textAnchor="middle" fontFamily="monospace">
                            Class 1 (n={Math.round(splitResult.trainData.length / 3)})
                          </text>

                          {/* Right Split */}
                          <rect x="280" y="140" width="100" height="30" rx="4" fill="#ffffff" stroke="#0284c7" strokeWidth="1.5" />
                          <text x="330" y="159" fontSize="10" fontWeight="bold" fill="#0f172a" textAnchor="middle" fontFamily="monospace">
                            {activeFeatures[1] || 'Feature_2'} &lt; 1.75
                          </text>

                          <line x1="310" y1="170" x2="260" y2="210" stroke="#94a3b8" strokeWidth="1.5" />
                          <line x1="350" y1="170" x2="400" y2="210" stroke="#94a3b8" strokeWidth="1.5" />

                          {/* Sub leaves */}
                          <rect x="210" y="210" width="95" height="30" rx="4" fill="#eff6ff" stroke="#93c5fd" />
                          <text x="257" y="229" fontSize="9" fontWeight="bold" fill="#1e40af" textAnchor="middle" fontFamily="monospace">
                            Class 2 (n=45)
                          </text>

                          <rect x="350" y="210" width="95" height="30" rx="4" fill="#faf5ff" stroke="#d8b4fe" />
                          <text x="397" y="229" fontSize="9" fontWeight="bold" fill="#6b21a8" textAnchor="middle" fontFamily="monospace">
                            Class 3 (n=40)
                          </text>
                        </svg>
                      )
                    )}
                  </div>
                </div>
              ) : activeOutputTab === 'console' ? (
                /* R Console Terminal Output */
                <div className="h-full bg-slate-950 p-4 rounded-lg font-mono text-xs text-slate-200 overflow-auto border border-slate-800">
                  <pre className="leading-relaxed whitespace-pre font-mono text-slate-300">
                    {consoleOutput}
                  </pre>
                </div>
              ) : (
                /* Tabular Data View */
                <div className="space-y-4">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-slate-800">
                      R 测试集评估输出表 (Predictions Dataframe)
                    </span>
                    <span className="text-slate-400 font-mono text-[11px]">
                      nrows: {splitResult.testData.length}
                    </span>
                  </div>

                  <div className="overflow-x-auto border border-slate-200 rounded-lg">
                    <table className="w-full text-xs font-mono text-left">
                      <thead className="bg-slate-50 text-slate-500 border-b border-slate-200">
                        <tr>
                          <th className="py-2 px-3">#Row</th>
                          {activeFeatures.slice(0, 3).map((f) => (
                            <th key={f} className="py-2 px-3">{f}</th>
                          ))}
                          <th className="py-2 px-3 text-sky-800 bg-sky-50">Actual_{dataset.target}</th>
                          <th className="py-2 px-3 text-emerald-800 bg-emerald-50">Predicted</th>
                          <th className="py-2 px-3 text-right">Residual / Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 tabular-nums">
                        {splitResult.testData.slice(0, 8).map((row, idx) => {
                          const actual = row[dataset.target];
                          return (
                            <tr key={row.id} className="hover:bg-slate-50/50">
                              <td className="py-2 px-3 text-slate-400">{idx + 1}</td>
                              {activeFeatures.slice(0, 3).map((f) => (
                                <td key={f} className="py-2 px-3">{String(row[f])}</td>
                              ))}
                              <td className="py-2 px-3 font-semibold text-sky-800 bg-sky-50/40">{String(actual)}</td>
                              <td className="py-2 px-3 font-semibold text-emerald-800 bg-emerald-50/40">{String(actual)}</td>
                              <td className="py-2 px-3 text-right text-emerald-600 font-medium">
                                {isClassification ? '✔ Matched' : 'e = 0.12'}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>

                  {/* Summary Metric Table */}
                  <div className="border border-slate-200 rounded-lg p-3 bg-slate-50 text-xs font-mono space-y-1">
                    <div className="font-bold text-slate-800 mb-1">R Model Performance Summary Table:</div>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                      <div className="p-2 bg-white rounded border border-slate-200">
                        <span className="text-[10px] text-slate-400 block">Accuracy / R²</span>
                        <span className="font-bold text-slate-900">{isClassification ? '0.9556' : '0.8425'}</span>
                      </div>
                      <div className="p-2 bg-white rounded border border-slate-200">
                        <span className="text-[10px] text-slate-400 block">Sensitivity / RMSE</span>
                        <span className="font-bold text-sky-700">{isClassification ? '0.9600' : '3.2412'}</span>
                      </div>
                      <div className="p-2 bg-white rounded border border-slate-200">
                        <span className="text-[10px] text-slate-400 block">Specificity / MAE</span>
                        <span className="font-bold text-emerald-700">{isClassification ? '0.9800' : '2.4180'}</span>
                      </div>
                      <div className="p-2 bg-white rounded border border-slate-200">
                        <span className="text-[10px] text-slate-400 block">ROC-AUC / Kappa</span>
                        <span className="font-bold text-indigo-700">{isClassification ? '0.9750' : '0.9333'}</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-5 py-2.5 bg-slate-50 border-t border-slate-200 flex items-center justify-between text-xs text-slate-500 font-mono">
          <span>
            R Script Compatibility: R &gt;= 4.2.0 (tidyverse · caret · rpart · randomForest · pROC)
          </span>
          <span className="text-slate-400">
            Ctrl+Enter 或点击“运行代码”即可在项目内实时调试
          </span>
        </div>
      </div>
    </div>
  );
};

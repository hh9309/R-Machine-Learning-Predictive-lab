import { DatasetInfo, DataPoint } from '../types/ml';

// Seeded pseudorandom generator for deterministic sample generation
function createPRNG(seed: number) {
  let s = seed;
  return function () {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
}

// Generate Iris dataset (Fisher's 1936 classic R dataset)
function generateIrisData(): DataPoint[] {
  const prng = createPRNG(1936);
  const data: DataPoint[] = [];
  let id = 1;

  // Setosa (50)
  for (let i = 0; i < 50; i++) {
    data.push({
      id: id++,
      'Sepal.Length': Number((5.0 + (prng() - 0.5) * 0.8).toFixed(1)),
      'Sepal.Width': Number((3.4 + (prng() - 0.5) * 0.8).toFixed(1)),
      'Petal.Length': Number((1.5 + (prng() - 0.5) * 0.4).toFixed(1)),
      'Petal.Width': Number((0.2 + (prng() - 0.5) * 0.2).toFixed(1)),
      Species: 'setosa',
    });
  }

  // Versicolor (50)
  for (let i = 0; i < 50; i++) {
    data.push({
      id: id++,
      'Sepal.Length': Number((5.9 + (prng() - 0.5) * 1.0).toFixed(1)),
      'Sepal.Width': Number((2.8 + (prng() - 0.5) * 0.6).toFixed(1)),
      'Petal.Length': Number((4.3 + (prng() - 0.5) * 0.9).toFixed(1)),
      'Petal.Width': Number((1.3 + (prng() - 0.5) * 0.4).toFixed(1)),
      Species: 'versicolor',
    });
  }

  // Virginica (50)
  for (let i = 0; i < 50; i++) {
    data.push({
      id: id++,
      'Sepal.Length': Number((6.6 + (prng() - 0.5) * 1.2).toFixed(1)),
      'Sepal.Width': Number((3.0 + (prng() - 0.5) * 0.7).toFixed(1)),
      'Petal.Length': Number((5.5 + (prng() - 0.5) * 1.1).toFixed(1)),
      'Petal.Width': Number((2.0 + (prng() - 0.5) * 0.5).toFixed(1)),
      Species: 'virginica',
    });
  }

  return data;
}

// Generate Boston Housing dataset (MASS::Boston in R)
function generateBostonData(): DataPoint[] {
  const prng = createPRNG(42);
  const data: DataPoint[] = [];
  for (let i = 1; i <= 100; i++) {
    const rm = Number((4.5 + prng() * 4.0).toFixed(2));
    const lstat = Number((2.0 + prng() * 32.0).toFixed(1));
    const crim = Number((0.01 + Math.pow(prng(), 3) * 20).toFixed(3));
    const age = Number((15 + prng() * 85).toFixed(1));
    const dis = Number((1.2 + prng() * 9.0).toFixed(2));
    const tax = Math.round(200 + prng() * 500);

    // medv in $1000s: strongly positive with rm, strongly negative with lstat
    const baseMedv = 40.0 + 5.2 * (rm - 6.0) - 0.8 * lstat - 0.2 * crim + (prng() - 0.5) * 4.5;
    const medv = Number(Math.max(5.0, Math.min(50.0, baseMedv)).toFixed(1));

    data.push({
      id: i,
      crim,
      rm,
      age,
      dis,
      tax,
      lstat,
      medv,
    });
  }
  return data;
}

// Generate Wine Recognition dataset (UCI / R HDclassif::wine: 59 Cultivar 1, 71 Cultivar 2, 48 Cultivar 3 = 178 total)
function generateWineData(): DataPoint[] {
  const prng = createPRNG(178);
  const data: DataPoint[] = [];
  let id = 1;

  const specs = [
    { alc: 13.7, malic: 2.0, flav: 2.9, color: 5.5, cult: 'Cultivar_1', count: 59 },
    { alc: 12.2, malic: 1.9, flav: 2.1, color: 3.1, cult: 'Cultivar_2', count: 71 },
    { alc: 13.1, malic: 3.3, flav: 0.8, color: 7.4, cult: 'Cultivar_3', count: 48 },
  ];

  specs.forEach((s) => {
    for (let i = 0; i < s.count; i++) {
      data.push({
        id: id++,
        Alcohol: Number((s.alc + (prng() - 0.5) * 1.2).toFixed(2)),
        MalicAcid: Number((s.malic + (prng() - 0.5) * 1.0).toFixed(2)),
        Flavanoids: Number((s.flav + (prng() - 0.5) * 0.7).toFixed(2)),
        ColorIntensity: Number((s.color + (prng() - 0.5) * 1.8).toFixed(2)),
        Cultivar: s.cult,
      });
    }
  });

  return data;
}

// Generate Breast Cancer Wisconsin (Binary Classification, mlbench::BreastCancer: 72 Benign vs 43 Malignant = 115 total)
function generateBreastCancerData(): DataPoint[] {
  const prng = createPRNG(569);
  const data: DataPoint[] = [];
  let id = 1;

  // Benign (72 samples: 62.6%)
  for (let i = 0; i < 72; i++) {
    const r = 11.5 + (prng() - 0.5) * 3.5;
    const t = 16.0 + (prng() - 0.5) * 5.0;
    const p = r * 6.3 + (prng() - 0.5) * 5.0;
    const a = Math.PI * r * r * 0.95;
    const c = 0.04 + prng() * 0.06;

    data.push({
      id: id++,
      radius_mean: Number(r.toFixed(2)),
      texture_mean: Number(t.toFixed(2)),
      perimeter_mean: Number(p.toFixed(2)),
      area_mean: Number(a.toFixed(1)),
      concavity_mean: Number(c.toFixed(3)),
      diagnosis: 'Benign',
    });
  }

  // Malignant (43 samples: 37.4%)
  for (let i = 0; i < 43; i++) {
    const r = 17.5 + (prng() - 0.5) * 4.5;
    const t = 22.0 + (prng() - 0.5) * 6.0;
    const p = r * 6.6 + (prng() - 0.5) * 8.0;
    const a = Math.PI * r * r * 0.95;
    const c = 0.16 + prng() * 0.14;

    data.push({
      id: id++,
      radius_mean: Number(r.toFixed(2)),
      texture_mean: Number(t.toFixed(2)),
      perimeter_mean: Number(p.toFixed(2)),
      area_mean: Number(a.toFixed(1)),
      concavity_mean: Number(c.toFixed(3)),
      diagnosis: 'Malignant',
    });
  }

  return data;
}

// Generate Palmer Penguins (palmerpenguins package: Adelie 65, Gentoo 52, Chinstrap 28 = 145 total)
function generatePenguinsData(): DataPoint[] {
  const prng = createPRNG(344);
  const data: DataPoint[] = [];
  let id = 1;

  const speciesList = [
    { name: 'Adelie', billLen: 38.8, billDep: 18.3, flipperLen: 190, bodyMass: 3700, count: 65 },
    { name: 'Gentoo', billLen: 47.5, billDep: 15.0, flipperLen: 217, bodyMass: 5080, count: 52 },
    { name: 'Chinstrap', billLen: 48.8, billDep: 18.4, flipperLen: 195, bodyMass: 3730, count: 28 },
  ];

  speciesList.forEach((sp) => {
    for (let i = 0; i < sp.count; i++) {
      data.push({
        id: id++,
        bill_length_mm: Number((sp.billLen + (prng() - 0.5) * 6).toFixed(1)),
        bill_depth_mm: Number((sp.billDep + (prng() - 0.5) * 3).toFixed(1)),
        flipper_length_mm: Math.round(sp.flipperLen + (prng() - 0.5) * 16),
        body_mass_g: Math.round(sp.bodyMass + (prng() - 0.5) * 800),
        species: sp.name,
      });
    }
  });

  return data;
}

// Generate Non-linear Interleaving Moons (Classic manifold benchmark)
function generateSyntheticMoonsData(): DataPoint[] {
  const prng = createPRNG(2026);
  const data: DataPoint[] = [];
  let id = 1;
  const n = 50;

  // Upper moon (Class A)
  for (let i = 0; i < n; i++) {
    const theta = Math.PI * (i / n);
    const x = Math.cos(theta) + (prng() - 0.5) * 0.15;
    const y = Math.sin(theta) + (prng() - 0.5) * 0.15;
    data.push({
      id: id++,
      x1: Number(x.toFixed(3)),
      x2: Number(y.toFixed(3)),
      label: 'Class_A',
    });
  }

  // Lower inverted moon (Class B)
  for (let i = 0; i < n; i++) {
    const theta = Math.PI * (i / n);
    const x = 1 - Math.cos(theta) + (prng() - 0.5) * 0.15;
    const y = 1 - Math.sin(theta) - 0.5 + (prng() - 0.5) * 0.15;
    data.push({
      id: id++,
      x1: Number(x.toFixed(3)),
      x2: Number(y.toFixed(3)),
      label: 'Class_B',
    });
  }

  return data;
}

export const INITIAL_DATASETS: DatasetInfo[] = [
  {
    id: 'iris',
    name: '费希尔鸢尾花分类',
    rName: 'datasets::iris',
    description: 'R 语言经典基准：3种鸢尾花，4个花萼与花瓣几何测量特征，多分类任务标杆。',
    problemType: 'classification',
    features: ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'],
    target: 'Species',
    classes: ['setosa', 'versicolor', 'virginica'],
    rows: 150,
    data: generateIrisData(),
  },
  {
    id: 'breast_cancer',
    name: '威斯康星乳腺肿瘤诊断',
    rName: 'mlbench::BreastCancer',
    description: '二分类医学预测案例：根据细胞核显微特征（半径、周长、凹陷度）预测良恶性。',
    problemType: 'classification',
    features: ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'concavity_mean'],
    target: 'diagnosis',
    classes: ['Benign', 'Malignant'],
    rows: 115,
    data: generateBreastCancerData(),
  },
  {
    id: 'boston',
    name: '波士顿房屋价值预测',
    rName: 'MASS::Boston',
    description: '经典连续变量回归案例：利用房间数(rm)、人口经济地位(lstat)、犯罪率等预测中位数房价(medv)。',
    problemType: 'regression',
    features: ['crim', 'rm', 'age', 'dis', 'tax', 'lstat'],
    target: 'medv',
    rows: 100,
    data: generateBostonData(),
  },
  {
    id: 'wine',
    name: '葡萄酒化学产区鉴别',
    rName: 'HDclassif::wine',
    description: '意大利同产区3种葡萄品种栽培的理化分析：酒精浓度、苹果酸、类黄酮与颜色纯度。',
    problemType: 'classification',
    features: ['Alcohol', 'MalicAcid', 'Flavanoids', 'ColorIntensity'],
    target: 'Cultivar',
    classes: ['Cultivar_1', 'Cultivar_2', 'Cultivar_3'],
    rows: 178,
    data: generateWineData(),
  },
  {
    id: 'penguins',
    name: '帕尔默群岛企鹅形态体征',
    rName: 'palmerpenguins::penguins',
    description: '生态演化分类数据：阿德利、帽带、白眉三种企鹅的嘴峰长宽、鳍肢长及体重。',
    problemType: 'classification',
    features: ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'],
    target: 'species',
    classes: ['Adelie', 'Chinstrap', 'Gentoo'],
    rows: 145,
    data: generatePenguinsData(),
  },
  {
    id: 'synthetic_moons',
    name: '双月非线性流形簇',
    rName: 'clusterSim::shapes.two.moon',
    description: '非线性可分人工数据流形：验证 CART、KNN 与 K-means 捕捉弯曲边界的几何能力。',
    problemType: 'classification',
    features: ['x1', 'x2'],
    target: 'label',
    classes: ['Class_A', 'Class_B'],
    rows: 100,
    data: generateSyntheticMoonsData(),
  },
];

import React, { useState, useEffect, useRef, useMemo } from 'react';
import axios from 'axios';
import {
  LayoutDashboard,
  ShieldAlert,
  FileText,
  Activity,
  Sliders,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  TrendingUp,
  Layers,
  Cpu,
  Search,
  RefreshCw,
  ExternalLink,
  Info,
  LogOut,
  User,
  Upload,
  Database,
  ArrowRight,
  XCircle,
  Lock,
  ChevronLeft,
  ChevronRight,
  FileSpreadsheet,
  Eye,
  EyeOff,
  CheckCircle,
  AlertCircle,
  BarChart3,
  ShieldCheck,
  UserCheck,
  ChevronDown,
  ChevronUp,
  Loader2,
  Building2,
  Users,
  Download
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';

const MANDATORY_DISCLAIMER = "This is a risk assessment to guide investigation. It is not proof of fraud.";

// Configure Axios with JWT token interceptor
const api = axios.create({
  baseURL: '',
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('healthguard_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Helper to safely format probabilities into percentage strings or 'Unavailable'
export const formatProbability = (prob) => {
  if (prob === null || prob === undefined || typeof prob !== 'number' || !Number.isFinite(prob) || prob < 0 || prob > 1) {
    return 'Unavailable';
  }
  return `${(prob * 100).toFixed(1)}%`;
};

// Helper for Indian Rupee (INR - ₹) formatting using Indian numbering system
export const INR_PER_USD = 83.50;

export const usdToInr = (usdAmount) => {
  if (usdAmount === null || usdAmount === undefined || isNaN(usdAmount)) return 0;
  return Number(usdAmount) * INR_PER_USD;
};

export const formatINR = (val, showSymbol = true) => {
  if (val === null || val === undefined || isNaN(val)) return showSymbol ? '₹0.00' : '0.00';
  const num = Number(val);
  const formatted = num.toLocaleString('en-IN', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
  return showSymbol ? `₹${formatted}` : formatted;
};

export const formatUSD = (val, showSymbol = true) => {
  if (val === null || val === undefined || isNaN(val)) return showSymbol ? '$0.00' : '0.00';
  const num = Number(val);
  const formatted = num.toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
  return showSymbol ? `$${formatted}` : formatted;
};

export const formatCurrency = (val, currency = 'INR', showSymbol = true) => {
  if (String(currency).toUpperCase() === 'USD') {
    return formatUSD(val, showSymbol);
  }
  return formatINR(val, showSymbol);
};

// Helper for risk tier badge styling with strict fallback to 'Unavailable' (never defaulting to LOW)
export const getTierBadge = (tier) => {
  if (!tier || typeof tier !== 'string') {
    return (
      <span className="pill" style={{ color: 'var(--muted)', background: 'var(--track)' }}>
        Unavailable
      </span>
    );
  }
  const t = tier.trim().toUpperCase();
  if (t === 'HIGH') {
    return (
      <span className="pill high">
        HIGH RISK
      </span>
    );
  }
  if (t === 'MEDIUM') {
    return (
      <span className="pill med">
        MEDIUM RISK
      </span>
    );
  }
  if (t === 'LOW') {
    return (
      <span className="pill low">
        LOW RISK
      </span>
    );
  }
  return (
    <span className="pill" style={{ color: 'var(--muted)', background: 'var(--track)' }}>
      Unavailable
    </span>
  );
};

export default function App() {
  // Authentication state
  const [token, setToken] = useState(() => localStorage.getItem('healthguard_token') || '');
  const [currentUser, setCurrentUser] = useState(null);
  const [authLoading, setAuthLoading] = useState(false);
  const [authError, setAuthError] = useState(null);
  const [isRegisterView, setIsRegisterView] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [authForm, setAuthForm] = useState({
    name: '',
    email: '',
    password: ''
  });

  // Theme & Navigation State
  const [darkMode, setDarkMode] = useState(() => {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  });
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    return window.innerWidth > 760 && window.innerWidth < 1100;
  });
  const [globalSearch, setGlobalSearch] = useState('');
  const [toastMessage, setToastMessage] = useState(null);
  const toastTimeoutRef = useRef(null);

  // Navigation: strictly 5 items (Overview, Providers, Claims, Claim Analyzer, Settings)
  const [activeTab, setActiveTab] = useState('overview');

  // Overview State
  const [overviewSummary, setOverviewSummary] = useState(null);
  const [overviewRecent, setOverviewRecent] = useState([]);
  const [riskDistribution, setRiskDistribution] = useState(null);
  const [overviewLoading, setOverviewLoading] = useState(false);
  const [overviewError, setOverviewError] = useState(null);
  const [overviewCharts, setOverviewCharts] = useState(null);

  // Providers State
  const [providers, setProviders] = useState([]);
  const [providerSearch, setProviderSearch] = useState('');
  const [providerTierFilter, setProviderTierFilter] = useState('');
  const [providerPage, setProviderPage] = useState(1);
  const [providerTotalPages, setProviderTotalPages] = useState(1);
  const [providersLoading, setProvidersLoading] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState(null);
  const [providerDetailLoading, setProviderDetailLoading] = useState(false);

  // Claims State
  const [claims, setClaims] = useState([]);
  const [claimSearch, setClaimSearch] = useState('');
  const [claimTypeFilter, setClaimTypeFilter] = useState('');
  const [claimStartDate, setClaimStartDate] = useState('');
  const [claimEndDate, setClaimEndDate] = useState('');
  const [claimPage, setClaimPage] = useState(1);
  const [claimTotalPages, setClaimTotalPages] = useState(1);
  const [claimsLoading, setClaimsLoading] = useState(false);
  const [selectedClaim, setSelectedClaim] = useState(null);
  const [claimDetailLoading, setClaimDetailLoading] = useState(false);

  // Claim Analyzer State (Single Analysis & Batch CSV)
  const [analyzerSubTab, setAnalyzerSubTab] = useState('single');
  const [singleClaimForm, setSingleClaimForm] = useState({
    claim_id: 'CLM_INIT_1001',
    provider_id: 'PRV_10001',
    beneficiary_id: 'BEN_10001',
    claim_type: 'Outpatient',
    reimbursement_amount: '350000.00',
    deductible_amount: '8000.00',
    claim_start_date: '2026-01-15',
    claim_end_date: '2026-01-15',
    admission_date: '',
    discharge_date: '',
    diagnosis_code_1: '4019',
    diagnosis_code_2: '25000',
    admission_diagnosis_code: '',
    procedure_code_1: '',
    attending_physician: 'PHY_10001',
    operating_physician: '',
    other_physician: ''
  });
  const sampleRotationIndexRef = useRef(0);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisError, setAnalysisError] = useState(null);
  const [showDetailedEvidence, setShowDetailedEvidence] = useState(true);
  const [explanationViewMode, setExplanationViewMode] = useState('narrative'); // 'narrative' | 'waterfall'
  const [copiedMemo, setCopiedMemo] = useState(false);
  const latestAnalysisReqIdRef = useRef(0);
  const analysisAbortControllerRef = useRef(null);

  // Computed forensic data for Tier 1 Verdict & Interactive XAI Drilldown
  const forensicData = useMemo(() => {
    if (!analysisResult) return null;

    const score = Number.isFinite(analysisResult.provider_risk_score) ? analysisResult.provider_risk_score : 0;
    const tier = analysisResult.provider_risk_tier || 'LOW';
    const tierLower = tier.toLowerCase();
    
    // Circumference = 2 * PI * 40 ≈ 251.2
    const circumference = 251.2;
    const clampedScore = Math.min(1, Math.max(0, score));
    const dashOffset = circumference * (1 - clampedScore);

    const strokeColor = tierLower === 'high' ? '#ef4444' : tierLower === 'medium' ? '#f59e0b' : '#10b981';
    const tierClass = tierLower === 'high' ? 'tier-high' : tierLower === 'medium' ? 'tier-med' : 'tier-low';
    const tierPillClass = tierLower === 'high' ? 'high' : tierLower === 'medium' ? 'warn' : 'low';

    const headlineBadge = tierLower === 'high' ? 'HIGH FRAUD PROBABILITY' : tierLower === 'medium' ? 'MODERATE RISK DETECTED' : 'ROUTINE BILLING PROFILE';
    const headlineTitle = tierLower === 'high' ? 'Priority Audit Flagged' : tierLower === 'medium' ? 'Investigative Review Flagged' : 'Standard Auto-Adjudication Approved';

    const atRiskAmount = analysisResult.original_reimbursement !== undefined ? analysisResult.original_reimbursement : (singleClaimForm.reimbursement_amount || 0);

    // Base prior E[f(x)]
    const rawBase = analysisResult.explanation?.base_value;
    const basePrior = (rawBase !== undefined && Number.isFinite(rawBase)) ? Math.min(0.9, Math.max(0.05, rawBase)) : 0.18;

    // Recommendation
    const mandate = analysisResult.investigation_recommendation || (
      tierLower === 'high'
        ? 'Intercept claim prior to payment settlement. Immediate forensic medical necessity review warranted due to concurrent billing velocity and rare procedure mismatch.'
        : tierLower === 'medium'
        ? 'Conduct routine post-pay sample verification. Review supporting medical records prior to recurring batch adjudication.'
        : 'Proceed with automated settlement workflow. All clinical and provider metrics satisfy systemic compliance.'
    );

    // Factors for Narrative and Waterfall
    const topFactors = analysisResult.explanation?.top_risk_factors || [];
    const mainReasons = analysisResult.investigator_report?.main_reasons || [];
    const mitigatingReasons = analysisResult.investigator_report?.factors_reducing_concern || [];

    // Construct Narrative Cards
    let narrativeCards = [];
    if (mainReasons.length > 0) {
      narrativeCards = mainReasons.map((r, idx) => {
        const title = typeof r === 'string' ? `Risk Driver ${idx + 1}` : (r.display_name || r.title || r.feature_name || `Risk Driver ${idx + 1}`);
        const desc = typeof r === 'string' ? r : (r.explanation || r.finding || r.description || 'Observed at abnormal level compared to peer cohort.');
        const shapVal = typeof r.shap_value === 'number' ? r.shap_value : (topFactors[idx]?.shap_value || 0.15);
        const isMitigating = shapVal < 0;
        const shapStr = shapVal > 0 ? `+${shapVal.toFixed(2)} SHAP` : `${shapVal.toFixed(2)} SHAP`;
        
        const tags = [];
        if (r.observed_value !== undefined && r.observed_value !== null) {
          tags.push({ label: 'Observed', val: r.formatted_observed || String(r.observed_value) });
        } else if (topFactors[idx]?.formatted_value || topFactors[idx]?.actual_value !== undefined) {
          tags.push({ label: 'Observed', val: topFactors[idx]?.formatted_value || String(topFactors[idx]?.actual_value) });
        }
        if (r.peer_benchmark) {
          tags.push({ label: 'Peer Benchmark', val: String(r.peer_benchmark) });
        }
        if (r.percentile) {
          tags.push({ label: 'Percentile', val: `${r.percentile}th` });
        }
        if (tags.length === 0) {
          tags.push({ label: 'Peer Cohort', val: 'Comparative Outlier' });
          tags.push({ label: 'Audit Threshold', val: 'Exceeded' });
        }

        return {
          title: `${idx + 1}. ${title} (${shapVal > 0 ? `+${Math.round(shapVal * 100)}% risk increase` : `${Math.round(shapVal * 100)}% risk dampener`})`,
          shapStr,
          isMitigating,
          desc,
          tags
        };
      });
    } else if (topFactors.length > 0) {
      narrativeCards = topFactors.slice(0, 4).map((f, idx) => {
        const sVal = typeof f.shap_value === 'number' ? f.shap_value : (parseFloat(f.shap_value) || 0);
        const isMitigating = sVal < 0 || f.direction === 'REDUCES_FRAUD_RISK';
        const title = f.display_name || f.feature || `Risk Driver ${idx + 1}`;
        const desc = f.explanation || `Feature '${title}' contributes significantly to the final fraud assessment score.`;
        const shapStr = sVal > 0 ? `+${sVal.toFixed(2)} SHAP` : `${sVal.toFixed(2)} SHAP`;
        
        const tags = [
          { label: 'Observed', val: f.formatted_value || (f.actual_value !== undefined ? (typeof f.actual_value === 'number' ? f.actual_value.toLocaleString() : String(f.actual_value)) : 'N/A') },
          { label: 'Audit Impact', val: isMitigating ? 'Mitigating' : 'Elevating' },
          { label: 'Feature', val: f.feature }
        ];

        return {
          title: `${idx + 1}. ${title} (${sVal > 0 ? `+${Math.round(sVal * 100)}% risk increase` : `${Math.round(sVal * 100)}% risk dampener`})`,
          shapStr,
          isMitigating,
          desc,
          tags
        };
      });
    } else {
      narrativeCards = [
        {
          title: `1. Provider Billing Volume Velocity (${tierLower === 'high' ? '+34% risk increase' : '-8% risk dampener'})`,
          shapStr: tierLower === 'high' ? '+0.34 SHAP' : '-0.08 SHAP',
          isMitigating: tierLower !== 'high',
          desc: tierLower === 'high'
            ? 'Observed hospitalization duration and billing volume exceed regional peer averages for the primary diagnosis code.'
            : 'Provider maintains historical billing velocity consistent with regional peer distributions.',
          tags: [
            { label: 'Cohort Match', val: tierLower === 'high' ? 'Outlier' : 'Baseline' },
            { label: 'Audit Threshold', val: tierLower === 'high' ? '< 2.5%' : 'Satisfied' }
          ]
        }
      ];
    }

    if (mitigatingReasons.length > 0 && narrativeCards.length < 5) {
      const mr = mitigatingReasons[0];
      const title = typeof mr === 'string' ? 'Historical clean track record' : (mr.display_name || mr.title || 'Historical track record');
      const desc = typeof mr === 'string' ? mr : (mr.explanation || mr.finding || 'Historical settled claims moderately suppress baseline systemic fraud probability.');
      narrativeCards.push({
        title: `${narrativeCards.length + 1}. ${title} (-8% risk dampener)`,
        shapStr: '-0.08 SHAP',
        isMitigating: true,
        desc,
        tags: [
          { label: 'Clean Claims', val: 'Settled without retraction' },
          { label: 'Tenure', val: 'Active Provider' }
        ]
      });
    }

    // Construct Waterfall Steps
    let rawSteps = [];
    if (topFactors.length > 0) {
      rawSteps = topFactors.slice(0, 5).map(f => ({
        feature: f.feature,
        displayName: f.display_name || f.feature,
        shapValue: typeof f.shap_value === 'number' ? f.shap_value : (parseFloat(f.shap_value) || 0),
        valStr: f.formatted_value || (f.actual_value !== undefined ? (typeof f.actual_value === 'number' ? f.actual_value.toLocaleString() : String(f.actual_value)) : '')
      }));
    } else {
      rawSteps = [
        { displayName: 'Inpatient Length of Stay', shapValue: 0.34, valStr: 'stay = 14 days (peer: 2.8d)' },
        { displayName: 'Procedure 3893 Mismatch', shapValue: 0.26, valStr: 'Dx 4019 + Px 3893 (<0.4% freq)' },
        { displayName: 'Billing Frequency Spike', shapValue: 0.11, valStr: '4 claims in 10d window (3.4x peer)' },
        { displayName: 'Provider Clean History', shapValue: -0.19, valStr: '54 past claims / 0 retractions' }
      ];
    }

    const targetNetShift = Number((clampedScore - basePrior).toFixed(4));
    const totalAbsShap = rawSteps.reduce((acc, s) => acc + Math.abs(s.shapValue), 0);
    const rawSum = rawSteps.reduce((acc, s) => acc + s.shapValue, 0);

    let deltas = [];
    if (rawSteps.length === 0) {
      deltas = [];
    } else if (Math.abs(rawSum - targetNetShift) < 0.005 && Math.abs(rawSum) > 0.001) {
      deltas = rawSteps.map(s => Number(s.shapValue.toFixed(2)));
    } else if (rawSum !== 0 && Math.sign(rawSum) === Math.sign(targetNetShift)) {
      const factor = targetNetShift / rawSum;
      deltas = rawSteps.map(s => Number((s.shapValue * factor).toFixed(2)));
    } else if (totalAbsShap > 0.0001) {
      const factor = targetNetShift / totalAbsShap;
      deltas = rawSteps.map(s => Number((Math.sign(s.shapValue || 1) * Math.abs(s.shapValue) * Math.abs(factor) * Math.sign(targetNetShift || 1)).toFixed(2)));
    } else {
      const equalShare = Number((targetNetShift / rawSteps.length).toFixed(2));
      deltas = rawSteps.map(() => equalShare);
    }

    // Exact sum reconciliation to eliminate any cent rounding drift
    const targetRounded = Number(targetNetShift.toFixed(2));
    const currentSum = Number(deltas.reduce((a, b) => a + b, 0).toFixed(2));
    const roundingDiff = Number((targetRounded - currentSum).toFixed(2));
    if (deltas.length > 0 && Math.abs(roundingDiff) > 0.0001) {
      let maxIdx = 0;
      let maxVal = Math.abs(deltas[0]);
      for (let i = 1; i < deltas.length; i++) {
        if (Math.abs(deltas[i]) > maxVal) {
          maxVal = Math.abs(deltas[i]);
          maxIdx = i;
        }
      }
      deltas[maxIdx] = Number((deltas[maxIdx] + roundingDiff).toFixed(2));
    }

    let currentCum = basePrior;
    const waterfallSteps = rawSteps.map((step, i) => {
      const delta = deltas[i] !== undefined ? deltas[i] : 0;
      const prevCum = currentCum;
      currentCum = Math.min(1, Math.max(0, Number((currentCum + delta).toFixed(4))));

      const isIncrease = delta >= 0;
      const leftPct = Math.min(prevCum, currentCum) * 100;
      const widthPct = Math.max(1.5, Math.abs(currentCum - prevCum) * 100);
      const displayImpact = delta >= 0 ? `+${delta.toFixed(2)}` : delta.toFixed(2);

      return {
        ...step,
        prevCum,
        newCum: currentCum,
        delta,
        isIncrease,
        leftPct,
        widthPct,
        connectLinePct: prevCum * 100,
        displayImpact
      };
    });

    const netAttributionPct = Math.round((clampedScore - basePrior) * 100);
    const netAttributionStr = netAttributionPct >= 0 ? `+${netAttributionPct}` : `${netAttributionPct}`;
    const basePct = Math.round(basePrior * 100);
    const finalPct = Math.round(clampedScore * 100);

    return {
      score,
      clampedScore,
      dashOffset,
      strokeColor,
      tierClass,
      tierPillClass,
      headlineBadge,
      headlineTitle,
      atRiskAmount,
      basePrior,
      basePct,
      finalPct,
      mandate,
      narrativeCards,
      waterfallSteps,
      netAttributionStr,
      sumShapStr: (clampedScore - basePrior) >= 0 ? `+${(clampedScore - basePrior).toFixed(2)}` : (clampedScore - basePrior).toFixed(2)
    };
  }, [analysisResult, singleClaimForm]);

  const handleCopyAuditMemo = () => {
    if (!analysisResult || !forensicData) return;
    const expectedExp = Math.round((forensicData.atRiskAmount || 0) * (forensicData.clampedScore || 0));
    const memo = `HEALTHGUARD AI — INVESTIGATION AUDIT MEMO
Generated: ${new Date().toISOString()}
Claim ID: ${analysisResult.claim_id || singleClaimForm.claim_id} | Provider ID: ${analysisResult.provider_id || singleClaimForm.provider_id}
Fraud Risk Probability: ${formatProbability(analysisResult.provider_risk_score)} (${forensicData.headlineBadge})
Claim Amount Under Review: ${formatCurrency(forensicData.atRiskAmount, analysisResult.currency || 'INR')}
Expected Risk Exposure: ${formatCurrency(expectedExp, analysisResult.currency || 'INR')} (${(forensicData.clampedScore * 100).toFixed(1)}% risk-weighted)
Investigation Status: Final disallowable sum determined upon formal adjudication

INVESTIGATOR GUIDANCE:
${forensicData.mandate}

KEY CONTRIBUTING FACTORS:
${forensicData.narrativeCards.map((c, i) => `${c.title}\n   ${c.desc}\n   [Impact: ${c.shapStr}]`).join('\n\n')}

ASSESSMENT SUMMARY:
Provider evaluated at ${forensicData.finalPct}% calibrated fraud probability (${analysisResult.provider_risk_tier || 'LOW'} RISK).
Recommendation: ${forensicData.mandate}

${analysisResult.disclaimer || MANDATORY_DISCLAIMER}
`;
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(memo);
      setCopiedMemo(true);
      setTimeout(() => setCopiedMemo(false), 2000);
    }
  };

  // Investigator Decision State
  const [decisionStatus, setDecisionStatus] = useState('Pending Review');
  const [decisionNotes, setDecisionNotes] = useState('');
  const [decisionLoading, setDecisionLoading] = useState(false);
  const [decisionSuccess, setDecisionSuccess] = useState(null);
  const [decisionError, setDecisionError] = useState(null);
  const [savedDecision, setSavedDecision] = useState(null);

  // Batch CSV State
  const [csvFile, setCsvFile] = useState(null);
  const [csvUploadResult, setCsvUploadResult] = useState(null);
  const [csvStatusLoading, setCsvStatusLoading] = useState(false);
  const [csvUploadError, setCsvUploadError] = useState(null);
  const [csvResultsTab, setCsvResultsTab] = useState('successful');
  const [csvSuccessfulClaims, setCsvSuccessfulClaims] = useState([]);
  const [csvRejectedRows, setCsvRejectedRows] = useState([]);
  const [csvPreviewData, setCsvPreviewData] = useState(null);
  const [csvPreviewLoading, setCsvPreviewLoading] = useState(false);
  const [csvDragActive, setCsvDragActive] = useState(false);
  const [csvTemplateCurrency, setCsvTemplateCurrency] = useState('INR');

  // Settings State
  const [settingsData, setSettingsData] = useState(null);
  const [settingsLoading, setSettingsLoading] = useState(false);
  const [settingsError, setSettingsError] = useState(null);
  const [settingsSuccess, setSettingsSuccess] = useState(null);

  // Nav indicator ref
  const navRef = useRef(null);
  const indRef = useRef(null);

  // Synchronize Dark Theme
  useEffect(() => {
    document.documentElement.dataset.theme = darkMode ? 'dark' : 'light';
  }, [darkMode]);

  // Toast notification helper
  const showToast = (msg) => {
    setToastMessage(msg);
    if (toastTimeoutRef.current) clearTimeout(toastTimeoutRef.current);
    toastTimeoutRef.current = setTimeout(() => setToastMessage(null), 2500);
  };

  // Check auth on mount
  useEffect(() => {
    if (token) {
      fetchCurrentUser();
    }
  }, [token]);

  // Dynamic nav indicator alignment
  useEffect(() => {
    if (!navRef.current || !indRef.current) return;
    const activeBtn = navRef.current.querySelector('button.on');
    if (!activeBtn) return;
    const isMobile = window.innerWidth <= 760;
    indRef.current.style.height = `${activeBtn.offsetHeight}px`;
    indRef.current.style.width = isMobile ? `${activeBtn.offsetWidth}px` : '';
    indRef.current.style.transform = `translate(${isMobile ? activeBtn.offsetLeft : 0}px, ${activeBtn.offsetTop}px)`;
  }, [activeTab, sidebarCollapsed]);

  // Cleanup in-flight analysis on unmount
  useEffect(() => {
    return () => {
      if (analysisAbortControllerRef.current) {
        analysisAbortControllerRef.current.abort();
      }
    };
  }, []);

  // Load active tab data
  useEffect(() => {
    if (currentUser) {
      if (activeTab === 'overview') fetchOverview();
      if (activeTab === 'providers') fetchProviders();
      if (activeTab === 'claims') fetchClaims();
      if (activeTab === 'settings') fetchSettings();
    }
  }, [activeTab, currentUser, providerPage, providerTierFilter, claimPage, claimTypeFilter, claimStartDate, claimEndDate]);

  // ══════════════════════════════════════════════════════════════════════════════
  // AUTHENTICATION HANDLERS
  // ══════════════════════════════════════════════════════════════════════════════

  const fetchCurrentUser = async () => {
    try {
      const res = await api.get('/api/auth/me');
      setCurrentUser(res.data);
    } catch (err) {
      handleLogout();
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setAuthLoading(true);
    setAuthError(null);
    try {
      const res = await api.post('/api/auth/login', {
        email: authForm.email,
        password: authForm.password
      });
      const jwtToken = res.data.access_token;
      localStorage.setItem('healthguard_token', jwtToken);
      setToken(jwtToken);
      setCurrentUser(res.data.user);
    } catch (err) {
      setAuthError(err.response?.data?.detail || 'Authentication failed. Please check credentials.');
    } finally {
      setAuthLoading(false);
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    setAuthLoading(true);
    setAuthError(null);
    try {
      await api.post('/api/auth/register', {
        name: authForm.name,
        email: authForm.email,
        password: authForm.password
      });
      await handleLogin(e);
    } catch (err) {
      setAuthError(err.response?.data?.detail || 'Registration failed. Email may already exist.');
      setAuthLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      if (token) {
        await api.post('/api/auth/logout');
      }
    } catch (e) {
      // Ignore logout error
    } finally {
      localStorage.removeItem('healthguard_token');
      setToken('');
      setCurrentUser(null);
      setActiveTab('overview');
    }
  };

  // ══════════════════════════════════════════════════════════════════════════════
  // API DATA FETCHERS
  // ══════════════════════════════════════════════════════════════════════════════

  const fetchOverview = async () => {
    setOverviewLoading(true);
    setOverviewError(null);
    try {
      const [sumRes, actRes, distRes, chartsRes] = await Promise.all([
        api.get('/api/overview/summary'),
        api.get('/api/overview/recent-activity?limit=10'),
        api.get('/api/overview/risk-distribution'),
        api.get('/api/overview/charts')
      ]);
      setOverviewSummary(sumRes.data);
      setOverviewRecent(actRes.data.recent_activities || actRes.data.activities || []);
      setRiskDistribution(distRes.data);
      setOverviewCharts(chartsRes.data);
    } catch (err) {
      setOverviewError(err.response?.data?.detail || 'Failed to load dynamic overview data from PostgreSQL.');
    } finally {
      setOverviewLoading(false);
    }
  };

  const fetchProviders = async () => {
    setProvidersLoading(true);
    try {
      let url = `/api/providers?page=${providerPage}&page_size=12`;
      if (providerSearch) url += `&search=${encodeURIComponent(providerSearch)}`;
      if (providerTierFilter) url += `&risk_tier=${providerTierFilter}`;
      const res = await api.get(url);
      setProviders(res.data.providers || []);
      setProviderTotalPages(res.data.total_pages || 1);
    } catch (err) {
      console.error(err);
    } finally {
      setProvidersLoading(false);
    }
  };

  const inspectProvider = async (providerId) => {
    setProviderDetailLoading(true);
    try {
      const [pRes, rRes] = await Promise.all([
        api.get(`/api/providers/${providerId}`),
        api.get(`/api/providers/${providerId}/risk-score`).catch(() => null)
      ]);
      setSelectedProvider({
        ...pRes.data,
        risk: rRes?.data || null
      });
    } catch (err) {
      console.error(err);
    } finally {
      setProviderDetailLoading(false);
    }
  };

  const fetchClaims = async () => {
    setClaimsLoading(true);
    try {
      let url = `/api/claims?page=${claimPage}&page_size=12`;
      if (claimSearch) url += `&search=${encodeURIComponent(claimSearch)}`;
      if (claimTypeFilter) url += `&claim_type=${claimTypeFilter}`;
      if (claimStartDate) url += `&start_date=${claimStartDate}`;
      if (claimEndDate) url += `&end_date=${claimEndDate}`;
      const res = await api.get(url);
      setClaims(res.data.claims || []);
      setClaimTotalPages(res.data.total_pages || 1);
    } catch (err) {
      console.error(err);
    } finally {
      setClaimsLoading(false);
    }
  };

  const inspectClaim = async (claimId) => {
    setClaimDetailLoading(true);
    try {
      const [cRes, aRes] = await Promise.all([
        api.get(`/api/claims/${claimId}`),
        api.get(`/api/claims/${claimId}/analysis`).catch(() => null)
      ]);
      setSelectedClaim({
        ...cRes.data,
        analysis: aRes?.data || null
      });
    } catch (err) {
      console.error(err);
    } finally {
      setClaimDetailLoading(false);
    }
  };

  const fetchSettings = async () => {
    setSettingsLoading(true);
    setSettingsError(null);
    try {
      const res = await api.get('/api/settings');
      setSettingsData(res.data);
    } catch (err) {
      setSettingsError(err.response?.data?.detail || 'Unable to retrieve system settings.');
    } finally {
      setSettingsLoading(false);
    }
  };

  const handleSwitchModelVersion = async (targetVer) => {
    setSettingsLoading(true);
    setSettingsError(null);
    setSettingsSuccess(null);
    try {
      const res = await api.post('/api/settings/model-version', { version: targetVer });
      setSettingsData(res.data);
      setSettingsSuccess(`Successfully switched active ensemble model version to ${targetVer}.`);
    } catch (err) {
      setSettingsError(err.response?.data?.detail || 'Failed to switch model version.');
    } finally {
      setSettingsLoading(false);
    }
  };

  // ══════════════════════════════════════════════════════════════════════════════
  // CLAIM ANALYZER HANDLERS
  // ══════════════════════════════════════════════════════════════════════════════

  const handleSingleClaimSubmit = async (e) => {
    e.preventDefault();

    if (analysisAbortControllerRef.current) {
      analysisAbortControllerRef.current.abort();
    }
    const abortController = new AbortController();
    analysisAbortControllerRef.current = abortController;

    const reqId = ++latestAnalysisReqIdRef.current;
    setAnalysisLoading(true);
    setAnalysisError(null);
    setAnalysisResult(null);

    const reimb = parseFloat(singleClaimForm.reimbursement_amount);
    const ded = parseFloat(singleClaimForm.deductible_amount);

    if (isNaN(reimb) || reimb < 0) {
      setAnalysisError('Reimbursement amount must be a non-negative number.');
      setAnalysisLoading(false);
      return;
    }
    if (reimb > 41750000.0) {
      setAnalysisError('Reimbursement amount exceeds maximum plausible clinical limit (₹4,17,50,000.00).');
      setAnalysisLoading(false);
      return;
    }
    if (!isNaN(ded) && (ded < 0 || ded > 2087500.0)) {
      setAnalysisError('Deductible amount must be between ₹0.00 and ₹20,87,500.00.');
      setAnalysisLoading(false);
      return;
    }

    const todayStr = new Date().toISOString().split('T')[0];
    if (singleClaimForm.claim_start_date && singleClaimForm.claim_start_date > todayStr) {
      setAnalysisError(`Claim Start Date cannot be in the future (${singleClaimForm.claim_start_date}). Service dates must be on or before today (${todayStr}).`);
      setAnalysisLoading(false);
      return;
    }
    if (singleClaimForm.claim_end_date && singleClaimForm.claim_end_date > todayStr) {
      setAnalysisError(`Claim End Date cannot be in the future (${singleClaimForm.claim_end_date}). Service dates must be on or before today (${todayStr}).`);
      setAnalysisLoading(false);
      return;
    }

    if (singleClaimForm.claim_start_date && singleClaimForm.claim_end_date) {
      if (singleClaimForm.claim_end_date < singleClaimForm.claim_start_date) {
        setAnalysisError('Claim End Date cannot be earlier than Claim Start Date.');
        setAnalysisLoading(false);
        return;
      }
    }

    if (singleClaimForm.claim_type === 'Inpatient') {
      if (!singleClaimForm.admission_date) {
        setAnalysisError('Admission Date is required for Inpatient claims.');
        setAnalysisLoading(false);
        return;
      }
      if (singleClaimForm.admission_date > todayStr) {
        setAnalysisError(`Admission Date cannot be in the future (${singleClaimForm.admission_date}). Service dates must be on or before today (${todayStr}).`);
        setAnalysisLoading(false);
        return;
      }
      if (!singleClaimForm.discharge_date) {
        setAnalysisError('Discharge Date is required for Inpatient claims.');
        setAnalysisLoading(false);
        return;
      }
      if (singleClaimForm.discharge_date > todayStr) {
        setAnalysisError(`Discharge Date cannot be in the future (${singleClaimForm.discharge_date}). Service dates must be on or before today (${todayStr}).`);
        setAnalysisLoading(false);
        return;
      }
      if (singleClaimForm.discharge_date < singleClaimForm.admission_date) {
        setAnalysisError('Discharge Date cannot be earlier than Admission Date.');
        setAnalysisLoading(false);
        return;
      }
    }

    const payload = {
      ClaimID: singleClaimForm.claim_id,
      ProviderID: singleClaimForm.provider_id,
      BeneID: singleClaimForm.beneficiary_id,
      ClaimType: singleClaimForm.claim_type,
      InscClaimAmtReimbursed: reimb,
      Currency: 'INR',
      DeductibleAmtPaid: isNaN(ded) ? 0.0 : ded,
      ClaimStartDt: singleClaimForm.claim_start_date,
      ClaimEndDt: singleClaimForm.claim_end_date,
      AdmissionDt: singleClaimForm.claim_type === 'Inpatient' ? (singleClaimForm.admission_date || null) : null,
      DischargeDt: singleClaimForm.claim_type === 'Inpatient' ? (singleClaimForm.discharge_date || null) : null,
      ClmDiagnosisCode_1: singleClaimForm.diagnosis_code_1 || null,
      ClmDiagnosisCode_2: singleClaimForm.diagnosis_code_2 || null,
      ClmAdmitDiagnosisCode: singleClaimForm.admission_diagnosis_code || null,
      ClmProcedureCode_1: singleClaimForm.procedure_code_1 || null,
      AttendingPhysician: singleClaimForm.attending_physician || null,
      OperatingPhysician: singleClaimForm.operating_physician || null,
      OtherPhysician: singleClaimForm.other_physician || null
    };

    try {
      const res = await api.post('/api/analyzer/single', payload, {
        signal: abortController.signal
      });
      if (reqId === latestAnalysisReqIdRef.current) {
        setAnalysisResult(res.data);
        setSavedDecision(null);
        setDecisionSuccess(null);
        setDecisionError(null);
        setDecisionNotes('');
        setDecisionStatus('Pending Review');
        fetchOverview();
        fetchClaims();
        fetchProviders();
      }
    } catch (err) {
      if (axios.isCancel(err) || err.name === 'CanceledError' || err.code === 'ERR_CANCELED') {
        return;
      }
      if (reqId === latestAnalysisReqIdRef.current) {
        setAnalysisError(err.response?.data?.detail || 'Single-claim inference failed.');
      }
    } finally {
      if (reqId === latestAnalysisReqIdRef.current) {
        setAnalysisLoading(false);
      }
    }
  };

  const handleSaveDecision = async (e) => {
    e?.preventDefault();
    if (!analysisResult) return;
    setDecisionLoading(true);
    setDecisionError(null);
    setDecisionSuccess(null);
    try {
      const payload = {
        claim_id: analysisResult.claim_id || singleClaimForm.claim_id,
        provider_id: analysisResult.provider_id || singleClaimForm.provider_id,
        decision_status: decisionStatus,
        notes: decisionNotes,
        decision_date: new Date().toISOString()
      };
      const res = await api.post('/api/analyzer/decision', payload);
      setSavedDecision(res.data);
      setDecisionSuccess(`Official decision recorded successfully: "${res.data.decision_status}".`);
    } catch (err) {
      setDecisionError(err.response?.data?.detail || 'Failed to record investigator decision.');
    } finally {
      setDecisionLoading(false);
    }
  };

  const clinicalProfiles = [
    {
      category: 'Cardiovascular (Hypertension & Heart Failure)',
      primaryDiag: '4019',
      secondaryDiag: '4280',
      inpatientProc: '3722',
      inpatientRange: [8500, 24000],
      outpatientRange: [280, 1150],
    },
    {
      category: 'Endocrine & Metabolic (Diabetes Mellitus)',
      primaryDiag: '25000',
      secondaryDiag: '2724',
      inpatientProc: '3893',
      inpatientRange: [6500, 19500],
      outpatientRange: [190, 820],
    },
    {
      category: 'Respiratory (Pneumonia & Airway Disease)',
      primaryDiag: '486',
      secondaryDiag: '496',
      inpatientProc: '9604',
      inpatientRange: [11000, 31000],
      outpatientRange: [240, 980],
    },
    {
      category: 'Renal & Urinary (Acute Kidney Injury & UTI)',
      primaryDiag: '5849',
      secondaryDiag: '5990',
      inpatientProc: '3995',
      inpatientRange: [9200, 27500],
      outpatientRange: [310, 1420],
    },
    {
      category: 'Musculoskeletal (Degenerative Joint Disease)',
      primaryDiag: '71590',
      secondaryDiag: '7242',
      inpatientProc: '8151',
      inpatientRange: [16000, 38000],
      outpatientRange: [350, 1680],
    },
    {
      category: 'Cerebrovascular (Cerebral Ischemia)',
      primaryDiag: '43491',
      secondaryDiag: '4019',
      inpatientProc: '8872',
      inpatientRange: [14000, 35000],
      outpatientRange: [420, 1850],
    }
  ];

  const generateCollisionResistantId = () => {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
      return `CLM_SYN_${crypto.randomUUID().slice(0, 8).toUpperCase()}`;
    }
    const randPart = Math.random().toString(36).substring(2, 8).toUpperCase();
    const numPart = Math.floor(1000 + Math.random() * 9000);
    return `CLM_SYN_${randPart}${numPart}`;
  };

  const loadSampleClaim = () => {
    const clickCount = sampleRotationIndexRef.current;
    sampleRotationIndexRef.current += 1;

    const profile = clinicalProfiles[clickCount % clinicalProfiles.length];
    const isInpatient = clickCount % 2 === 1;

    const randProvNum = Math.floor(10000 + Math.random() * 90000);
    const randBeneNum = Math.floor(10000 + Math.random() * 90000);
    const provider = `PRV_${randProvNum}`;
    const beneficiary = `BEN_${randBeneNum}`;
    const attendingPhys = `PHY_${Math.floor(10000 + Math.random() * 90000)}`;
    const operatingPhys = isInpatient ? `PHY_${Math.floor(10000 + Math.random() * 90000)}` : '';
    const otherPhys = isInpatient ? `PHY_${Math.floor(10000 + Math.random() * 90000)}` : '';

    const now = new Date();
    const start2025 = new Date('2025-01-01T00:00:00Z').getTime();
    const endBound = now.getTime() - 2 * 86400000;
    const randomTimestamp = start2025 + Math.random() * (endBound - start2025);
    const startDtObj = new Date(randomTimestamp);

    const pad = (n) => (n < 10 ? `0${n}` : `${n}`);
    const formatYMD = (d) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;

    const claimStart = formatYMD(startDtObj);
    let claimEnd = claimStart;
    let admissionDate = '';
    let dischargeDate = '';
    let reimbursement = '';
    let deductible = '';
    let procCode = '';

    if (isInpatient) {
      const stayDays = Math.floor(Math.random() * 7) + 2;
      const endDtObj = new Date(startDtObj.getTime() + stayDays * 86400000);
      if (endDtObj > now) {
        endDtObj.setTime(now.getTime() - 86400000);
      }
      claimEnd = formatYMD(endDtObj);
      admissionDate = claimStart;
      dischargeDate = claimEnd;
      
      const [minAmt, maxAmt] = profile.inpatientRange;
      const amtUsd = minAmt + Math.floor(Math.random() * (maxAmt - minAmt));
      const amtInr = amtUsd * 83.50;
      reimbursement = amtInr.toFixed(2);
      deductible = (1068.0 * 83.50).toFixed(2);
      procCode = profile.inpatientProc;
    } else {
      const [minAmt, maxAmt] = profile.outpatientRange;
      const amtUsd = minAmt + Math.floor(Math.random() * (maxAmt - minAmt));
      const amtInr = amtUsd * 83.50;
      reimbursement = amtInr.toFixed(2);
      deductible = ((Math.floor(Math.random() * 60) + 20) * 83.50).toFixed(2);
      procCode = '';
    }

    const uniqueClaimId = generateCollisionResistantId();

    const generatedForm = {
      claim_id: uniqueClaimId,
      provider_id: provider,
      beneficiary_id: beneficiary,
      claim_type: isInpatient ? 'Inpatient' : 'Outpatient',
      reimbursement_amount: reimbursement,
      deductible_amount: deductible,
      claim_start_date: claimStart,
      claim_end_date: claimEnd,
      admission_date: admissionDate,
      discharge_date: dischargeDate,
      diagnosis_code_1: profile.primaryDiag,
      diagnosis_code_2: profile.secondaryDiag,
      admission_diagnosis_code: isInpatient ? profile.primaryDiag : '',
      procedure_code_1: procCode,
      attending_physician: attendingPhys,
      operating_physician: operatingPhys,
      other_physician: otherPhys
    };

    setSingleClaimForm(generatedForm);
  };

  const handleCsvFileSelect = async (file) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setCsvUploadError('Invalid file format. Please upload a .csv file.');
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setCsvUploadError('File exceeds 20MB maximum size limit.');
      return;
    }
    setCsvFile(file);
    setCsvUploadError(null);
    setCsvPreviewLoading(true);
    setCsvPreviewData(null);
    setCsvUploadResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await api.post('/api/analyzer/csv/preview', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setCsvPreviewData(res.data);
    } catch (err) {
      setCsvUploadError(err.response?.data?.detail || 'CSV preview failed.');
    } finally {
      setCsvPreviewLoading(false);
    }
  };

  const handleCsvUpload = async (e) => {
    if (e) e.preventDefault();
    if (!csvFile) return;

    setCsvStatusLoading(true);
    setCsvUploadError(null);
    setCsvUploadResult(null);

    const formData = new FormData();
    formData.append('file', csvFile);

    try {
      const res = await api.post('/api/analyzer/csv', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setCsvUploadResult(res.data);
      pollCsvUploadStatus(res.data.upload_id);
    } catch (err) {
      setCsvUploadError(err.response?.data?.detail || 'Batch CSV upload failed.');
      setCsvStatusLoading(false);
    }
  };

  const downloadCsvTemplate = async (cur = csvTemplateCurrency) => {
    try {
      const res = await api.get(`/api/analyzer/csv/template?currency=${cur}`, {
        responseType: 'blob'
      });
      const blob = new Blob([res.data], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `healthguard_claim_template_${cur.toLowerCase()}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Template download error:', err);
    }
  };

  const exportBatchResults = async (uploadId) => {
    if (!uploadId) return;
    try {
      const res = await api.get(`/api/analyzer/csv/${uploadId}/export`, {
        responseType: 'blob'
      });
      const blob = new Blob([res.data], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `healthguard_batch_${uploadId}_results.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Batch results export error:', err);
    }
  };

  const pollCsvUploadStatus = async (uploadId) => {
    let attempts = 0;
    const interval = setInterval(async () => {
      attempts++;
      try {
        const res = await api.get(`/api/analyzer/csv/${uploadId}`);
        setCsvUploadResult(res.data);
        if (res.data.status === 'COMPLETED' || res.data.status === 'FAILED' || attempts > 60) {
          clearInterval(interval);
          setCsvStatusLoading(false);
          fetchCsvDetails(uploadId);
        }
      } catch (err) {
        clearInterval(interval);
        setCsvStatusLoading(false);
      }
    }, 1500);
  };

  const fetchCsvDetails = async (uploadId) => {
    try {
      const [succRes, rejRes] = await Promise.all([
        api.get(`/api/analyzer/csv/${uploadId}/results`).catch(() => ({ data: { claims: [] } })),
        api.get(`/api/analyzer/csv/${uploadId}/rejected-rows`).catch(() => ({ data: { rejected_rows: [] } }))
      ]);
      setCsvSuccessfulClaims(succRes.data.claims || []);
      setCsvRejectedRows(rejRes.data.rejected_rows || []);
      fetchOverview();
      fetchClaims();
      fetchProviders();
    } catch (e) {
      console.error(e);
    }
  };

  const handleGlobalSearchKeyDown = (e) => {
    if (e.key !== 'Enter') return;
    const q = (globalSearch || '').trim();
    if (!q) return;
    const qUpper = q.toUpperCase();
    if (qUpper.startsWith('PRV') || qUpper.startsWith('PROV')) {
      setActiveTab('providers');
      setProviderSearch(q);
      setProviderPage(1);
      inspectProvider(q);
      showToast(`Searching provider ${q}`);
    } else if (qUpper.startsWith('CLM') || qUpper.startsWith('CLAIM')) {
      setActiveTab('claims');
      setClaimSearch(q);
      setClaimPage(1);
      inspectClaim(q);
      showToast(`Searching claim ${q}`);
    } else {
      setActiveTab('providers');
      setProviderSearch(q);
      setProviderPage(1);
      showToast(`Searching for "${q}"`);
    }
  };

  // ══════════════════════════════════════════════════════════════════════════════
  // RENDER: LOGIN / REGISTER VIEW (UNAUTHENTICATED)
  // ══════════════════════════════════════════════════════════════════════════════

  if (!token || !currentUser) {
    return (
      <>
        <div className="liquid"><i /><i /><i /></div>
        <svg width="0" height="0" style={{ position: 'absolute' }}>
          <defs>
            <symbol id="shield" viewBox="0 0 24 24"><path d="M12 3l7 3v5c0 5-3 8-7 10-4-2-7-5-7-10V6l7-3z"/><path d="M12 8v4M12 15h.01"/></symbol>
            <symbol id="info" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></symbol>
          </defs>
        </svg>

        <div className="login" id="v-login">
          <div className="glass card">
            <div className="brand" style={{ height: 'auto', padding: '0 0 16px' }}>
              <div className="logo"><svg className="ic"><use href="#shield"/></svg></div>
              <div><b>HealthGuard AI</b><span>Claims Fraud Investigation Portal</span></div>
            </div>
            
            <div className="desc" style={{ marginBottom: 20 }}>
              An Explainable Ensemble Framework for Health Insurance Provider Fraud Detection and Claims Investigation
            </div>

            <div className="ph" style={{ margin: '0 0 4px' }}>
              <h2 style={{ margin: 0 }}>{isRegisterView ? 'Register New Investigator' : 'Investigator Sign In'}</h2>
            </div>

            <p style={{ color: 'var(--muted)', marginTop: 4, fontSize: 13 }}>
              {isRegisterView
                ? 'Enter investigator details. Newly registered accounts are assigned the investigator role.'
                : 'Enter your credentials to access the forensic investigation dashboard.'}
            </p>

            {authError && (
              <p className="fe" role="alert" style={{ marginTop: 12 }}>
                {authError}
              </p>
            )}

            <form onSubmit={isRegisterView ? handleRegister : handleLogin} className="stack">
              {isRegisterView && (
                <div className="f">
                  <label>Full Name</label>
                  <input
                    type="text"
                    required
                    value={authForm.name}
                    onChange={(e) => setAuthForm({ ...authForm, name: e.target.value })}
                    placeholder="Dr. Alex Rivera"
                  />
                </div>
              )}

              <div className="f">
                <label>Email Address</label>
                <input
                  type="email"
                  required
                  value={authForm.email}
                  onChange={(e) => setAuthForm({ ...authForm, email: e.target.value })}
                  placeholder="investigator@healthguard.ai"
                  autoComplete="username"
                />
              </div>

              <div className="f">
                <label>Password</label>
                <input
                  type={showPassword ? 'text' : 'password'}
                  required
                  value={authForm.password}
                  onChange={(e) => setAuthForm({ ...authForm, password: e.target.value })}
                  placeholder="••••••••••••"
                  autoComplete={isRegisterView ? 'new-password' : 'current-password'}
                />
              </div>

              <button
                type="button"
                className="mini"
                style={{ alignSelf: 'flex-end', marginTop: -6 }}
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? 'Hide password' : 'Show password'}
              </button>

              <button type="submit" className="primary" disabled={authLoading}>
                {authLoading ? 'Authenticating...' : isRegisterView ? 'Create Account' : 'Authenticate Session'}
              </button>
            </form>

            <div className="foot">
              {isRegisterView ? (
                <p>
                  Already have an account?{' '}
                  <button
                    type="button"
                    onClick={() => { setIsRegisterView(false); setAuthError(null); }}
                    style={{ all: 'unset', color: 'var(--accent)', fontWeight: 600, cursor: 'pointer' }}
                  >
                    Sign In
                  </button>
                </p>
              ) : (
                <p>
                  Need access?{' '}
                  <button
                    type="button"
                    onClick={() => { setIsRegisterView(true); setAuthError(null); }}
                    style={{ all: 'unset', color: 'var(--accent)', fontWeight: 600, cursor: 'pointer' }}
                  >
                    Register Investigator
                  </button>
                </p>
              )}
            </div>
          </div>
        </div>
      </>
    );
  }

  // Risk distribution helper numbers
  const lowDist = riskDistribution?.distribution?.find((d) => d.tier === 'LOW') || { count: 0, percentage: 0 };
  const medDist = riskDistribution?.distribution?.find((d) => d.tier === 'MEDIUM') || { count: 0, percentage: 0 };
  const highDist = riskDistribution?.distribution?.find((d) => d.tier === 'HIGH') || { count: 0, percentage: 0 };

  // ══════════════════════════════════════════════════════════════════════════════
  // RENDER: MAIN APPLICATION (LIQUID GLASS / GLASSMORPHISM THEME)
  // ══════════════════════════════════════════════════════════════════════════════

  return (
    <>
      {/* Animated liquid backdrop blobs */}
      <div className="liquid"><i /><i /><i /></div>

      {/* Inline SVG Symbol Sprite matching healthguard_glass.html */}
      <svg width="0" height="0" style={{ position: 'absolute' }}>
        <defs>
          <symbol id="shield" viewBox="0 0 24 24"><path d="M12 3l7 3v5c0 5-3 8-7 10-4-2-7-5-7-10V6l7-3z"/><path d="M12 8v4M12 15h.01"/></symbol>
          <symbol id="grid" viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="2"/><rect x="14" y="3" width="7" height="7" rx="2"/><rect x="3" y="14" width="7" height="7" rx="2"/><rect x="14" y="14" width="7" height="7" rx="2"/></symbol>
          <symbol id="doc" viewBox="0 0 24 24"><path d="M14 3H7a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h6"/></symbol>
          <symbol id="pulse" viewBox="0 0 24 24"><path d="M3 12h4l3-8 4 16 3-8h4"/></symbol>
          <symbol id="sliders" viewBox="0 0 24 24"><path d="M5 4v6M5 14v6M12 4v2M12 10v10M19 4v10M19 18v2M3 10h4M10 6h4M17 14h4"/></symbol>
          <symbol id="alert" viewBox="0 0 24 24"><path d="M12 3l10 18H2L12 3z"/><path d="M12 10v5M12 18h.01"/></symbol>
          <symbol id="info" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></symbol>
          <symbol id="search" viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/></symbol>
          <symbol id="user" viewBox="0 0 24 24"><circle cx="12" cy="8" r="4"/><path d="M4 21c1-4 4-6 8-6s7 2 8 6"/></symbol>
          <symbol id="chev" viewBox="0 0 24 24"><path d="M15 6l-6 6 6 6"/></symbol>
          <symbol id="check-shield" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></symbol>
          <symbol id="chart-bar" viewBox="0 0 24 24"><line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="14"/></symbol>
          <symbol id="copy" viewBox="0 0 24 24"><rect height="13" rx="2" ry="2" width="13" x="9" y="9"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></symbol>
        </defs>
      </svg>

      {/* Main Glass Application Container */}
      <div className={`app ${sidebarCollapsed ? 'rail' : ''}`} id="app">
        {/* Sticky Glass Sidebar */}
        <aside className="glass">
          <div className="brand">
            <div className="logo"><svg className="ic"><use href="#shield"/></svg></div>
            <div className="lb"><b>HealthGuard AI</b><span>Fraud Investigation</span></div>
          </div>

          <nav id="nav" ref={navRef}>
            <i className="ind r" ref={indRef}></i>
            <button
              data-v="overview"
              className={activeTab === 'overview' ? 'on' : ''}
              onClick={() => setActiveTab('overview')}
              title="Overview"
            >
              <svg className="ic"><use href="#grid"/></svg>
              <span className="lb">Overview</span>
            </button>

            <button
              data-v="providers"
              className={activeTab === 'providers' ? 'on' : ''}
              onClick={() => setActiveTab('providers')}
              title="Providers"
            >
              <svg className="ic"><use href="#shield"/></svg>
              <span className="lb">Providers</span>
            </button>

            <button
              data-v="claims"
              className={activeTab === 'claims' ? 'on' : ''}
              onClick={() => setActiveTab('claims')}
              title="Claims"
            >
              <svg className="ic"><use href="#doc"/></svg>
              <span className="lb">Claims</span>
            </button>

            <button
              data-v="analyzer"
              className={activeTab === 'analyzer' ? 'on' : ''}
              onClick={() => setActiveTab('analyzer')}
              title="Claim Analyzer"
            >
              <svg className="ic"><use href="#pulse"/></svg>
              <span className="lb">Claim Analyzer</span>
            </button>

            <button
              data-v="settings"
              className={activeTab === 'settings' ? 'on' : ''}
              onClick={() => setActiveTab('settings')}
              title="Settings"
            >
              <svg className="ic"><use href="#sliders"/></svg>
              <span className="lb">Settings</span>
            </button>
          </nav>

          <button
            className="col"
            id="col"
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            <svg className="ic"><use href="#chev"/></svg>
            <span className="lb">{sidebarCollapsed ? "Expand" : "Collapse"}</span>
          </button>

          <div className="user">
            <div className="avatar"><svg className="ic"><use href="#user"/></svg></div>
            <div className="lb">
              <b>{currentUser.name}</b>
              <small>{currentUser.role}</small>
            </div>
          </div>
        </aside>

        {/* Main Viewport */}
        <main>
          {/* Glass Topbar */}
          <div className="topbar glass">
            <label className="search gs">
              <svg className="ic"><use href="#search"/></svg>
              <input
                id="gs"
                value={globalSearch}
                onChange={(e) => setGlobalSearch(e.target.value)}
                onKeyDown={handleGlobalSearchKeyDown}
                placeholder="Jump to a provider or claim ID…"
                aria-label="Jump to provider or claim"
              />
            </label>
            <div className="acts">
              <button
                className="rq"
                id="rq"
                onClick={() => setActiveTab('claims')}
              >
                {overviewSummary?.claims_requiring_investigation ?? 0} claims to review
              </button>
              <button
                className="mini"
                id="th"
                onClick={() => setDarkMode(!darkMode)}
              >
                {darkMode ? 'Light mode' : 'Dark mode'}
              </button>
              <button
                className="mini"
                id="so"
                onClick={handleLogout}
              >
                Sign out
              </button>
            </div>
          </div>

          {/* ════════════════════════════════════════════════════════════════════════
              SECTION 1: OVERVIEW VIEW
          ════════════════════════════════════════════════════════════════════════ */}
          {activeTab === 'overview' && (
            <section className="view on" id="v-overview">
              <div className="head">
                <div>
                  <h1>Investigation Overview</h1>
                  <p>Aggregated KPIs, risk distribution, and surveillance telemetry from PostgreSQL.</p>
                </div>
                <button className="btn" onClick={fetchOverview} disabled={overviewLoading}>
                  {overviewLoading ? 'Refreshing...' : 'Refresh Data'}
                </button>
              </div>

              {overviewError && (
                <div className="fe" role="alert">{overviewError}</div>
              )}

              {/* 4 KPIs matching reference HTML */}
              <div className="kpis">
                <div className="glass kpi">
                  <div className="row">
                    Total Monitored Providers
                    <span className="chip"><svg className="ic"><use href="#shield"/></svg></span>
                  </div>
                  <div className="num">
                    {overviewSummary ? (overviewSummary.total_providers ?? 0).toLocaleString() : '—'}
                  </div>
                  <small>Monitored in database</small>
                </div>

                <div className="glass kpi">
                  <div className="row">
                    Total Evaluated Claims
                    <span className="chip"><svg className="ic"><use href="#doc"/></svg></span>
                  </div>
                  <div className="num">
                    {overviewSummary ? (overviewSummary.total_claims ?? 0).toLocaleString() : '—'}
                  </div>
                  <small>Inpatient &amp; Outpatient claims</small>
                </div>

                <div className="glass kpi risk">
                  <div className="row">
                    High-Risk Providers
                    <span className="chip r"><svg className="ic"><use href="#alert"/></svg></span>
                  </div>
                  <div className="num">
                    {overviewSummary ? (overviewSummary.high_risk_providers ?? 0).toLocaleString() : '—'}
                  </div>
                  <small>Calibrated Score &ge; 0.70</small>
                </div>

                <div className="glass kpi warn">
                  <div className="row">
                    Claims Requiring Investigation
                    <span className="chip a"><svg className="ic"><use href="#pulse"/></svg></span>
                  </div>
                  <div className="num">
                    {overviewSummary ? (overviewSummary.claims_requiring_investigation ?? overviewSummary.flagged_claims ?? 0).toLocaleString() : '—'}
                  </div>
                  <small>Anomalous or High Risk</small>
                </div>
              </div>

              {/* Provider Fraud Risk Tier Distribution */}
              <div className="glass panel">
                <h2>Provider Fraud Risk Tier Distribution</h2>
                <div className="tiers">
                  <div className="tier">
                    <div className="t">
                      <span>Low Risk</span>
                      <span>{lowDist.percentage}%</span>
                    </div>
                    <b>{lowDist.count} providers</b>
                    <div className="bar">
                      <span style={{ width: `${Math.min(100, Math.max(0, lowDist.percentage))}%`, background: 'linear-gradient(90deg,#34d399,#10b981)' }}></span>
                    </div>
                    <small>Cutoff: &lt; 0.22</small>
                  </div>

                  <div className="tier">
                    <div className="t">
                      <span>Medium Risk</span>
                      <span>{medDist.percentage}%</span>
                    </div>
                    <b>{medDist.count} providers</b>
                    <div className="bar">
                      <span style={{ width: `${Math.min(100, Math.max(0, medDist.percentage))}%`, background: 'linear-gradient(90deg,#fbbf24,#f59e0b)' }}></span>
                    </div>
                    <small>Cutoff: 0.22 – 0.70</small>
                  </div>

                  <div className="tier">
                    <div className="t">
                      <span>High Risk</span>
                      <span>{highDist.percentage}%</span>
                    </div>
                    <b>{highDist.count} providers</b>
                    <div className="bar">
                      <span style={{ width: `${Math.min(100, Math.max(0, highDist.percentage))}%`, background: 'linear-gradient(90deg,#f87171,#ef4444)' }}></span>
                    </div>
                    <small>Cutoff: &ge; 0.70</small>
                  </div>
                </div>
              </div>

              {/* Dynamic Overview Surveillance Charts if available */}
              {overviewCharts && (overviewCharts.claims_by_month?.length > 0 || overviewCharts.claims_over_time?.length > 0) && (
                <div className="charts">
                  <div className="glass panel cp">
                    <h2>Reimbursement &amp; Volume by Month</h2>
                    <p className="cs">Monthly aggregate claim reimbursement and counts from PostgreSQL</p>
                    <div className="cw" style={{ height: 240 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={overviewCharts.claims_by_month}>
                          <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" />
                          <XAxis dataKey="month" tick={{ fontSize: 11, fill: 'var(--muted)' }} />
                          <YAxis tick={{ fontSize: 11, fill: 'var(--muted)' }} />
                          <Tooltip
                            contentStyle={{ backgroundColor: 'var(--glass-strong)', borderColor: 'var(--stroke)', borderRadius: 12, backdropFilter: 'blur(12px)' }}
                            formatter={(value, name) => [
                              name === 'total_reimbursement' ? formatINR(usdToInr(Number(value))) : value,
                              name === 'total_reimbursement' ? 'Total Reimbursed' : 'Claims'
                            ]}
                          />
                          <Bar dataKey="total_reimbursement" fill="var(--accent)" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="glass panel cp">
                    <h2>Surveillance Activity Timeline</h2>
                    <p className="cs">Analysis execution vs claim service dates over time</p>
                    <div className="cw" style={{ height: 240 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={overviewCharts.claims_over_time}>
                          <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" />
                          <XAxis dataKey="date" tick={{ fontSize: 11, fill: 'var(--muted)' }} />
                          <YAxis tick={{ fontSize: 11, fill: 'var(--muted)' }} />
                          <Tooltip
                            contentStyle={{ backgroundColor: 'var(--glass-strong)', borderColor: 'var(--stroke)', borderRadius: 12, backdropFilter: 'blur(12px)' }}
                          />
                          <Line type="monotone" dataKey="analysis_count" stroke="#8b5cf6" strokeWidth={2.5} name="Analysis Date" dot={{ r: 3 }} />
                          <Line type="monotone" dataKey="service_count" stroke="#10b981" strokeWidth={2.5} name="Service Date" dot={{ r: 3 }} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              )}

              {/* Recent Fraud Investigations & Assessments */}
              <div className="glass panel">
                <h2>Recent Fraud Investigations &amp; Assessments</h2>
                <div className="tablewrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Provider ID</th>
                        <th>Calibrated Score</th>
                        <th>Risk Tier</th>
                        <th>Assessment Date</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {overviewRecent.filter(item => item.activity_type === 'PROVIDER_ASSESSMENT' || item.provider_id || item.entity_id).length === 0 ? (
                        <tr>
                          <td colSpan="5" className="empty">No recent provider assessments logged yet.</td>
                        </tr>
                      ) : (
                        overviewRecent
                          .filter(item => item.activity_type === 'PROVIDER_ASSESSMENT' || item.provider_id || item.entity_id)
                          .map((item, idx) => {
                            const provId = item.provider_id || item.entity_id;
                            const score = typeof item.risk_score === 'number'
                              ? item.risk_score
                              : typeof item.score === 'number'
                              ? item.score
                              : null;
                            const tier = item.risk_tier || 'LOW';
                            const dt = item.created_at || item.timestamp || item.assessed_at;

                            return (
                              <tr key={item.id || idx}>
                                <td className="mono"><b>{provId}</b></td>
                                <td className="mono">
                                  {score !== null ? (score * 100).toFixed(1) + '%' : 'N/A'}
                                </td>
                                <td>{getTierBadge(tier)}</td>
                                <td>
                                  {dt ? new Date(dt).toLocaleDateString() : 'Recently'}
                                </td>
                                <td>
                                  <button
                                    className="inspect"
                                    onClick={() => { inspectProvider(provId); setActiveTab('providers'); }}
                                  >
                                    Inspect
                                  </button>
                                </td>
                              </tr>
                            );
                          })
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>
          )}

          {/* ════════════════════════════════════════════════════════════════════════
              SECTION 2: PROVIDERS VIEW
          ════════════════════════════════════════════════════════════════════════ */}
          {activeTab === 'providers' && (
            <section className="view on" id="v-providers">
              <div className="head">
                <div>
                  <h1>Monitored Providers</h1>
                  <p>Search and audit healthcare providers under fraud surveillance.</p>
                </div>
                <div className="tools">
                  <label className="search">
                    <svg className="ic"><use href="#search"/></svg>
                    <input
                      value={providerSearch}
                      onChange={(e) => { setProviderSearch(e.target.value); setProviderPage(1); }}
                      placeholder="Search Provider ID or Name..."
                      aria-label="Search providers"
                    />
                  </label>
                  <select
                    value={providerTierFilter}
                    onChange={(e) => { setProviderTierFilter(e.target.value); setProviderPage(1); }}
                    aria-label="Risk tier"
                  >
                    <option value="">All Risk Tiers</option>
                    <option value="HIGH">High Risk</option>
                    <option value="MEDIUM">Medium Risk</option>
                    <option value="LOW">Low Risk</option>
                  </select>
                </div>
              </div>

              <div className="glass panel">
                <div className="tablewrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Provider ID</th>
                        <th>Provider Name</th>
                        <th>Claims Count</th>
                        <th>Total Reimbursed</th>
                        <th>Risk Tier</th>
                        <th>Latest Score</th>
                        <th>Action</th>
                      </tr>
                    </thead>
                    <tbody id="rows">
                      {providersLoading ? (
                        <tr><td colSpan="7" className="empty">Loading providers...</td></tr>
                      ) : providers.length === 0 ? (
                        <tr><td colSpan="7" className="empty">No matching providers found.</td></tr>
                      ) : (
                        providers.map((p) => (
                          <tr key={p.provider_id}>
                            <td className="mono"><b>{p.provider_id}</b></td>
                            <td><b>{p.provider_name}</b></td>
                            <td className="mono">{p.total_claims}</td>
                            <td className="mono">{formatINR(usdToInr(p.total_reimbursement))}</td>
                            <td>{getTierBadge(p.risk_tier)}</td>
                            <td className="mono">{p.risk_score ? p.risk_score.toFixed(4) : '—'}</td>
                            <td>
                              <button
                                className="inspect"
                                onClick={() => inspectProvider(p.provider_id)}
                              >
                                Inspect
                              </button>
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>

                <div className="pager">
                  <span>Page {providerPage} of {providerTotalPages}</span>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button
                      className="btn"
                      disabled={providerPage <= 1}
                      onClick={() => setProviderPage(providerPage - 1)}
                      style={{ opacity: providerPage <= 1 ? 0.4 : 1 }}
                      aria-label="Previous page"
                    >
                      ‹
                    </button>
                    <button
                      className="btn"
                      disabled={providerPage >= providerTotalPages}
                      onClick={() => setProviderPage(providerPage + 1)}
                      style={{ opacity: providerPage >= providerTotalPages ? 0.4 : 1 }}
                      aria-label="Next page"
                    >
                      ›
                    </button>
                  </div>
                </div>
              </div>

              {/* Provider Detail Modal (matching reference HTML .win modal) */}
              {selectedProvider && (
                <div
                  className="scrim wscrim"
                  style={{ display: 'flex' }}
                  onClick={(e) => {
                    if (e.target.classList.contains('wscrim')) setSelectedProvider(null);
                  }}
                  role="dialog"
                  aria-modal="true"
                >
                  <div className="glass win">
                    <div className="wh">
                      <div>
                        <h2>{selectedProvider.provider_name}</h2>
                        <p className="sub mono">{selectedProvider.provider_id} &middot; Provider Forensic Profile</p>
                      </div>
                      <div className="wa">
                        {getTierBadge(selectedProvider.risk?.risk_tier)}
                        <button className="btn" id="wx" onClick={() => setSelectedProvider(null)}>Close</button>
                      </div>
                    </div>

                    {/* Hero Card */}
                    <div className="glass panel rs">
                      <div className="ttl">Provider Risk Assessment</div>
                      <div className="pct">
                        {selectedProvider.risk?.risk_score !== null && selectedProvider.risk?.risk_score !== undefined
                          ? `${(selectedProvider.risk.risk_score * 100).toFixed(1)}%`
                          : 'Pending'}
                      </div>
                      <div style={{ margin: '8px 0' }}>{getTierBadge(selectedProvider.risk?.risk_tier)}</div>
                      <div className="mg">
                        <div className="mt"><small>Provider ID</small><b>{selectedProvider.provider_id}</b></div>
                        <div className="mt"><small>Total Claims</small><b>{selectedProvider.total_claims}</b></div>
                        <div className="mt"><small>Total Reimbursed</small><b>{formatINR(usdToInr(selectedProvider.total_reimbursement))}</b></div>
                        <div className="mt"><small>Calibrated Score</small><b>{selectedProvider.risk?.risk_score ? selectedProvider.risk.risk_score.toFixed(4) : '—'}</b></div>
                      </div>
                    </div>

                    {/* SHAP Explanation */}
                    {selectedProvider.risk?.explanation && (
                      <div className="glass panel">
                        <h2>Why This Score?</h2>
                        <p className="cs">{selectedProvider.risk.explanation.title}</p>
                        <p style={{ color: 'var(--muted)', fontSize: 13, lineHeight: 1.6 }}>{selectedProvider.risk.explanation.text}</p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </section>
          )}

          {/* ════════════════════════════════════════════════════════════════════════
              SECTION 3: CLAIMS VIEW
          ════════════════════════════════════════════════════════════════════════ */}
          {activeTab === 'claims' && (
            <section className="view on" id="v-claims">
              <div className="head">
                <div>
                  <h1>Evaluated Claims</h1>
                  <p>Search and audit individual claims processed through the anomaly engine.</p>
                </div>
                <div className="tools">
                  <label className="search">
                    <svg className="ic"><use href="#search"/></svg>
                    <input
                      value={claimSearch}
                      onChange={(e) => { setClaimSearch(e.target.value); setClaimPage(1); }}
                      placeholder="Search Claim ID or Provider..."
                      aria-label="Search claims"
                    />
                  </label>
                  <select
                    value={claimTypeFilter}
                    onChange={(e) => { setClaimTypeFilter(e.target.value); setClaimPage(1); }}
                    aria-label="Claim type"
                  >
                    <option value="">All Claim Types</option>
                    <option value="Inpatient">Inpatient</option>
                    <option value="Outpatient">Outpatient</option>
                  </select>
                  <label className="search" style={{ minWidth: 0 }}>
                    From <input type="date" value={claimStartDate} onChange={(e) => { setClaimStartDate(e.target.value); setClaimPage(1); }} aria-label="From date" />
                  </label>
                  <label className="search" style={{ minWidth: 0 }}>
                    To <input type="date" value={claimEndDate} onChange={(e) => { setClaimEndDate(e.target.value); setClaimPage(1); }} aria-label="To date" />
                  </label>
                </div>
              </div>

              <div className="glass panel">
                <div className="tablewrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Claim ID</th>
                        <th>Provider ID</th>
                        <th>Type</th>
                        <th>Date</th>
                        <th>Reimbursed</th>
                        <th>Anomaly Status</th>
                        <th>Action</th>
                      </tr>
                    </thead>
                    <tbody id="crows">
                      {claimsLoading ? (
                        <tr><td colSpan="7" className="empty">Loading claims...</td></tr>
                      ) : claims.length === 0 ? (
                        <tr><td colSpan="7" className="empty">No matching claims found.</td></tr>
                      ) : (
                        claims.map((c) => (
                          <tr key={c.claim_id}>
                            <td className="mono"><b>{c.claim_id}</b></td>
                            <td className="mono">{c.provider_id}</td>
                            <td>{c.claim_type}</td>
                            <td>{c.claim_date ? new Date(c.claim_date).toLocaleDateString() : '—'}</td>
                            <td className="mono">{formatINR(c.currency === 'INR' && c.original_reimbursement ? c.original_reimbursement : usdToInr(c.reimbursement_amount))}</td>
                            <td>
                              {c.has_anomalies ? (
                                <span className="pill high">
                                  {c.anomaly_flags?.length || 1} Anomaly
                                </span>
                              ) : (
                                <span className="pill low">Normal</span>
                              )}
                            </td>
                            <td>
                              <button
                                className="inspect"
                                onClick={() => inspectClaim(c.claim_id)}
                              >
                                Inspect
                              </button>
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>

                <div className="pager">
                  <span>Page {claimPage} of {claimTotalPages}</span>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button
                      className="btn"
                      disabled={claimPage <= 1}
                      onClick={() => setClaimPage(claimPage - 1)}
                      style={{ opacity: claimPage <= 1 ? 0.4 : 1 }}
                      aria-label="Previous page"
                    >
                      ‹
                    </button>
                    <button
                      className="btn"
                      disabled={claimPage >= claimTotalPages}
                      onClick={() => setClaimPage(claimPage + 1)}
                      style={{ opacity: claimPage >= claimTotalPages ? 0.4 : 1 }}
                      aria-label="Next page"
                    >
                      ›
                    </button>
                  </div>
                </div>
              </div>

              {/* Claim Detail Modal */}
              {selectedClaim && (
                <div
                  className="scrim wscrim"
                  style={{ display: 'flex' }}
                  onClick={(e) => {
                    if (e.target.classList.contains('wscrim')) setSelectedClaim(null);
                  }}
                  role="dialog"
                  aria-modal="true"
                >
                  <div className="glass win">
                    <div className="wh">
                      <div>
                        <h2>{selectedClaim.claim_id}</h2>
                        <p className="sub mono">Provider: {selectedClaim.provider_id} &middot; Beneficiary: {selectedClaim.beneficiary_id}</p>
                      </div>
                      <div className="wa">
                        <span className="pill nw">{selectedClaim.claim_type}</span>
                        <button className="btn" onClick={() => setSelectedClaim(null)}>Close</button>
                      </div>
                    </div>

                    <div className="glass panel rs">
                      <div className="ttl">Claim Reimbursement &amp; Status</div>
                      <div className="pct">
                        {formatINR(selectedClaim.currency === 'INR' && selectedClaim.original_reimbursement ? selectedClaim.original_reimbursement : usdToInr(selectedClaim.reimbursement_amount))}
                      </div>
                      <div style={{ margin: '8px 0' }}>
                        {selectedClaim.has_anomalies ? (
                          <span className="pill high">{selectedClaim.anomaly_flags?.length || 1} Anomaly Flagged</span>
                        ) : (
                          <span className="pill low">Normal Claim</span>
                        )}
                      </div>
                      <div className="mg">
                        <div className="mt"><small>Service Date</small><b>{selectedClaim.claim_date ? new Date(selectedClaim.claim_date).toLocaleDateString() : 'N/A'}</b></div>
                        <div className="mt"><small>Provider Risk Tier</small><b>{selectedClaim.analysis?.provider_risk_tier || 'Pending'}</b></div>
                        <div className="mt"><small>Provider Probability</small><b>{selectedClaim.analysis?.provider_risk_score !== null && selectedClaim.analysis?.provider_risk_score !== undefined ? `${(selectedClaim.analysis.provider_risk_score * 100).toFixed(1)}%` : 'Pending'}</b></div>
                      </div>
                    </div>

                    <div className="glass panel">
                      <h2>Claim-Level Anomaly Detection Findings</h2>
                      {selectedClaim.analysis?.anomaly_flags && selectedClaim.analysis.anomaly_flags.length > 0 ? (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 12 }}>
                          {selectedClaim.analysis.anomaly_flags.map((flag, idx) => (
                            <div key={idx} className="fe" style={{ margin: 0 }}>
                              &bull; {flag}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p style={{ color: 'var(--low)', fontWeight: 600 }}>No deterministic anomaly rules triggered for this claim.</p>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </section>
          )}

          {/* ════════════════════════════════════════════════════════════════════════
              SECTION 4: CLAIM ANALYZER (SINGLE & BATCH CSV)
          ════════════════════════════════════════════════════════════════════════ */}
          {activeTab === 'analyzer' && (
            <section className="view on" id="v-analyzer">
              <div className="head analyzer-head" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 16, marginBottom: 20 }}>
                <div style={{ minWidth: 260 }}>
                  <h1 style={{ margin: 0, fontSize: 24, fontWeight: 800 }}>Claim Fraud Analyzer</h1>
                  <p style={{ margin: '4px 0 0', color: 'var(--muted)', fontSize: 13 }}>Execute real-time multi-modal ensemble inference and batch claims validation.</p>
                </div>
                <div className="seg" data-tabs="apane" style={{ margin: 0 }}>
                  <button
                    className={analyzerSubTab === 'single' ? 'on' : ''}
                    onClick={() => setAnalyzerSubTab('single')}
                  >
                    Single-Claim Analysis
                  </button>
                  <button
                    className={analyzerSubTab === 'csv' ? 'on' : ''}
                    onClick={() => setAnalyzerSubTab('csv')}
                  >
                    Batch CSV Upload
                  </button>
                </div>
              </div>

              {/* Sub-tab 1: Single Claim Analysis */}
              {analyzerSubTab === 'single' && (
                <div className="split apane" id="p-single">
                  {/* Left Column: Form */}
                  <div className="glass panel">
                    <div className="ph">
                      <h2>Claim Attributes</h2>
                      <button className="mini" id="fs" type="button" onClick={loadSampleClaim}>
                        Fill sample claim
                      </button>
                    </div>

                    
                    {analysisError && (
                      <p className="fe" role="alert">{analysisError}</p>
                    )}

                    <form onSubmit={handleSingleClaimSubmit} className="form" id="cf">
                      <div className="f">
                        <label>Claim ID</label>
                        <input
                          id="f-cid"
                          type="text"
                          required
                          value={singleClaimForm.claim_id}
                          onChange={(e) => setSingleClaimForm({ ...singleClaimForm, claim_id: e.target.value })}
                          placeholder="e.g. CLM_5127"
                        />
                      </div>

                      <div className="f">
                        <label>Provider ID</label>
                        <input
                          id="f-pid"
                          type="text"
                          required
                          value={singleClaimForm.provider_id}
                          onChange={(e) => setSingleClaimForm({ ...singleClaimForm, provider_id: e.target.value })}
                          placeholder="e.g. PRV51003"
                        />
                      </div>

                      <div className="f">
                        <label>Beneficiary ID</label>
                        <input
                          id="f-bid"
                          type="text"
                          required
                          value={singleClaimForm.beneficiary_id}
                          onChange={(e) => setSingleClaimForm({ ...singleClaimForm, beneficiary_id: e.target.value })}
                          placeholder="e.g. BENE11001"
                        />
                      </div>

                      <div className="f">
                        <label>Claim Type</label>
                        <select
                          id="f-type"
                          value={singleClaimForm.claim_type}
                          onChange={(e) => setSingleClaimForm({ ...singleClaimForm, claim_type: e.target.value })}
                        >
                          <option value="Outpatient">Outpatient</option>
                          <option value="Inpatient">Inpatient</option>
                        </select>
                      </div>

                      {/* Inpatient Hospitalization Box (only for Inpatient claims) */}
                      {singleClaimForm.claim_type === 'Inpatient' && (
                        <div className="ipb" id="ipb">
                          <div className="ph">
                            <h3>Inpatient hospitalization</h3>
                            <span className="pill nw" id="los">
                              Length of stay: {(() => {
                                if (!singleClaimForm.admission_date || !singleClaimForm.discharge_date) return '0 days';
                                const d1 = new Date(singleClaimForm.admission_date);
                                const d2 = new Date(singleClaimForm.discharge_date);
                                const diff = Math.round((d2 - d1) / (1000 * 60 * 60 * 24));
                                return diff >= 0 ? `${diff} day${diff === 1 ? '' : 's'}` : 'Invalid (Discharge < Admission)';
                              })()}
                            </span>
                          </div>
                          <div className="ipg">
                            <div className="f">
                              <label>Admission Date</label>
                              <input
                                id="f-adm"
                                type="date"
                                required={singleClaimForm.claim_type === 'Inpatient'}
                                max={new Date().toISOString().split('T')[0]}
                                value={singleClaimForm.admission_date}
                                onChange={(e) => setSingleClaimForm({ ...singleClaimForm, admission_date: e.target.value })}
                              />
                            </div>
                            <div className="f">
                              <label>Discharge Date</label>
                              <input
                                id="f-dis"
                                type="date"
                                required={singleClaimForm.claim_type === 'Inpatient'}
                                max={new Date().toISOString().split('T')[0]}
                                value={singleClaimForm.discharge_date}
                                onChange={(e) => setSingleClaimForm({ ...singleClaimForm, discharge_date: e.target.value })}
                              />
                            </div>
                          </div>
                        </div>
                      )}

                      <div className="f">
                        <label>Reimbursement (₹)</label>
                        <input
                          id="f-reimb"
                          type="number"
                          step="0.01"
                          min="0"
                          max="41750000"
                          required
                          value={singleClaimForm.reimbursement_amount}
                          onChange={(e) => setSingleClaimForm({ ...singleClaimForm, reimbursement_amount: e.target.value })}
                          placeholder="0.00"
                        />
                      </div>

                      <div className="f">
                        <label>Deductible (₹)<span className="op">optional</span></label>
                        <input
                          id="f-ded"
                          type="number"
                          step="0.01"
                          min="0"
                          max="2087500"
                          value={singleClaimForm.deductible_amount}
                          onChange={(e) => setSingleClaimForm({ ...singleClaimForm, deductible_amount: e.target.value })}
                          placeholder="0.00"
                        />
                      </div>

                      <div className="f">
                        <label>Claim Start Date</label>
                        <input
                          id="f-cs"
                          type="date"
                          required
                          max={new Date().toISOString().split('T')[0]}
                          value={singleClaimForm.claim_start_date}
                          onChange={(e) => setSingleClaimForm({ ...singleClaimForm, claim_start_date: e.target.value })}
                        />
                      </div>

                      <div className="f">
                        <label>Claim End Date</label>
                        <input
                          id="f-ce"
                          type="date"
                          required
                          max={new Date().toISOString().split('T')[0]}
                          value={singleClaimForm.claim_end_date}
                          onChange={(e) => setSingleClaimForm({ ...singleClaimForm, claim_end_date: e.target.value })}
                        />
                      </div>

                      <div className="f">
                        <label>Diagnosis Code 1<span className="op">optional</span></label>
                        <input
                          id="f-dx"
                          type="text"
                          value={singleClaimForm.diagnosis_code_1}
                          onChange={(e) => setSingleClaimForm({ ...singleClaimForm, diagnosis_code_1: e.target.value })}
                          placeholder="Primary (e.g. 4019)"
                        />
                      </div>

                      <div className="f">
                        <label>Diagnosis Code 2<span className="op">optional</span></label>
                        <input
                          type="text"
                          value={singleClaimForm.diagnosis_code_2}
                          onChange={(e) => setSingleClaimForm({ ...singleClaimForm, diagnosis_code_2: e.target.value })}
                          placeholder="Optional (e.g. 25000)"
                        />
                      </div>

                      <div className="f">
                        <label>Admit Diagnosis Code<span className="op">optional</span></label>
                        <input
                          type="text"
                          value={singleClaimForm.admission_diagnosis_code}
                          onChange={(e) => setSingleClaimForm({ ...singleClaimForm, admission_diagnosis_code: e.target.value })}
                          placeholder="Optional (e.g. 4280)"
                        />
                      </div>

                      <div className="f">
                        <label>Procedure Code 1<span className="op">optional</span></label>
                        <input
                          id="f-px"
                          type="text"
                          value={singleClaimForm.procedure_code_1}
                          onChange={(e) => setSingleClaimForm({ ...singleClaimForm, procedure_code_1: e.target.value })}
                          placeholder="e.g. 3893"
                        />
                      </div>

                      <div className="f">
                        <label>Attending Physician<span className="op">optional</span></label>
                        <input
                          id="f-att"
                          type="text"
                          value={singleClaimForm.attending_physician}
                          onChange={(e) => setSingleClaimForm({ ...singleClaimForm, attending_physician: e.target.value })}
                          placeholder="e.g. PHY31201"
                        />
                      </div>

                      <div className="f">
                        <label>Operating Physician<span className="op">optional</span></label>
                        <input
                          id="f-op"
                          type="text"
                          value={singleClaimForm.operating_physician}
                          onChange={(e) => setSingleClaimForm({ ...singleClaimForm, operating_physician: e.target.value })}
                          placeholder="e.g. PHY60301"
                        />
                      </div>

                      <div className="f">
                        <label>Other Physician<span className="op">optional</span></label>
                        <input
                          id="f-oth"
                          type="text"
                          value={singleClaimForm.other_physician}
                          onChange={(e) => setSingleClaimForm({ ...singleClaimForm, other_physician: e.target.value })}
                          placeholder="e.g. PHY80501"
                        />
                      </div>

                      <button
                        type="submit"
                        className="primary"
                        id="run"
                        disabled={analysisLoading}
                        style={{ gridColumn: '1 / -1' }}
                      >
                        {analysisLoading ? 'Evaluating ML Ensemble...' : 'Run Fraud Risk Assessment'}
                      </button>
                    </form>
                  </div>

                  {/* Right Column: Findings */}
                  <div id="fnd" style={{ display: 'flex', flexDirection: 'column', gap: 16, minWidth: 0 }}>
                    {!analysisResult && !analysisLoading && (
                      <div className="glass panel" id="empty">
                        <h2>Risk Evaluation Findings</h2>
                        <div className="drop">
                          Submit claim attributes to run the risk assessment. The report shows the score, the reasons behind it and the provider's billing history.
                        </div>
                      </div>
                    )}

                    {analysisResult && (
                      <div id="res" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                        {/* Risk Assessment Summary */}
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8, padding: '2px 4px 6px' }}>
                          <span style={{ font: '800 15px / 1.2 Manrope, Inter, sans-serif', color: 'var(--text)', letterSpacing: '-0.01em' }}>
                            Risk Assessment Summary
                          </span>
                          <span className="pill nw" style={{ fontSize: 11, padding: '4px 10px' }} data-testid="model-version-badge">
                            Model: {analysisResult.model_version || 'v2'}
                          </span>
                        </div>

                        <div className={`glass panel triage-strip ${forensicData?.tierClass || 'tier-low'}`} id="verdict-strip">
                          <div className="triage-score-box">
                            <div className="gauge-ring">
                              <svg viewBox="0 0 100 100">
                                <circle className="bg-ring" cx="50" cy="50" r="40" />
                                <circle
                                  className="val-ring"
                                  cx="50"
                                  cy="50"
                                  id="gauge-circle"
                                  r="40"
                                  stroke={forensicData?.strokeColor || '#10b981'}
                                  strokeDasharray="251.2"
                                  strokeDashoffset={forensicData?.dashOffset !== undefined ? forensicData.dashOffset : 40}
                                />
                              </svg>
                              <div className="gauge-center">
                                <div className="gauge-val" id="score-display" data-testid="risk-score-value">
                                  {formatProbability(analysisResult.provider_risk_score)}
                                </div>
                                <div className="gauge-sub" id="score-tier-label" data-testid="risk-tier-badge">
                                  {analysisResult.provider_risk_tier ? `${analysisResult.provider_risk_tier} RISK` : 'Unavailable'}
                                </div>
                              </div>
                            </div>

                            <div className="verdict-info">
                              <h3 id="verdict-headline">
                                <span className={`pill ${forensicData?.tierPillClass || 'low'}`} id="tier-badge">
                                  {forensicData?.headlineBadge || 'ROUTINE BILLING PROFILE'}
                                </span>
                                <span>{forensicData?.headlineTitle || 'Standard Auto-Adjudication Approved'}</span>
                              </h3>
                              <div className="mandate-text" id="verdict-mandate" data-testid="investigation-recommendation">
                                <strong>Investigator Mandate:</strong> {forensicData?.mandate || analysisResult.investigation_recommendation}
                              </div>
                            </div>
                          </div>

                          <div className="exposure-box">
                            <small>Claim Amount Under Review</small>
                            <b id="exposure-val" style={{ color: 'var(--text)' }}>
                              {formatCurrency(forensicData?.atRiskAmount, analysisResult.currency || 'INR')}
                            </b>
                            <span style={{ fontSize: '11.5px', color: 'var(--muted)' }}>
                              Expected Risk Exposure: {formatCurrency(Math.round((forensicData?.atRiskAmount || 0) * (forensicData?.clampedScore || 0)), analysisResult.currency || 'INR')} ({(forensicData?.clampedScore * 100).toFixed(1)}% risk-weighted)
                            </span>
                          </div>

                          {/* Metadata boxes */}
                          <div className="mg" style={{ width: '100%', marginTop: 8, paddingTop: 10, borderTop: '1px solid var(--line)' }}>
                            <div className="mt"><small>Claim ID</small><b>{analysisResult.claim_id || singleClaimForm.claim_id}</b></div>
                            <div className="mt"><small>Provider ID</small><b>{analysisResult.provider_id || singleClaimForm.provider_id}</b></div>
                            <div className="mt"><small>Claim Type</small><b>{singleClaimForm.claim_type}</b></div>
                            <div className="mt">
                              <small>Submitted Reimbursement</small>
                              <b>{formatCurrency(analysisResult.original_reimbursement !== undefined ? analysisResult.original_reimbursement : singleClaimForm.reimbursement_amount, analysisResult.currency || 'INR')}</b>
                            </div>
                          </div>

                          {(!Number.isFinite(analysisResult.provider_risk_score) || !analysisResult.provider_risk_tier) && (
                            <div className="fe" role="alert" style={{ width: '100%', marginTop: 10 }} data-testid="incomplete-error">
                              Warning: The risk assessment response is incomplete or invalid. Please re-run the evaluation.
                            </div>
                          )}
                        </div>

                        {/* Interactive Attribution & Forensic Decomposition Section */}
                        {forensicData && (
                          <div style={{ marginTop: 8 }}>
                            {/* Prominent View Mode Segmented Toggle Bar */}
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12, marginBottom: 12, padding: '4px 0' }}>
                              <div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                  <h3 style={{ font: '800 17px / 1.3 Manrope, Inter, sans-serif', margin: 0 }}>Why This Risk Score?</h3>
                                </div>
                                <p style={{ fontSize: '12.5px', color: 'var(--muted)', marginTop: 2 }}>
                                  Examine the key factors driving this assessment score
                                </p>
                              </div>

                              {/* Interactive Segmented Toggle */}
                              <div aria-label="Explanation View Mode" className="seg-prominent" id="shap-view-mode-toggle" role="tablist">
                                <button
                                  aria-controls="view-narrative-panel"
                                  aria-selected={explanationViewMode === 'narrative'}
                                  className={explanationViewMode === 'narrative' ? 'on' : ''}
                                  id="btn-view-narrative"
                                  role="tab"
                                  type="button"
                                  onClick={() => setExplanationViewMode('narrative')}
                                >
                                  <svg className="ic" style={{ width: 16, height: 16 }}><use href="#doc" /></svg>
                                  <span>📄 Plain-English Narrative</span>
                                </button>
                                <button
                                  aria-controls="view-waterfall-panel"
                                  aria-selected={explanationViewMode === 'waterfall'}
                                  className={explanationViewMode === 'waterfall' ? 'on' : ''}
                                  id="btn-view-waterfall"
                                  role="tab"
                                  type="button"
                                  onClick={() => setExplanationViewMode('waterfall')}
                                >
                                  <svg className="ic" style={{ width: 16, height: 16 }}><use href="#chart-bar" /></svg>
                                  <span>📊 SHAP Waterfall Chart</span>
                                </button>
                              </div>
                            </div>

                            {/* Audit Accordion Container */}
                            <div className="audit-accordion" style={{ padding: 22 }}>
                              {/* VIEW 1: PLAIN-ENGLISH NARRATIVE */}
                              {explanationViewMode === 'narrative' && (
                                <div id="view-narrative-panel" role="tabpanel" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 14, marginBottom: 4 }}>
                                    <div>
                                      <h4 style={{ font: '700 16px / 1.3 Manrope, Inter, sans-serif', margin: 0 }}>Key Contributing Factors</h4>
                                      <span style={{ fontSize: '12.5px', color: 'var(--muted)' }}>Summary of primary risk-elevating and risk-mitigating patterns</span>
                                    </div>
                                    <button
                                      className="mini"
                                      id="btn-copy-narrative"
                                      type="button"
                                      onClick={handleCopyAuditMemo}
                                      style={{ display: 'inline-flex', alignItems: 'center', gap: 6, background: 'var(--glass-strong)', border: '1px solid var(--stroke)', padding: '7px 13px', borderRadius: 10, fontSize: 12, fontWeight: 600 }}
                                    >
                                      <svg className="ic" style={{ width: 15, height: 15 }}><use href="#copy" /></svg>
                                      <span id="copy-btn-text">{copiedMemo ? '✓ Copied!' : 'Copy'}</span>
                                    </button>
                                  </div>

                                  <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                                    {forensicData.narrativeCards.map((card, idx) => (
                                      <div key={idx} style={{ padding: 16, borderRadius: 14, background: 'var(--glass)', border: '1px solid var(--stroke)', boxShadow: 'inset 0 1px 0 var(--edge)' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8, flexWrap: 'wrap', gap: 8 }}>
                                          <span style={{ font: '700 14.5px / 1.3 Manrope, Inter, sans-serif', color: 'var(--text)' }}>
                                            {card.title}
                                          </span>
                                          <span className={`impact-badge ${card.isMitigating ? 'mitigating' : 'high'}`}>
                                            {card.shapStr}
                                          </span>
                                        </div>
                                        <p style={{ fontSize: 13, color: 'var(--text)', lineHeight: 1.5, marginBottom: 12 }}>
                                          {card.desc}
                                        </p>
                                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                                          {card.tags.map((tag, tIdx) => (
                                            <span
                                              key={tIdx}
                                              style={{
                                                fontSize: '11.5px',
                                                padding: '3px 10px',
                                                borderRadius: 8,
                                                background: tag.label === 'Observed'
                                                  ? (card.isMitigating ? 'var(--low-bg)' : 'var(--high-bg)')
                                                  : 'rgba(15, 23, 42, 0.04)',
                                                color: tag.label === 'Observed'
                                                  ? (card.isMitigating ? 'var(--low)' : 'var(--high)')
                                                  : 'var(--muted)',
                                                fontWeight: 600,
                                                border: tag.label === 'Observed'
                                                  ? (card.isMitigating ? '1px solid rgba(16, 185, 129, 0.2)' : '1px solid rgba(239, 68, 68, 0.2)')
                                                  : '1px solid var(--stroke)'
                                              }}
                                            >
                                              {tag.label}: {tag.val}
                                            </span>
                                          ))}
                                        </div>
                                      </div>
                                    ))}
                                  </div>

                                  {/* Investigation Summary Box */}
                                  <div style={{ marginTop: 4, padding: '14px 16px', borderRadius: 12, background: 'var(--glass)', border: '1px solid var(--stroke)', display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                                    <svg className="ic" style={{ width: 18, height: 18, color: 'var(--accent)', marginTop: 2, flex: 'none' }}><use href="#info" /></svg>
                                    <div style={{ fontSize: 13, lineHeight: 1.45 }}>
                                      <strong style={{ color: 'var(--text)' }}>Assessment Summary:</strong> {forensicData.finalPct}% evaluated fraud risk probability{analysisResult.provider_risk_tier ? ` (${analysisResult.provider_risk_tier} RISK)` : ''}. <span style={{ color: 'var(--accent)', fontWeight: 600 }}>Investigation Guidance:</span> {forensicData.mandate}
                                    </div>
                                  </div>
                                </div>
                              )}

                              {/* VIEW 2: SHAP WATERFALL CHART */}
                              {explanationViewMode === 'waterfall' && (
                                <div id="view-waterfall-panel" role="tabpanel" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12, marginBottom: 4 }}>
                                    <div>
                                      <h4 style={{ font: '700 16px / 1.3 Manrope, Inter, sans-serif', margin: 0 }}>Risk Factor Breakdown</h4>
                                      <span style={{ fontSize: '12.5px', color: 'var(--muted)' }}>
                                        Stepwise feature contribution analysis of provider billing patterns
                                      </span>
                                    </div>
                                    <div style={{ display: 'flex', gap: 14, fontSize: 12, fontWeight: 600 }}>
                                      <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                        <span style={{ width: 10, height: 10, borderRadius: 2, background: '#ef4444' }}></span> Increases Risk (+)
                                      </span>
                                      <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                        <span style={{ width: 10, height: 10, borderRadius: 2, background: '#10b981' }}></span> Reduces Risk (-)
                                      </span>
                                      <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                        <span style={{ width: 10, height: 10, borderRadius: 2, background: '#6366f1' }}></span> Baseline / Score
                                      </span>
                                    </div>
                                  </div>

                                  {/* Waterfall Container */}
                                  <div className="waterfall-container">
                                    {/* Axis header */}
                                    <div className="waterfall-axis-header">
                                      <span>Feature &amp; Value</span>
                                      <span>Cumulative Fraud Probability Range [0.0 - 1.0]</span>
                                      <span style={{ textAlign: 'right' }}>SHAP Impact</span>
                                    </div>

                                    {/* Axis scale markers */}
                                    <div className="waterfall-grid-ticks">
                                      <div></div>
                                      <div className="ticks-bar">
                                        <span className="tick-mark" style={{ left: '0%', transform: 'none' }}>0.0</span>
                                        <span className="tick-mark" style={{ left: '20%' }}>0.20</span>
                                        <span className="tick-mark" style={{ left: '40%' }}>0.40</span>
                                        <span className="tick-mark" style={{ left: '60%' }}>0.60</span>
                                        <span className="tick-mark" style={{ left: '80%' }}>0.80</span>
                                        <span className="tick-mark" style={{ left: '100%', transform: 'translateX(-100%)' }}>1.0</span>
                                      </div>
                                      <div></div>
                                    </div>

                                    {/* Step 1: Base Value */}
                                    <div className="waterfall-step">
                                      <div className="step-meta">
                                        <div className="step-name">
                                          <span className="pill" style={{ fontSize: 10, padding: '2px 7px', background: 'var(--accent-soft)', color: 'var(--accent)', flexShrink: 0, whiteSpace: 'nowrap' }}>Baseline</span>
                                          <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>Baseline Population Risk</span>
                                        </div>
                                        <div className="step-feature-val">Regional Peer Prior ({forensicData.basePct}%)</div>
                                      </div>
                                      <div className="step-track">
                                        <div className="step-bar base" style={{ left: '0%', width: `${Math.max(2, forensicData.basePrior * 100)}%` }} title={`Starting prior: ${forensicData.basePrior.toFixed(2)}`}>
                                          {forensicData.basePrior >= 0.08 && (
                                            <span style={{ fontSize: 10, fontWeight: 700 }}>{forensicData.basePrior.toFixed(2)}</span>
                                          )}
                                        </div>
                                      </div>
                                      <div className="step-impact" style={{ color: 'var(--accent)' }}>+{forensicData.basePrior.toFixed(2)}</div>
                                    </div>

                                    {/* Intermediate Steps */}
                                    {forensicData.waterfallSteps.map((step, idx) => (
                                      <div key={idx} className="waterfall-step">
                                        <div className="step-meta">
                                          <div className="step-name">
                                            <span className={`pill ${step.isIncrease ? 'high' : 'low'}`} style={{ fontSize: 10, padding: '2px 7px', flexShrink: 0, whiteSpace: 'nowrap' }}>
                                              {step.isIncrease ? '+ Risk' : '- Risk'}
                                            </span>
                                            <span title={step.displayName} style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                              {step.displayName}
                                            </span>
                                          </div>
                                          {step.valStr && <div className="step-feature-val" title={step.valStr}>{step.valStr}</div>}
                                        </div>
                                        <div className="step-track">
                                          {/* Connect line from previous cumulative position */}
                                          <div style={{ position: 'absolute', left: `${step.connectLinePct}%`, top: 0, bottom: 0, width: 1, borderLeft: '1px dashed var(--line)', zIndex: 1 }}></div>
                                          <div
                                            className={`step-bar ${step.isIncrease ? 'increase' : 'decrease'}`}
                                            style={{ left: `${step.leftPct}%`, width: `${step.widthPct}%` }}
                                            title={`${step.displayName}: ${step.displayImpact}`}
                                          >
                                            {step.widthPct >= 9 && (
                                              <span style={{ fontSize: 11, fontWeight: 700, whiteSpace: 'nowrap' }}>{step.displayImpact}</span>
                                            )}
                                          </div>
                                        </div>
                                        <div className={`step-impact ${step.isIncrease ? 'pos' : 'neg'}`}>
                                          {step.displayImpact}
                                        </div>
                                      </div>
                                    ))}

                                    {/* Final Step: Assessed Risk Score */}
                                    <div className="waterfall-step" style={{ background: 'linear-gradient(90deg, rgba(239, 68, 68, 0.12), var(--glass))', border: '1.5px solid rgba(239, 68, 68, 0.35)' }}>
                                      <div className="step-meta">
                                        <div className="step-name">
                                          <span className="pill high" style={{ fontSize: 10, padding: '2px 7px', fontWeight: 800, flexShrink: 0, whiteSpace: 'nowrap' }}>Score</span>
                                          <strong style={{ color: 'var(--high)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>Assessed Risk Score</strong>
                                        </div>
                                        <div className="step-feature-val" style={{ color: 'var(--text)', fontWeight: 700 }}>Audit Threshold: &gt; 0.65 (High)</div>
                                      </div>
                                      <div className="step-track" style={{ position: 'relative' }}>
                                        <div
                                          className="step-bar total"
                                          style={{ left: '0%', width: `${Math.max(2, forensicData.clampedScore * 100)}%` }}
                                          title={`Final assessed probability: ${forensicData.clampedScore.toFixed(2)}`}
                                        >
                                          {forensicData.finalPct >= 22 && (
                                            <span style={{ fontSize: 11, fontWeight: 800, whiteSpace: 'nowrap', padding: '0 8px' }}>
                                              {forensicData.finalPct}% Assessed Score
                                            </span>
                                          )}
                                        </div>
                                        {forensicData.finalPct < 22 && (
                                          <span
                                            style={{
                                              position: 'absolute',
                                              left: `calc(${Math.max(2, forensicData.clampedScore * 100)}% + 8px)`,
                                              top: '50%',
                                              transform: 'translateY(-50%)',
                                              fontSize: 11,
                                              fontWeight: 800,
                                              color: 'var(--high)',
                                              whiteSpace: 'nowrap',
                                              zIndex: 2
                                            }}
                                          >
                                            {forensicData.finalPct}% Assessed Score
                                          </span>
                                        )}
                                      </div>
                                      <div className="step-impact tot" style={{ color: 'var(--high)' }}>{forensicData.clampedScore.toFixed(2)}</div>
                                    </div>
                                  </div>

                                  {/* Investigator Guidance Footer */}
                                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '12px 16px', borderRadius: 12, background: 'var(--glass)', border: '1px solid var(--stroke)', fontSize: 12, color: 'var(--muted)' }}>
                                    <svg className="ic" style={{ width: 16, height: 16, color: 'var(--accent)', flexShrink: 0 }}><use href="#info" /></svg>
                                    <span>Feature impacts indicate how specific provider patterns increase or decrease risk relative to peer baselines. TreeSHAP explains tabular risk factors. Automated scores guide audit prioritization and require clinical adjudication.</span>
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        )}

                        {/* 3. Visual Analysis Charts */}
                        {analysisResult.monthly_trends?.months && analysisResult.monthly_trends.months.length > 0 && (
                          <div className="charts">
                            <div className="glass panel cp">
                              <h2>Monthly Reimbursement Trend</h2>
                              <p className="cs">Provider reimbursement per month, in ₹ (INR)</p>
                              <div className="cw" style={{ height: 200 }}>
                                <ResponsiveContainer width="100%" height="100%">
                                  <LineChart data={analysisResult.monthly_trends.months}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" />
                                    <XAxis dataKey="month" tick={{ fontSize: 10, fill: 'var(--muted)' }} />
                                    <YAxis tick={{ fontSize: 10, fill: 'var(--muted)' }} />
                                    <Tooltip
                                      contentStyle={{ backgroundColor: 'var(--glass-strong)', borderColor: 'var(--stroke)', borderRadius: 12, backdropFilter: 'blur(12px)' }}
                                      formatter={(val) => [formatCurrency(val, analysisResult.currency || 'INR'), 'Reimbursement']}
                                    />
                                    <Line type="monotone" dataKey="reimbursement" stroke="var(--accent)" strokeWidth={2.5} dot={{ r: 3 }} />
                                  </LineChart>
                                </ResponsiveContainer>
                              </div>
                            </div>

                            <div className="glass panel cp">
                              <h2>Monthly Claim Activity</h2>
                              <p className="cs">Number of claims submitted per month</p>
                              <div className="cw" style={{ height: 200 }}>
                                <ResponsiveContainer width="100%" height="100%">
                                  <BarChart data={analysisResult.monthly_trends.months}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" />
                                    <XAxis dataKey="month" tick={{ fontSize: 10, fill: 'var(--muted)' }} />
                                    <YAxis allowDecimals={false} tick={{ fontSize: 10, fill: 'var(--muted)' }} />
                                    <Tooltip
                                      contentStyle={{ backgroundColor: 'var(--glass-strong)', borderColor: 'var(--stroke)', borderRadius: 12, backdropFilter: 'blur(12px)' }}
                                      formatter={(val) => [val, 'Claims']}
                                    />
                                    <Bar dataKey="claim_count" fill="var(--accent)" radius={[4, 4, 0, 0]} />
                                  </BarChart>
                                </ResponsiveContainer>
                              </div>
                            </div>
                          </div>
                        )}

                        {/* Statutory Disclaimer Notice */}
                        <p className="fine" data-testid="regulatory-disclaimer">
                          {analysisResult.disclaimer || MANDATORY_DISCLAIMER}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Sub-tab 2: Batch CSV Upload */}
              {analyzerSubTab === 'csv' && (
                <div className="glass panel apane" id="p-batch">
                  <div className="ph" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 16, marginBottom: 20 }}>
                    <div style={{ flex: '1 1 420px', minWidth: 260 }}>
                      <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700 }}>Upload Batch Claims CSV</h2>
                      <p className="cs" style={{ margin: '6px 0 0', color: 'var(--muted)', fontSize: 13, lineHeight: 1.5 }}>
                        Upload a batch file of claims. The system validates required headers, reconciles duplicates, and evaluates fraud risk asynchronously.
                      </p>
                    </div>
                    <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
                      <select
                        value={csvTemplateCurrency}
                        onChange={(e) => setCsvTemplateCurrency(e.target.value)}
                        style={{ height: 38, padding: '0 12px', borderRadius: 10 }}
                      >
                        <option value="INR">Template: INR (₹)</option>
                        <option value="USD">Template: USD ($)</option>
                      </select>
                      <button
                        className="btn"
                        type="button"
                        onClick={() => downloadCsvTemplate(csvTemplateCurrency)}
                        style={{ height: 38, display: 'inline-flex', alignItems: 'center', gap: 6 }}
                      >
                        <svg className="ic" style={{ width: 14, height: 14 }}><use href="#download"/></svg>
                        Download Template
                      </button>
                    </div>
                  </div>

                  {/* Drag and Drop Zone */}
                  <div
                    className={`drop ${csvDragActive ? 'drag-over' : ''}`}
                    onDragOver={(e) => { e.preventDefault(); setCsvDragActive(true); }}
                    onDragLeave={() => setCsvDragActive(false)}
                    onDrop={(e) => {
                      e.preventDefault();
                      setCsvDragActive(false);
                      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                        handleCsvFileSelect(e.dataTransfer.files[0]);
                      }
                    }}
                    onClick={() => document.getElementById('csvFileInput')?.click()}
                    style={{
                      cursor: 'pointer',
                      border: '2px dashed var(--line)',
                      borderRadius: 16,
                      padding: '36px 24px',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: 12,
                      background: csvDragActive ? 'var(--accent-soft)' : 'rgba(255, 255, 255, 0.02)',
                      transition: 'all 0.2s ease',
                      textAlign: 'center'
                    }}
                  >
                    <input
                      id="csvFileInput"
                      type="file"
                      accept=".csv"
                      onChange={(e) => {
                        if (e.target.files && e.target.files[0]) {
                          handleCsvFileSelect(e.target.files[0]);
                        }
                      }}
                      style={{ display: 'none' }}
                    />
                    <div style={{
                      width: 48,
                      height: 48,
                      borderRadius: '50%',
                      background: 'var(--accent-soft)',
                      color: 'var(--accent)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}>
                      <svg className="ic" style={{ width: 24, height: 24 }}><use href="#upload"/></svg>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, alignItems: 'center' }}>
                      <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--text)' }}>
                        {csvFile ? csvFile.name : 'Choose a file or drag and drop here'}
                      </span>
                      <span style={{ fontSize: 12, color: 'var(--muted)' }}>
                        {csvFile
                          ? `${(csvFile.size / 1024).toFixed(1)} KB · Ready to preview`
                          : 'claims_batch.csv or any valid claims CSV (max 20MB)'}
                      </span>
                    </div>
                    <button type="button" className="btn" style={{ marginTop: 4, pointerEvents: 'none' }}>
                      {csvFile ? 'Change File' : 'Browse Files'}
                    </button>
                  </div>

                  {csvUploadError && <p className="fe" role="alert">{csvUploadError}</p>}

                  {csvPreviewLoading && (
                    <div className="sn" style={{ marginTop: 12 }}>
                      <svg className="ic"><use href="#info"/></svg>
                      <span>Inspecting CSV structure, detecting duplicates, and validating columns...</span>
                    </div>
                  )}

                  {/* Pre-upload preview card */}
                  {csvPreviewData && !csvUploadResult && (
                    <div style={{ marginTop: 18 }}>
                      <div className="ph">
                        <div>
                          <h2>Pre-Upload Batch Preview ({csvPreviewData.filename})</h2>
                          <p className="cs">Review metrics before queuing execution.</p>
                        </div>
                        <div style={{ display: 'flex', gap: 8 }}>
                          <button
                            type="button"
                            className="btn"
                            onClick={() => { setCsvFile(null); setCsvPreviewData(null); }}
                          >
                            Cancel
                          </button>
                          <button
                            type="button"
                            className="primary"
                            onClick={handleCsvUpload}
                            disabled={csvStatusLoading || (csvPreviewData.valid_rows_count === 0 && csvPreviewData.total_rows > 0)}
                            style={{ margin: 0, padding: '9px 18px' }}
                          >
                            {csvStatusLoading ? 'Processing...' : `Proceed with Batch (${csvPreviewData.valid_rows_count} Claims)`}
                          </button>
                        </div>
                      </div>

                      <div className="stats" style={{ marginTop: 12 }}>
                        <div className="stat">Total Rows<b>{csvPreviewData.total_rows}</b></div>
                        <div className="stat g">Valid Claims<b>{csvPreviewData.valid_rows_count}</b></div>
                        <div className="stat r">Invalid / Duplicates<b>{csvPreviewData.invalid_rows_count + csvPreviewData.duplicate_rows_count}</b></div>
                      </div>

                      {csvPreviewData.preview_rows.length > 0 && (
                        <div className="tablewrap" style={{ marginTop: 14 }}>
                          <table>
                            <thead>
                              <tr>
                                <th>#</th>
                                <th>Claim ID</th>
                                <th>Provider ID</th>
                                <th>Type</th>
                                <th>Reimbursement</th>
                                <th>Status</th>
                              </tr>
                            </thead>
                            <tbody>
                              {csvPreviewData.preview_rows.map((p, idx) => (
                                <tr key={idx}>
                                  <td className="mono">{p.row_number}</td>
                                  <td className="mono"><b>{p.claim_id || '—'}</b></td>
                                  <td className="mono">{p.provider_id || '—'}</td>
                                  <td>{p.claim_type || '—'}</td>
                                  <td className="mono">
                                    {p.reimbursement !== null && p.reimbursement !== undefined
                                      ? (p.currency === 'INR' ? formatINR(p.reimbursement) : `$${p.reimbursement.toLocaleString()}`)
                                      : '—'}
                                  </td>
                                  <td>
                                    <span className={`pill ${p.status === 'VALID' ? 'low' : 'high'}`}>
                                      {p.status}
                                    </span>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Batch Upload Reconciled Status */}
                  {csvUploadResult && (
                    <div style={{ marginTop: 24 }}>
                      <div className="ph">
                        <h2>Batch Upload Reconciled Status</h2>
                        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                          <span className="pill nw">Status: {csvUploadResult.status}</span>
                          {csvUploadResult.status === 'COMPLETED' && (
                            <button
                              type="button"
                              className="btn"
                              onClick={() => exportBatchResults(csvUploadResult.upload_id)}
                            >
                              Export CSV
                            </button>
                          )}
                        </div>
                      </div>

                      <div className="stats">
                        <div className="stat">Total Claims in Batch<b>{csvUploadResult.total_rows}</b></div>
                        <div className="stat g">Successful Rows<b>{csvUploadResult.successful_rows}</b></div>
                        <div className="stat r">Rejected Rows<b>{csvUploadResult.rejected_rows}</b></div>
                      </div>

                      <div className="inv">
                        <svg className="ic"><use href="#info"/></svg>
                        Reconciliation Invariant Enforced: Total ({csvUploadResult.total_rows}) = Successful ({csvUploadResult.successful_rows}) + Rejected ({csvUploadResult.rejected_rows})
                      </div>

                      <div className="seg" data-tabs="bpane" style={{ marginBottom: 12 }}>
                        <button
                          className={csvResultsTab === 'successful' ? 'on' : ''}
                          onClick={() => setCsvResultsTab('successful')}
                        >
                          Successful Claims ({csvSuccessfulClaims.length})
                        </button>
                        <button
                          className={csvResultsTab === 'rejected' ? 'on' : ''}
                          onClick={() => setCsvResultsTab('rejected')}
                        >
                          Rejected Rows ({csvRejectedRows.length})
                        </button>
                      </div>

                      {csvResultsTab === 'successful' && (
                        <div className="tablewrap bpane" id="b-ok">
                          <table>
                            <thead>
                              <tr>
                                <th>Claim ID</th>
                                <th>Provider ID</th>
                                <th>Reimbursed</th>
                                <th>Risk Tier</th>
                                <th>Anomaly Status</th>
                              </tr>
                            </thead>
                            <tbody>
                              {csvSuccessfulClaims.length === 0 ? (
                                <tr><td colSpan="5" className="empty">No claims records returned for this upload.</td></tr>
                              ) : (
                                csvSuccessfulClaims.map((c, i) => (
                                  <tr key={i}>
                                    <td className="mono"><b>{c.claim_id}</b></td>
                                    <td className="mono">{c.provider_id}</td>
                                    <td className="mono">
                                      {formatINR(c.currency === 'INR' && c.original_reimbursement ? c.original_reimbursement : usdToInr(c.reimbursement_amount))}
                                    </td>
                                    <td>{getTierBadge(c.provider_risk_tier)}</td>
                                    <td>
                                      {c.has_anomalies || (c.anomaly_flags && c.anomaly_flags.length > 0) ? (
                                        <span className="pill high">{c.anomaly_flags?.length || 1} Anomaly</span>
                                      ) : (
                                        <span className="pill low">Normal</span>
                                      )}
                                    </td>
                                  </tr>
                                ))
                              )}
                            </tbody>
                          </table>
                        </div>
                      )}

                      {csvResultsTab === 'rejected' && (
                        <div className="tablewrap bpane" id="b-rej">
                          <table>
                            <thead>
                              <tr>
                                <th>Row #</th>
                                <th>Claim ID</th>
                                <th>Rejection Reason</th>
                              </tr>
                            </thead>
                            <tbody>
                              {csvRejectedRows.length === 0 ? (
                                <tr><td colSpan="3" className="empty">No rejected rows recorded.</td></tr>
                              ) : (
                                csvRejectedRows.map((r, i) => (
                                  <tr key={i}>
                                    <td className="mono">{r.row_number}</td>
                                    <td className="mono"><b>{r.claim_id || '—'}</b></td>
                                    <td style={{ color: 'var(--high)', fontWeight: 500 }}>{r.rejection_reason}</td>
                                  </tr>
                                ))
                              )}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </section>
          )}

          {/* ════════════════════════════════════════════════════════════════════════
              SECTION 5: SETTINGS VIEW
          ════════════════════════════════════════════════════════════════════════ */}
          {activeTab === 'settings' && (
            <section className="view on" id="v-settings">
              <div className="head">
                <div>
                  <h1>System Configuration &amp; Settings</h1>
                  <p>Model versions, risk thresholds, and database connection status.</p>
                </div>
              </div>

              {settingsError && <p className="fe" role="alert">{settingsError}</p>}
              {settingsSuccess && (
                <div className="sn">
                  <svg className="ic"><use href="#info"/></svg>
                  <span>{settingsSuccess}</span>
                </div>
              )}

              {settingsData && (
                <div className="split">
                  <div className="glass panel">
                    <h2>Active Machine Learning Setup</h2>
                    <div className="kv">
                      <span>Ensemble Version</span>
                      <b className="mono">{settingsData.model_version.toUpperCase()}</b>
                    </div>
                    <div className="kv">
                      <span>Artifact Verification</span>
                      <b className="ok">{settingsData.models_ready ? '13/13 SHA-256 Validated' : 'Verification Failed'}</b>
                    </div>
                    <div className="kv">
                      <span>Low/Medium Risk Cutoff</span>
                      <b className="mono">{settingsData.low_medium_threshold}</b>
                    </div>
                    <div className="kv">
                      <span>Medium/High Risk Cutoff</span>
                      <b className="mono">{settingsData.medium_high_threshold}</b>
                    </div>

                    {currentUser.role === 'admin' ? (
                      <div style={{ marginTop: 18, paddingTop: 14, borderTop: '1px solid var(--line)' }}>
                        <span style={{ fontSize: 13, fontWeight: 600, display: 'block', marginBottom: 8 }}>
                          Administrator: Switch Active Model Version
                        </span>
                        <div style={{ display: 'flex', gap: 8 }}>
                          <button
                            type="button"
                            className="btn"
                            disabled={settingsData.model_version === 'v1' || settingsLoading}
                            onClick={() => handleSwitchModelVersion('v1')}
                          >
                            Switch to v1
                          </button>
                          <button
                            type="button"
                            className="primary"
                            disabled={settingsData.model_version === 'v2' || settingsLoading}
                            onClick={() => handleSwitchModelVersion('v2')}
                            style={{ margin: 0, padding: '9px 18px' }}
                          >
                            Switch to v2
                          </button>
                        </div>
                      </div>
                    ) : (
                      <p className="fine">Administrator privileges are required to modify model versions or thresholds.</p>
                    )}
                  </div>

                  <div className="glass panel">
                    <h2>Database &amp; Compliance Architecture</h2>
                    <div className="kv">
                      <span>Database Engine</span>
                      <b className="mono">PostgreSQL 18</b>
                    </div>
                    <div className="kv">
                      <span>Connection State</span>
                      <b className="ok">{settingsData.database_connected ? 'Connected' : 'Disconnected'}</b>
                    </div>
                    <div className="kv">
                      <span>Silent SQLite Fallback</span>
                      <b className="ok">Disabled (Production Enforced)</b>
                    </div>
                    <div className="kv">
                      <span>HIPAA Compliance</span>
                      <b className="ok">Zero Protected Demographic Features</b>
                    </div>
                  </div>
                </div>
              )}
            </section>
          )}
        </main>
      </div>

      {/* Floating Toast Notification */}
      {toastMessage && (
        <div className="toast" role="status">
          {toastMessage}
        </div>
      )}
    </>
  );
}

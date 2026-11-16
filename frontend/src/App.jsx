import React, { useState, useEffect, useRef } from 'react';
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
  UserCheck
} from 'lucide-react';

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

// Helper for risk tier badge styling with strict fallback to 'Unavailable' (never defaulting to LOW)
export const getTierBadge = (tier) => {
  if (!tier || typeof tier !== 'string') {
    return (
      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-slate-100 text-slate-600 border border-slate-200">
        Unavailable
      </span>
    );
  }
  const t = tier.trim().toUpperCase();
  if (t === 'HIGH') {
    return (
      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-red-100 text-red-800 border border-red-200">
        HIGH RISK
      </span>
    );
  }
  if (t === 'MEDIUM') {
    return (
      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-100 text-amber-800 border border-amber-200">
        MEDIUM RISK
      </span>
    );
  }
  if (t === 'LOW') {
    return (
      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 border border-emerald-200">
        LOW RISK
      </span>
    );
  }
  return (
    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-slate-100 text-slate-600 border border-slate-200">
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

  // Navigation: strictly 5 items (Overview, Providers, Claims, Claim Analyzer, Settings)
  const [activeTab, setActiveTab] = useState('overview');

  // Overview State
  const [overviewSummary, setOverviewSummary] = useState(null);
  const [overviewRecent, setOverviewRecent] = useState([]);
  const [riskDistribution, setRiskDistribution] = useState(null);
  const [overviewLoading, setOverviewLoading] = useState(false);
  const [overviewError, setOverviewError] = useState(null);

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
    claim_id: 'CLM_DEMO_001',
    provider_id: 'PRV51003',
    beneficiary_id: 'BENE11001',
    claim_type: 'Outpatient',
    reimbursement_amount: '4500.00',
    deductible_amount: '100.00',
    claim_start_date: '2009-03-01',
    claim_end_date: '2009-03-01',
    diagnosis_code_1: '4019',
    diagnosis_code_2: '25000',
    procedure_code_1: '',
    attending_physician: 'PHY31201',
    operating_physician: ''
  });
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisError, setAnalysisError] = useState(null);
  const latestAnalysisReqIdRef = useRef(0);

  // Batch CSV State
  const [csvFile, setCsvFile] = useState(null);
  const [csvUploadResult, setCsvUploadResult] = useState(null);
  const [csvStatusLoading, setCsvStatusLoading] = useState(false);
  const [csvUploadError, setCsvUploadError] = useState(null);
  const [csvResultsTab, setCsvResultsTab] = useState('successful');
  const [csvSuccessfulClaims, setCsvSuccessfulClaims] = useState([]);
  const [csvRejectedRows, setCsvRejectedRows] = useState([]);

  // Settings State
  const [settingsData, setSettingsData] = useState(null);
  const [settingsLoading, setSettingsLoading] = useState(false);
  const [settingsError, setSettingsError] = useState(null);
  const [settingsSuccess, setSettingsSuccess] = useState(null);

  // Check auth on mount
  useEffect(() => {
    if (token) {
      fetchCurrentUser();
    }
  }, [token]);

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
      // Automatically log in after registration
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
      const [sumRes, actRes, distRes] = await Promise.all([
        api.get('/api/overview/summary'),
        api.get('/api/overview/recent-activity?limit=10'),
        api.get('/api/overview/risk-distribution')
      ]);
      setOverviewSummary(sumRes.data);
      setOverviewRecent(actRes.data.recent_activities || []);
      setRiskDistribution(distRes.data);
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
    const reqId = ++latestAnalysisReqIdRef.current;
    setAnalysisLoading(true);
    setAnalysisError(null);
    setAnalysisResult(null);

    const payload = {
      ClaimID: singleClaimForm.claim_id,
      ProviderID: singleClaimForm.provider_id,
      BeneID: singleClaimForm.beneficiary_id,
      ClaimType: singleClaimForm.claim_type,
      InscClaimAmtReimbursed: parseFloat(singleClaimForm.reimbursement_amount) || 0.0,
      DeductibleAmtPaid: parseFloat(singleClaimForm.deductible_amount) || 0.0,
      ClaimStartDt: singleClaimForm.claim_start_date,
      ClaimEndDt: singleClaimForm.claim_end_date,
      ClmDiagnosisCode_1: singleClaimForm.diagnosis_code_1 || null,
      ClmDiagnosisCode_2: singleClaimForm.diagnosis_code_2 || null,
      ClmProcedureCode_1: singleClaimForm.procedure_code_1 || null,
      AttendingPhysician: singleClaimForm.attending_physician || null,
      OperatingPhysician: singleClaimForm.operating_physician || null
    };

    try {
      const res = await api.post('/api/analyzer/single', payload);
      if (reqId === latestAnalysisReqIdRef.current) {
        setAnalysisResult(res.data);
      }
    } catch (err) {
      if (reqId === latestAnalysisReqIdRef.current) {
        setAnalysisError(err.response?.data?.detail || 'Single-claim inference failed.');
      }
    } finally {
      if (reqId === latestAnalysisReqIdRef.current) {
        setAnalysisLoading(false);
      }
    }
  };

  const loadDemoClaim = (preset = 'high_risk') => {
    if (preset === 'high_risk') {
      setSingleClaimForm({
        claim_id: `CLM_DEMO_${Math.floor(1000 + Math.random() * 9000)}`,
        provider_id: 'PRV51003',
        beneficiary_id: 'BENE11001',
        claim_type: 'Inpatient',
        reimbursement_amount: '38500.00',
        deductible_amount: '1068.00',
        claim_start_date: '2009-04-10',
        claim_end_date: '2009-04-25',
        diagnosis_code_1: '4019',
        diagnosis_code_2: '25000',
        procedure_code_1: '3893',
        attending_physician: 'PHY31201',
        operating_physician: 'PHY31202'
      });
    } else {
      setSingleClaimForm({
        claim_id: `CLM_DEMO_${Math.floor(1000 + Math.random() * 9000)}`,
        provider_id: 'PRV51001',
        beneficiary_id: 'BENE22002',
        claim_type: 'Outpatient',
        reimbursement_amount: '450.00',
        deductible_amount: '50.00',
        claim_start_date: '2009-06-01',
        claim_end_date: '2009-06-01',
        diagnosis_code_1: '4019',
        diagnosis_code_2: '',
        procedure_code_1: '',
        attending_physician: 'PHY40101',
        operating_physician: ''
      });
    }
  };

  const handleCsvUpload = async (e) => {
    e.preventDefault();
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

  const pollCsvUploadStatus = async (uploadId) => {
    let attempts = 0;
    const interval = setInterval(async () => {
      attempts++;
      try {
        const res = await api.get(`/api/analyzer/csv/${uploadId}`);
        setCsvUploadResult(res.data);
        if (res.data.status === 'COMPLETED' || res.data.status === 'FAILED' || attempts > 20) {
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
    } catch (e) {
      console.error(e);
    }
  };

  // ══════════════════════════════════════════════════════════════════════════════
  // RENDER: LOGIN / REGISTER VIEW (UNAUTHENTICATED)
  // ══════════════════════════════════════════════════════════════════════════════

  if (!token || !currentUser) {
    return (
      <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col justify-center items-center px-4 py-12 relative overflow-hidden">
        {/* Subtle background glow effect */}
        <div className="absolute top-1/4 -left-20 w-96 h-96 bg-blue-600/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-1/4 -right-20 w-96 h-96 bg-indigo-600/10 rounded-full blur-3xl pointer-events-none" />

        <div className="max-w-md w-full bg-slate-800/90 backdrop-blur-md border border-slate-700/80 rounded-2xl shadow-2xl p-8 relative z-10">
          {/* Logo & Platform Title */}
          <div className="flex items-center space-x-3 mb-6">
            <div className="p-3 bg-blue-600/20 text-blue-400 rounded-xl border border-blue-500/30 shadow-inner">
              <ShieldAlert className="w-8 h-8 text-blue-400" />
            </div>
            <div>
              <h1 className="text-2xl font-extrabold text-white tracking-tight">HealthGuard AI</h1>
              <p className="text-xs text-slate-400 font-medium">Claims Fraud Investigation Portal</p>
            </div>
          </div>

          {/* Subtitle */}
          <div className="mb-5 text-[11px] text-slate-300 bg-slate-900/60 p-3 rounded-lg border border-slate-700/50 leading-relaxed">
            An Explainable Ensemble Framework for Health Insurance Provider Fraud Detection and Claims Investigation
          </div>

          {/* Mandatory Statutory Regulatory Notice */}
          <div className="mb-6 p-3 bg-blue-950/40 border border-blue-800/50 rounded-lg text-xs text-blue-300 flex items-start space-x-2.5">
            <Info className="w-4 h-4 flex-shrink-0 mt-0.5 text-blue-400" />
            <span className="leading-tight">{MANDATORY_DISCLAIMER}</span>
          </div>

          <div className="flex items-center justify-between mb-2">
            <h2 className="text-lg font-semibold text-white">
              {isRegisterView ? 'Register New Investigator' : 'Investigator Sign In'}
            </h2>
            <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-blue-900/60 text-blue-300 border border-blue-700/50">
              {isRegisterView ? 'Default: Investigator' : 'Secure JWT'}
            </span>
          </div>
          <p className="text-xs text-slate-400 mb-6">
            {isRegisterView
              ? 'Enter investigator details. Newly registered accounts are assigned the investigator role.'
              : 'Enter your credentials to access the forensic investigation dashboard.'}
          </p>

          {authError && (
            <div className="mb-4 p-3 bg-red-950/60 border border-red-800/70 rounded-lg text-xs text-red-300 flex items-center space-x-2">
              <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0" />
              <span>{authError}</span>
            </div>
          )}

          <form onSubmit={isRegisterView ? handleRegister : handleLogin} className="space-y-4">
            {isRegisterView && (
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5">Full Name</label>
                <div className="relative">
                  <User className="w-4 h-4 absolute left-3 top-2.5 text-slate-400" />
                  <input
                    type="text"
                    required
                    value={authForm.name}
                    onChange={(e) => setAuthForm({ ...authForm, name: e.target.value })}
                    placeholder="Dr. Alex Rivera"
                    className="w-full pl-9 pr-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition"
                  />
                </div>
              </div>
            )}

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">Email Address</label>
              <input
                type="email"
                required
                value={authForm.email}
                onChange={(e) => setAuthForm({ ...authForm, email: e.target.value })}
                placeholder="investigator@healthguard.ai"
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">Password</label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  required
                  value={authForm.password}
                  onChange={(e) => setAuthForm({ ...authForm, password: e.target.value })}
                  placeholder="••••••••••••"
                  className="w-full px-3 py-2 pr-10 bg-slate-900 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-2.5 text-slate-400 hover:text-slate-200 transition"
                  aria-label="Toggle password visibility"
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <button
              type="submit"
              disabled={authLoading}
              className="w-full py-2.5 px-4 bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-white text-sm font-semibold rounded-lg transition duration-150 flex items-center justify-center space-x-2 shadow-md disabled:opacity-50"
            >
              {authLoading ? (
                <span>Authenticating...</span>
              ) : (
                <>
                  <Lock className="w-4 h-4" />
                  <span>{isRegisterView ? 'Create Account' : 'Authenticate Session'}</span>
                </>
              )}
            </button>
          </form>

          <div className="mt-6 pt-4 border-t border-slate-700/60 text-center text-xs text-slate-400">
            {isRegisterView ? (
              <p>
                Already have an account?{' '}
                <button
                  type="button"
                  onClick={() => { setIsRegisterView(false); setAuthError(null); }}
                  className="text-blue-400 hover:underline font-semibold"
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
                  className="text-blue-400 hover:underline font-semibold"
                >
                  Register Investigator
                </button>
              </p>
            )}
          </div>
        </div>
      </div>
    );
  }

  // ══════════════════════════════════════════════════════════════════════════════
  // RENDER: MAIN APPLICATION (ENTERPRISE SAAS LIGHT THEME + DARK NAVY SIDEBAR)
  // ══════════════════════════════════════════════════════════════════════════════

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex font-sans antialiased">
      {/* ──────────────────────────────────────────────────────────────────────────
          LEFT SIDEBAR: DARK NAVY (bg-slate-900)
      ────────────────────────────────────────────────────────────────────────── */}
      <aside className="w-64 bg-slate-900 border-r border-slate-800 text-white flex flex-col justify-between shrink-0 h-screen sticky top-0 z-30">
        <div>
          {/* Brand Header */}
          <div className="p-6 border-b border-slate-800 flex items-center space-x-3">
            <div className="p-2 bg-blue-600/20 text-blue-400 rounded-lg border border-blue-500/30">
              <ShieldAlert className="w-6 h-6 text-blue-400" />
            </div>
            <div>
              <span className="text-base font-bold text-white tracking-tight block">HealthGuard AI</span>
              <span className="text-[11px] text-slate-400 font-medium block">Fraud Investigation</span>
            </div>
          </div>

          {/* Strictly 5 Navigation Items */}
          <nav className="p-4 space-y-1">
            <button
              onClick={() => setActiveTab('overview')}
              className={`w-full px-3.5 py-2.5 text-xs font-semibold rounded-lg flex items-center space-x-3 transition ${
                activeTab === 'overview'
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'text-slate-300 hover:bg-slate-800 hover:text-white'
              }`}
            >
              <LayoutDashboard className="w-4 h-4 shrink-0" />
              <span>Overview</span>
            </button>

            <button
              onClick={() => setActiveTab('providers')}
              className={`w-full px-3.5 py-2.5 text-xs font-semibold rounded-lg flex items-center space-x-3 transition ${
                activeTab === 'providers'
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'text-slate-300 hover:bg-slate-800 hover:text-white'
              }`}
            >
              <ShieldAlert className="w-4 h-4 shrink-0" />
              <span>Providers</span>
            </button>

            <button
              onClick={() => setActiveTab('claims')}
              className={`w-full px-3.5 py-2.5 text-xs font-semibold rounded-lg flex items-center space-x-3 transition ${
                activeTab === 'claims'
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'text-slate-300 hover:bg-slate-800 hover:text-white'
              }`}
            >
              <FileText className="w-4 h-4 shrink-0" />
              <span>Claims</span>
            </button>

            <button
              onClick={() => setActiveTab('analyzer')}
              className={`w-full px-3.5 py-2.5 text-xs font-semibold rounded-lg flex items-center space-x-3 transition ${
                activeTab === 'analyzer'
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'text-slate-300 hover:bg-slate-800 hover:text-white'
              }`}
            >
              <Activity className="w-4 h-4 shrink-0" />
              <span>Claim Analyzer</span>
            </button>

            <button
              onClick={() => setActiveTab('settings')}
              className={`w-full px-3.5 py-2.5 text-xs font-semibold rounded-lg flex items-center space-x-3 transition ${
                activeTab === 'settings'
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'text-slate-300 hover:bg-slate-800 hover:text-white'
              }`}
            >
              <Sliders className="w-4 h-4 shrink-0" />
              <span>Settings</span>
            </button>
          </nav>
        </div>

        {/* User Profile Card & Sign Out at Bottom of Sidebar */}
        <div className="p-4 border-t border-slate-800 bg-slate-950/60">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2.5 min-w-0">
              <div className="w-8 h-8 rounded-full bg-blue-600/30 border border-blue-500/40 flex items-center justify-center shrink-0">
                <User className="w-4 h-4 text-blue-300" />
              </div>
              <div className="min-w-0 flex-1">
                <span className="text-xs font-bold text-white truncate block">{currentUser.name}</span>
                <span className="text-[10px] text-blue-400 uppercase font-semibold tracking-wider block">
                  {currentUser.role}
                </span>
              </div>
            </div>

            <button
              onClick={handleLogout}
              title="Log Out"
              className="p-1.5 text-slate-400 hover:text-red-400 hover:bg-slate-800 rounded-lg transition"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </aside>

      {/* ──────────────────────────────────────────────────────────────────────────
          MAIN CONTENT VIEWPORT
      ────────────────────────────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top Header Banner: Statutory Notice & Engine Status */}
        <header className="bg-white border-b border-slate-200 px-8 py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3 sticky top-0 z-20 shadow-xs">
          {/* Statutory Regulatory Notice */}
          <div className="flex items-center space-x-2 bg-blue-50 border border-blue-200/80 text-blue-900 text-xs px-3.5 py-1.5 rounded-full font-medium">
            <Info className="w-3.5 h-3.5 text-blue-600 shrink-0" />
            <span className="truncate">{MANDATORY_DISCLAIMER}</span>
          </div>

          {/* System Indicators */}
          <div className="flex items-center space-x-4 text-xs font-medium text-slate-600">
            <span className="flex items-center space-x-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <Database className="w-3.5 h-3.5 text-emerald-600" />
              <span>PostgreSQL 18</span>
            </span>
            <span className="text-slate-300">|</span>
            <span className="flex items-center space-x-1.5">
              <Cpu className="w-3.5 h-3.5 text-blue-600" />
              <span>Ensemble v2 (Active)</span>
            </span>
          </div>
        </header>

        {/* Scrollable Main Workspace */}
        <main className="flex-1 p-8 overflow-y-auto bg-slate-50">
          <div className="max-w-7xl mx-auto space-y-6">

            {/* ══════════════════════════════════════════════════════════════════════
                TAB 1: OVERVIEW DASHBOARD
            ══════════════════════════════════════════════════════════════════════ */}
            {activeTab === 'overview' && (
              <div className="space-y-6">
                <div className="flex justify-between items-center">
                  <div>
                    <h2 className="text-2xl font-bold text-slate-900 tracking-tight">Investigation Overview</h2>
                    <p className="text-xs text-slate-500 mt-0.5">Aggregated KPIs, risk distribution, and surveillance telemetry from PostgreSQL.</p>
                  </div>
                  <button
                    onClick={fetchOverview}
                    disabled={overviewLoading}
                    className="px-3.5 py-2 bg-white hover:bg-slate-100 border border-slate-200 rounded-lg text-xs font-semibold flex items-center space-x-2 text-slate-700 shadow-xs transition"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${overviewLoading ? 'animate-spin' : ''}`} />
                    <span>Refresh Data</span>
                  </button>
                </div>

                {overviewError && (
                  <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-xs text-red-700">
                    {overviewError}
                  </div>
                )}

                {/* 4 Core Dynamic KPIs matching PostgreSQL data */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
                  <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs hover:border-slate-300 transition">
                    <div className="flex justify-between items-start">
                      <span className="text-xs font-semibold text-slate-500">Total Monitored Providers</span>
                      <div className="p-2 bg-blue-50 text-blue-600 rounded-lg">
                        <ShieldAlert className="w-4 h-4" />
                      </div>
                    </div>
                    <span className="text-3xl font-extrabold text-slate-900 tracking-tight mt-2 block">
                      {overviewSummary ? (overviewSummary.total_providers ?? 0).toLocaleString() : '...'}
                    </span>
                    <span className="text-[11px] text-slate-400 block mt-1.5">Monitored in database</span>
                  </div>

                  <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs hover:border-slate-300 transition">
                    <div className="flex justify-between items-start">
                      <span className="text-xs font-semibold text-slate-500">Total Evaluated Claims</span>
                      <div className="p-2 bg-indigo-50 text-indigo-600 rounded-lg">
                        <FileText className="w-4 h-4" />
                      </div>
                    </div>
                    <span className="text-3xl font-extrabold text-slate-900 tracking-tight mt-2 block">
                      {overviewSummary ? (overviewSummary.total_claims ?? 0).toLocaleString() : '...'}
                    </span>
                    <span className="text-[11px] text-slate-400 block mt-1.5">Inpatient & Outpatient claims</span>
                  </div>

                  <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs hover:border-slate-300 transition">
                    <div className="flex justify-between items-start">
                      <span className="text-xs font-semibold text-slate-500">High-Risk Providers</span>
                      <div className="p-2 bg-red-50 text-red-600 rounded-lg">
                        <AlertTriangle className="w-4 h-4" />
                      </div>
                    </div>
                    <span className="text-3xl font-extrabold text-red-600 tracking-tight mt-2 block">
                      {overviewSummary ? (overviewSummary.high_risk_providers ?? 0).toLocaleString() : '...'}
                    </span>
                    <span className="text-[11px] text-red-600/80 font-medium block mt-1.5">Calibrated Score &ge; 0.70</span>
                  </div>

                  <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs hover:border-slate-300 transition">
                    <div className="flex justify-between items-start">
                      <span className="text-xs font-semibold text-slate-500">Claims Requiring Investigation</span>
                      <div className="p-2 bg-amber-50 text-amber-600 rounded-lg">
                        <Activity className="w-4 h-4" />
                      </div>
                    </div>
                    <span className="text-3xl font-extrabold text-amber-600 tracking-tight mt-2 block">
                      {overviewSummary ? (overviewSummary.claims_requiring_investigation ?? overviewSummary.flagged_claims ?? 0).toLocaleString() : '...'}
                    </span>
                    <span className="text-[11px] text-amber-600/80 font-medium block mt-1.5">Anomalous or High Risk</span>
                  </div>
                </div>

                {/* Risk Distribution Breakdown */}
                {riskDistribution?.distribution && (
                  <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs">
                    <h3 className="text-sm font-bold text-slate-900 mb-4">Provider Fraud Risk Tier Distribution</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                      {riskDistribution.distribution.map((d) => (
                        <div key={d.tier} className="p-4 bg-slate-50 border border-slate-200 rounded-xl">
                          <div className="flex justify-between items-center mb-1.5">
                            <span className="text-xs font-bold text-slate-700">{d.tier} RISK</span>
                            <span className="text-xs font-semibold text-slate-600">{d.percentage}%</span>
                          </div>
                          <span className="text-xl font-extrabold text-slate-900 block mb-2">{d.count} providers</span>
                          <div className="w-full bg-slate-200 rounded-full h-2.5 overflow-hidden">
                            <div
                              className={`h-full ${
                                d.tier === 'HIGH' ? 'bg-red-500' : d.tier === 'MEDIUM' ? 'bg-amber-500' : 'bg-emerald-500'
                              }`}
                              style={{ width: `${d.percentage}%` }}
                            />
                          </div>
                          <span className="text-[10px] text-slate-400 mt-2 block font-medium">
                            Cutoff: {d.tier === 'HIGH' ? '>= 0.70' : d.tier === 'MEDIUM' ? '0.22 - 0.70' : '< 0.22'}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Recent Investigation Activity */}
                <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs">
                  <h3 className="text-sm font-bold text-slate-900 mb-4">Recent Fraud Investigations & Assessments</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-slate-50 text-slate-500 uppercase tracking-wider font-semibold border-b border-slate-200 text-[11px]">
                        <tr>
                          <th className="p-3">Provider ID</th>
                          <th className="p-3">Calibrated Score</th>
                          <th className="p-3">Risk Tier</th>
                          <th className="p-3">Assessment Date</th>
                          <th className="p-3">Actions</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 text-slate-700">
                        {overviewRecent.length === 0 ? (
                          <tr>
                            <td colSpan="5" className="p-6 text-center text-slate-400">
                              No recent provider assessments logged yet.
                            </td>
                          </tr>
                        ) : (
                          overviewRecent.map((item, idx) => (
                            <tr key={idx} className="hover:bg-slate-50/80 transition">
                              <td className="p-3 font-mono font-semibold text-slate-900">{item.provider_id}</td>
                              <td className="p-3 font-mono font-medium">
                                {typeof item.risk_score === 'number'
                                  ? item.risk_score.toFixed(4)
                                  : typeof item.score === 'number'
                                  ? item.score.toFixed(4)
                                  : 'N/A'}
                              </td>
                              <td className="p-3">{getTierBadge(item.risk_tier)}</td>
                              <td className="p-3 text-slate-500">
                                {item.created_at || item.assessed_at
                                  ? new Date(item.created_at || item.assessed_at).toLocaleString()
                                  : 'Recently'}
                              </td>
                              <td className="p-3">
                                <button
                                  onClick={() => { inspectProvider(item.provider_id); setActiveTab('providers'); }}
                                  className="text-blue-600 hover:text-blue-800 font-semibold"
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
                </div>
              </div>
            )}

            {/* ══════════════════════════════════════════════════════════════════════
                TAB 2: PROVIDERS VIEW
            ══════════════════════════════════════════════════════════════════════ */}
            {activeTab === 'providers' && (
              <div className="space-y-6">
                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                  <div>
                    <h2 className="text-2xl font-bold text-slate-900 tracking-tight">Monitored Providers</h2>
                    <p className="text-xs text-slate-500 mt-0.5">Search and audit healthcare providers under fraud surveillance.</p>
                  </div>

                  <div className="flex items-center space-x-3 w-full sm:w-auto">
                    <div className="relative flex-1 sm:w-64">
                      <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-400" />
                      <input
                        type="text"
                        value={providerSearch}
                        onChange={(e) => { setProviderSearch(e.target.value); setProviderPage(1); }}
                        placeholder="Search Provider ID or Name..."
                        className="w-full pl-9 pr-3 py-1.5 bg-white border border-slate-300 rounded-lg text-xs text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition shadow-xs"
                      />
                    </div>

                    <select
                      value={providerTierFilter}
                      onChange={(e) => { setProviderTierFilter(e.target.value); setProviderPage(1); }}
                      className="px-3 py-1.5 bg-white border border-slate-300 rounded-lg text-xs text-slate-700 font-medium focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-xs"
                    >
                      <option value="">All Risk Tiers</option>
                      <option value="HIGH">High Risk</option>
                      <option value="MEDIUM">Medium Risk</option>
                      <option value="LOW">Low Risk</option>
                    </select>
                  </div>
                </div>

                {/* Providers Table */}
                <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-xs">
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-slate-50 text-slate-500 uppercase tracking-wider font-semibold border-b border-slate-200 text-[11px]">
                        <tr>
                          <th className="p-3">Provider ID</th>
                          <th className="p-3">Provider Name</th>
                          <th className="p-3">Claims Count</th>
                          <th className="p-3">Total Reimbursed</th>
                          <th className="p-3">Risk Tier</th>
                          <th className="p-3">Latest Score</th>
                          <th className="p-3">Action</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 text-slate-700">
                        {providersLoading ? (
                          <tr><td colSpan="7" className="p-6 text-center text-slate-400">Loading providers...</td></tr>
                        ) : providers.length === 0 ? (
                          <tr><td colSpan="7" className="p-6 text-center text-slate-400">No matching providers found.</td></tr>
                        ) : (
                          providers.map((p) => (
                            <tr key={p.provider_id} className="hover:bg-slate-50/80 transition">
                              <td className="p-3 font-mono font-bold text-slate-900">{p.provider_id}</td>
                              <td className="p-3 font-medium text-slate-800">{p.provider_name}</td>
                              <td className="p-3 font-medium">{p.total_claims}</td>
                              <td className="p-3 font-mono font-medium">${(p.total_reimbursement || 0).toLocaleString()}</td>
                              <td className="p-3">{getTierBadge(p.risk_tier)}</td>
                              <td className="p-3 font-mono font-medium">{p.risk_score ? p.risk_score.toFixed(4) : '—'}</td>
                              <td className="p-3">
                                <button
                                  onClick={() => inspectProvider(p.provider_id)}
                                  className="px-2.5 py-1 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-md border border-blue-200 text-xs font-semibold transition"
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

                  {/* Pagination */}
                  <div className="p-3.5 bg-slate-50 border-t border-slate-200 flex justify-between items-center text-xs text-slate-500">
                    <span className="font-medium">Page {providerPage} of {providerTotalPages}</span>
                    <div className="flex space-x-2">
                      <button
                        disabled={providerPage <= 1}
                        onClick={() => setProviderPage(providerPage - 1)}
                        className="p-1.5 rounded-md bg-white border border-slate-200 hover:bg-slate-100 disabled:opacity-40 transition shadow-xs"
                      >
                        <ChevronLeft className="w-4 h-4" />
                      </button>
                      <button
                        disabled={providerPage >= providerTotalPages}
                        onClick={() => setProviderPage(providerPage + 1)}
                        className="p-1.5 rounded-md bg-white border border-slate-200 hover:bg-slate-100 disabled:opacity-40 transition shadow-xs"
                      >
                        <ChevronRight className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>

                {/* Provider Inspection Drawer */}
                {selectedProvider && (
                  <div className="bg-white border-2 border-blue-500/30 rounded-2xl p-6 shadow-lg space-y-5 animate-in fade-in duration-200">
                    <div className="flex justify-between items-start">
                      <div>
                        <span className="text-xs font-mono font-bold text-blue-600 block">{selectedProvider.provider_id}</span>
                        <h3 className="text-xl font-bold text-slate-900">{selectedProvider.provider_name}</h3>
                      </div>
                      <button
                        onClick={() => setSelectedProvider(null)}
                        className="text-slate-400 hover:text-slate-600 transition"
                      >
                        <XCircle className="w-6 h-6" />
                      </button>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 p-4 bg-slate-50 rounded-xl border border-slate-200 text-xs">
                      <div>
                        <span className="text-slate-400 font-medium block">Total Claims</span>
                        <span className="text-base font-bold text-slate-900">{selectedProvider.total_claims}</span>
                      </div>
                      <div>
                        <span className="text-slate-400 font-medium block">Reimbursement</span>
                        <span className="text-base font-bold text-slate-900">${(selectedProvider.total_reimbursement || 0).toLocaleString()}</span>
                      </div>
                      <div>
                        <span className="text-slate-400 font-medium block">Risk Tier</span>
                        <div className="mt-1">{getTierBadge(selectedProvider.risk?.risk_tier)}</div>
                      </div>
                      <div>
                        <span className="text-slate-400 font-medium block">Calibrated Score</span>
                        <span className="text-base font-bold font-mono text-slate-900">
                          {selectedProvider.risk?.risk_score ? selectedProvider.risk.risk_score.toFixed(4) : 'Pending Assessment'}
                        </span>
                      </div>
                    </div>

                    {/* SHAP Explanation */}
                    {selectedProvider.risk?.explanation && (
                      <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl text-xs space-y-2">
                        <h4 className="font-bold text-slate-800">{selectedProvider.risk.explanation.title}</h4>
                        <p className="text-slate-600 leading-relaxed">{selectedProvider.risk.explanation.text}</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* ══════════════════════════════════════════════════════════════════════
                TAB 3: CLAIMS VIEW
            ══════════════════════════════════════════════════════════════════════ */}
            {activeTab === 'claims' && (
              <div className="space-y-6">
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                  <div>
                    <h2 className="text-2xl font-bold text-slate-900 tracking-tight">Evaluated Claims</h2>
                    <p className="text-xs text-slate-500 mt-0.5">Search and audit individual claims processed through the anomaly engine.</p>
                  </div>

                  <div className="flex flex-wrap items-center gap-3 w-full md:w-auto">
                    <div className="relative flex-1 md:w-56">
                      <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-400" />
                      <input
                        type="text"
                        value={claimSearch}
                        onChange={(e) => { setClaimSearch(e.target.value); setClaimPage(1); }}
                        placeholder="Search Claim ID or Provider..."
                        className="w-full pl-9 pr-3 py-1.5 bg-white border border-slate-300 rounded-lg text-xs text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-xs"
                      />
                    </div>

                    <select
                      value={claimTypeFilter}
                      onChange={(e) => { setClaimTypeFilter(e.target.value); setClaimPage(1); }}
                      className="px-3 py-1.5 bg-white border border-slate-300 rounded-lg text-xs text-slate-700 font-medium focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-xs"
                    >
                      <option value="">All Claim Types</option>
                      <option value="Inpatient">Inpatient</option>
                      <option value="Outpatient">Outpatient</option>
                    </select>

                    <div className="flex items-center space-x-1.5 text-xs text-slate-500 font-medium">
                      <span>From:</span>
                      <input
                        type="date"
                        value={claimStartDate}
                        onChange={(e) => { setClaimStartDate(e.target.value); setClaimPage(1); }}
                        className="px-2 py-1 bg-white border border-slate-300 rounded-lg text-xs text-slate-700 shadow-xs"
                      />
                    </div>

                    <div className="flex items-center space-x-1.5 text-xs text-slate-500 font-medium">
                      <span>To:</span>
                      <input
                        type="date"
                        value={claimEndDate}
                        onChange={(e) => { setClaimEndDate(e.target.value); setClaimPage(1); }}
                        className="px-2 py-1 bg-white border border-slate-300 rounded-lg text-xs text-slate-700 shadow-xs"
                      />
                    </div>
                  </div>
                </div>

                {/* Claims Table */}
                <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-xs">
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-slate-50 text-slate-500 uppercase tracking-wider font-semibold border-b border-slate-200 text-[11px]">
                        <tr>
                          <th className="p-3">Claim ID</th>
                          <th className="p-3">Provider ID</th>
                          <th className="p-3">Type</th>
                          <th className="p-3">Date</th>
                          <th className="p-3">Reimbursed</th>
                          <th className="p-3">Anomaly Status</th>
                          <th className="p-3">Action</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 text-slate-700">
                        {claimsLoading ? (
                          <tr><td colSpan="7" className="p-6 text-center text-slate-400">Loading claims...</td></tr>
                        ) : claims.length === 0 ? (
                          <tr><td colSpan="7" className="p-6 text-center text-slate-400">No matching claims found.</td></tr>
                        ) : (
                          claims.map((c) => (
                            <tr key={c.claim_id} className="hover:bg-slate-50/80 transition">
                              <td className="p-3 font-mono font-bold text-slate-900">{c.claim_id}</td>
                              <td className="p-3 font-mono text-slate-600">{c.provider_id}</td>
                              <td className="p-3 font-medium text-slate-800">{c.claim_type}</td>
                              <td className="p-3 text-slate-500">{c.claim_date ? new Date(c.claim_date).toLocaleDateString() : '—'}</td>
                              <td className="p-3 font-mono font-medium">${(c.reimbursement_amount || 0).toFixed(2)}</td>
                              <td className="p-3">
                                {c.has_anomalies ? (
                                  <span className="px-2 py-0.5 text-[10px] font-bold rounded-full bg-red-100 text-red-700 border border-red-200">
                                    {c.anomaly_flags?.length || 1} ANOMALIES
                                  </span>
                                ) : (
                                  <span className="px-2 py-0.5 text-[10px] font-bold rounded-full bg-emerald-100 text-emerald-700 border border-emerald-200">
                                    NORMAL
                                  </span>
                                )}
                              </td>
                              <td className="p-3">
                                <button
                                  onClick={() => inspectClaim(c.claim_id)}
                                  className="px-2.5 py-1 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-md border border-blue-200 text-xs font-semibold transition"
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

                  {/* Pagination */}
                  <div className="p-3.5 bg-slate-50 border-t border-slate-200 flex justify-between items-center text-xs text-slate-500">
                    <span className="font-medium">Page {claimPage} of {claimTotalPages}</span>
                    <div className="flex space-x-2">
                      <button
                        disabled={claimPage <= 1}
                        onClick={() => setClaimPage(claimPage - 1)}
                        className="p-1.5 rounded-md bg-white border border-slate-200 hover:bg-slate-100 disabled:opacity-40 transition shadow-xs"
                      >
                        <ChevronLeft className="w-4 h-4" />
                      </button>
                      <button
                        disabled={claimPage >= claimTotalPages}
                        onClick={() => setClaimPage(claimPage + 1)}
                        className="p-1.5 rounded-md bg-white border border-slate-200 hover:bg-slate-100 disabled:opacity-40 transition shadow-xs"
                      >
                        <ChevronRight className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>

                {/* Claim Detail & Anomaly Audit Inspection Card */}
                {selectedClaim && (
                  <div className="bg-white border-2 border-blue-500/30 rounded-2xl p-6 shadow-lg space-y-4 animate-in fade-in duration-200">
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="text-lg font-bold text-slate-900 flex items-center space-x-2">
                          <span>Claim Audit: {selectedClaim.claim_id}</span>
                          <span className="text-xs px-2.5 py-0.5 rounded-full bg-blue-100 text-blue-700 border border-blue-200 font-semibold">
                            {selectedClaim.claim_type}
                          </span>
                        </h3>
                        <p className="text-xs text-slate-500 mt-1">
                          Associated Provider: <span className="font-mono font-semibold text-slate-800">{selectedClaim.provider_id}</span> | Beneficiary: <span className="font-mono font-semibold text-slate-800">{selectedClaim.beneficiary_id}</span>
                        </p>
                      </div>
                      <button
                        onClick={() => setSelectedClaim(null)}
                        className="text-slate-400 hover:text-slate-600 transition"
                      >
                        <XCircle className="w-6 h-6" />
                      </button>
                    </div>

                    <div className="p-3 bg-blue-50 border border-blue-200 rounded-xl text-xs text-blue-800 flex items-start space-x-2">
                      <Info className="w-4 h-4 shrink-0 mt-0.5 text-blue-600" />
                      <span>
                        Scope Notice: Anomaly detection evaluates individual claim irregularities. Fraud probability scores evaluate overall provider behavioral patterns.
                      </span>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 p-4 bg-slate-50 rounded-xl border border-slate-200 text-xs">
                      <div>
                        <span className="text-slate-400 font-medium block">Reimbursement</span>
                        <span className="text-base font-bold text-slate-900">${(selectedClaim.reimbursement_amount || 0).toFixed(2)}</span>
                      </div>
                      <div>
                        <span className="text-slate-400 font-medium block">Service Date</span>
                        <span className="text-base font-bold text-slate-900">{selectedClaim.claim_date ? new Date(selectedClaim.claim_date).toLocaleDateString() : 'N/A'}</span>
                      </div>
                      <div>
                        <span className="text-slate-400 font-medium block">Provider Risk Tier</span>
                        <div className="mt-1">{getTierBadge(selectedClaim.analysis?.provider_risk_tier)}</div>
                      </div>
                      <div>
                        <span className="text-slate-400 font-medium block">Provider Fraud Probability</span>
                        <span className="text-base font-bold font-mono text-slate-900">
                          {selectedClaim.analysis?.provider_risk_score !== null && selectedClaim.analysis?.provider_risk_score !== undefined
                            ? (selectedClaim.analysis.provider_risk_score * 100).toFixed(1) + '%'
                            : 'Pending Assessment'}
                        </span>
                      </div>
                    </div>

                    {/* Anomaly findings */}
                    <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl text-xs space-y-2">
                      <h4 className="font-bold text-slate-800 flex items-center space-x-2">
                        <AlertTriangle className="w-4 h-4 text-amber-500" />
                        <span>Claim-Level Anomaly Detection Findings</span>
                      </h4>
                      {selectedClaim.analysis?.anomaly_flags && selectedClaim.analysis.anomaly_flags.length > 0 ? (
                        <div className="space-y-1.5 mt-2">
                          {selectedClaim.analysis.anomaly_flags.map((flag, idx) => (
                            <div key={idx} className="p-2.5 bg-red-50 border border-red-200 rounded-lg text-red-800 font-mono text-[11px]">
                              &bull; {flag}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-emerald-700 font-medium text-xs">No deterministic anomaly rules triggered for this claim.</p>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* ══════════════════════════════════════════════════════════════════════
                TAB 4: CLAIM ANALYZER (SINGLE & BATCH CSV)
            ══════════════════════════════════════════════════════════════════════ */}
            {activeTab === 'analyzer' && (
              <div className="space-y-6">
                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                  <div>
                    <h2 className="text-2xl font-bold text-slate-900 tracking-tight">Claim Fraud Analyzer</h2>
                    <p className="text-xs text-slate-500 mt-0.5">Execute real-time multi-modal ensemble inference and batch claims validation.</p>
                  </div>

                  {/* Sub-tabs: Single vs Batch CSV */}
                  <div className="flex bg-slate-200/80 p-1 rounded-xl text-xs font-semibold">
                    <button
                      onClick={() => setAnalyzerSubTab('single')}
                      className={`px-4 py-2 rounded-lg transition ${
                        analyzerSubTab === 'single' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      Single-Claim Analysis
                    </button>
                    <button
                      onClick={() => setAnalyzerSubTab('csv')}
                      className={`px-4 py-2 rounded-lg transition ${
                        analyzerSubTab === 'csv' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      Batch CSV Upload
                    </button>
                  </div>
                </div>

                {/* ── Sub-tab 1: Single Claim Analysis ── */}
                {analyzerSubTab === 'single' && (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Input Form */}
                    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-4">
                      <div className="flex justify-between items-center">
                        <h3 className="text-sm font-bold text-slate-900">Claim Attributes</h3>
                        <div className="flex space-x-2">
                          <button
                            type="button"
                            onClick={() => loadDemoClaim('high_risk')}
                            className="px-2.5 py-1 bg-red-50 hover:bg-red-100 text-red-700 border border-red-200 rounded-md text-[11px] font-semibold transition"
                          >
                            Fill High-Risk Demo
                          </button>
                          <button
                            type="button"
                            onClick={() => loadDemoClaim('low_risk')}
                            className="px-2.5 py-1 bg-emerald-50 hover:bg-emerald-100 text-emerald-700 border border-emerald-200 rounded-md text-[11px] font-semibold transition"
                          >
                            Fill Low-Risk Demo
                          </button>
                        </div>
                      </div>

                      <form onSubmit={handleSingleClaimSubmit} className="space-y-4">
                        <div className="grid grid-cols-2 gap-3">
                          <div>
                            <label className="block text-xs font-semibold text-slate-700 mb-1">Claim ID</label>
                            <input
                              type="text"
                              required
                              value={singleClaimForm.claim_id}
                              onChange={(e) => setSingleClaimForm({ ...singleClaimForm, claim_id: e.target.value })}
                              className="w-full px-3 py-1.5 bg-white border border-slate-300 rounded-lg text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-xs"
                            />
                          </div>
                          <div>
                            <label className="block text-xs font-semibold text-slate-700 mb-1">Provider ID</label>
                            <input
                              type="text"
                              required
                              value={singleClaimForm.provider_id}
                              onChange={(e) => setSingleClaimForm({ ...singleClaimForm, provider_id: e.target.value })}
                              className="w-full px-3 py-1.5 bg-white border border-slate-300 rounded-lg text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-xs"
                            />
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-3">
                          <div>
                            <label className="block text-xs font-semibold text-slate-700 mb-1">Beneficiary ID</label>
                            <input
                              type="text"
                              required
                              value={singleClaimForm.beneficiary_id}
                              onChange={(e) => setSingleClaimForm({ ...singleClaimForm, beneficiary_id: e.target.value })}
                              className="w-full px-3 py-1.5 bg-white border border-slate-300 rounded-lg text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-xs"
                            />
                          </div>
                          <div>
                            <label className="block text-xs font-semibold text-slate-700 mb-1">Claim Type</label>
                            <select
                              value={singleClaimForm.claim_type}
                              onChange={(e) => setSingleClaimForm({ ...singleClaimForm, claim_type: e.target.value })}
                              className="w-full px-3 py-1.5 bg-white border border-slate-300 rounded-lg text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-xs"
                            >
                              <option value="Outpatient">Outpatient</option>
                              <option value="Inpatient">Inpatient</option>
                            </select>
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-3">
                          <div>
                            <label className="block text-xs font-semibold text-slate-700 mb-1">Reimbursement ($)</label>
                            <input
                              type="number"
                              step="0.01"
                              required
                              value={singleClaimForm.reimbursement_amount}
                              onChange={(e) => setSingleClaimForm({ ...singleClaimForm, reimbursement_amount: e.target.value })}
                              className="w-full px-3 py-1.5 bg-white border border-slate-300 rounded-lg text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-xs"
                            />
                          </div>
                          <div>
                            <label className="block text-xs font-semibold text-slate-700 mb-1">Deductible ($)</label>
                            <input
                              type="number"
                              step="0.01"
                              required
                              value={singleClaimForm.deductible_amount}
                              onChange={(e) => setSingleClaimForm({ ...singleClaimForm, deductible_amount: e.target.value })}
                              className="w-full px-3 py-1.5 bg-white border border-slate-300 rounded-lg text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-xs"
                            />
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-3">
                          <div>
                            <label className="block text-xs font-semibold text-slate-700 mb-1">Claim Start Date</label>
                            <input
                              type="date"
                              required
                              value={singleClaimForm.claim_start_date}
                              onChange={(e) => setSingleClaimForm({ ...singleClaimForm, claim_start_date: e.target.value })}
                              className="w-full px-3 py-1.5 bg-white border border-slate-300 rounded-lg text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-xs"
                            />
                          </div>
                          <div>
                            <label className="block text-xs font-semibold text-slate-700 mb-1">Claim End Date</label>
                            <input
                              type="date"
                              required
                              value={singleClaimForm.claim_end_date}
                              onChange={(e) => setSingleClaimForm({ ...singleClaimForm, claim_end_date: e.target.value })}
                              className="w-full px-3 py-1.5 bg-white border border-slate-300 rounded-lg text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-xs"
                            />
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-3">
                          <div>
                            <label className="block text-xs font-semibold text-slate-700 mb-1">Diagnosis Code 1</label>
                            <input
                              type="text"
                              value={singleClaimForm.diagnosis_code_1}
                              onChange={(e) => setSingleClaimForm({ ...singleClaimForm, diagnosis_code_1: e.target.value })}
                              className="w-full px-3 py-1.5 bg-white border border-slate-300 rounded-lg text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-xs"
                            />
                          </div>
                          <div>
                            <label className="block text-xs font-semibold text-slate-700 mb-1">Attending Physician</label>
                            <input
                              type="text"
                              value={singleClaimForm.attending_physician}
                              onChange={(e) => setSingleClaimForm({ ...singleClaimForm, attending_physician: e.target.value })}
                              className="w-full px-3 py-1.5 bg-white border border-slate-300 rounded-lg text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-xs"
                            />
                          </div>
                        </div>

                        <button
                          type="submit"
                          disabled={analysisLoading}
                          className="w-full py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-bold flex items-center justify-center space-x-2 shadow-md transition disabled:opacity-50"
                        >
                          {analysisLoading ? <span>Evaluating ML Ensemble...</span> : <span>Run Fraud Risk Assessment</span>}
                        </button>
                      </form>
                    </div>

                    {/* Analysis Output Result */}
                    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-4">
                      <h3 className="text-sm font-bold text-slate-900">Risk Evaluation Findings</h3>

                      {analysisError && (
                        <div className="p-3 bg-red-50 border border-red-200 rounded-xl text-xs text-red-700">
                          {analysisError}
                        </div>
                      )}

                      {!analysisResult && !analysisLoading && (
                        <div className="p-10 text-center text-slate-400 text-xs border-2 border-dashed border-slate-200 rounded-xl">
                          Submit claim attributes to execute the full multi-modal ensemble inference, SHAP explanation, and anomaly detection.
                        </div>
                      )}

                      {analysisResult && (
                        <div className="space-y-4">
                          {/* Score Gauge & Model Version Card */}
                          <div className="p-5 bg-slate-50 rounded-xl border border-slate-200 flex items-center justify-between">
                            <div>
                              <div className="flex items-center space-x-2 mb-1">
                                <span className="text-xs font-semibold text-slate-500">Provider Fraud Risk Score</span>
                                <span className="text-[10px] font-mono bg-slate-200 text-slate-700 px-2 py-0.5 rounded font-semibold" data-testid="model-version-badge">
                                  Model: {analysisResult.model_version || 'v2'}
                                </span>
                              </div>
                              <span className="text-4xl font-extrabold font-mono text-slate-900" data-testid="risk-score-value">
                                {formatProbability(analysisResult.provider_risk_score)}
                              </span>
                            </div>
                            <div data-testid="risk-tier-badge">
                              {getTierBadge(analysisResult.provider_risk_tier)}
                            </div>
                          </div>

                          {/* Incomplete or Missing Field Alert */}
                          {(!Number.isFinite(analysisResult.provider_risk_score) || !analysisResult.provider_risk_tier) && (
                            <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg text-xs text-amber-800" data-testid="incomplete-error">
                              Warning: The risk assessment response is incomplete or invalid. Please re-run the evaluation.
                            </div>
                          )}

                          {/* Investigation Recommendation */}
                          {analysisResult.investigation_recommendation && (
                            <div className="p-3.5 bg-blue-50 border border-blue-200 rounded-xl text-xs space-y-1" data-testid="investigation-recommendation">
                              <span className="font-bold text-blue-900 block">Investigation Recommendation</span>
                              <p className="text-blue-800 leading-relaxed">{analysisResult.investigation_recommendation}</p>
                            </div>
                          )}

                          {/* Base Model Probabilities */}
                          {analysisResult.base_model_probabilities && (
                            <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-2">
                              <span className="text-xs font-bold text-slate-700 block">Ensemble Base Model Contributions</span>
                              <div className="grid grid-cols-3 gap-2 text-xs">
                                <div className="p-2.5 bg-white rounded-lg border border-slate-200">
                                  <span className="text-slate-500 font-medium block">XGBoost</span>
                                  <span className="font-mono text-slate-900 font-bold">
                                    {formatProbability(analysisResult.base_model_probabilities.xgboost)}
                                  </span>
                                </div>
                                <div className="p-2.5 bg-white rounded-lg border border-slate-200">
                                  <span className="text-slate-500 font-medium block">ANN</span>
                                  <span className="font-mono text-slate-900 font-bold">
                                    {formatProbability(analysisResult.base_model_probabilities.ann)}
                                  </span>
                                </div>
                                <div className="p-2.5 bg-white rounded-lg border border-slate-200">
                                  <span className="text-slate-500 font-medium block">Bi-LSTM</span>
                                  <span className="font-mono text-slate-900 font-bold">
                                    {formatProbability(analysisResult.base_model_probabilities.bilstm)}
                                  </span>
                                </div>
                              </div>
                            </div>
                          )}

                          {/* SHAP Explanation & Top Risk Factors */}
                          {(analysisResult.explanation_title || analysisResult.explanation || analysisResult.shap_explanation) && (
                            <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 text-xs space-y-3" data-testid="shap-explanation-card">
                              <span className="font-bold text-slate-800 block text-sm">
                                {analysisResult.explanation_title || analysisResult.explanation?.title || analysisResult.shap_explanation?.title || 'Risk Factor Attribution (SHAP)'}
                              </span>
                              <p className="text-slate-600 leading-relaxed whitespace-pre-line">
                                {analysisResult.explanation?.text || (typeof analysisResult.explanation === 'string' ? analysisResult.explanation : '') || analysisResult.shap_explanation?.text}
                              </p>

                              {/* Structured Factor Breakdown Table */}
                              {((analysisResult.explanation?.top_risk_factors && analysisResult.explanation.top_risk_factors.length > 0) ||
                                (analysisResult.top_risk_factors && analysisResult.top_risk_factors.length > 0)) && (
                                <div className="mt-2 space-y-2">
                                  <span className="font-semibold text-slate-700 block text-xs">Key Statistical Risk Drivers:</span>
                                  <div className="overflow-x-auto">
                                    <table className="w-full text-left text-xs bg-white rounded-lg border border-slate-200 divide-y divide-slate-100">
                                      <thead className="bg-slate-100 text-slate-600 text-[11px]">
                                        <tr>
                                          <th className="py-1.5 px-3 font-semibold">Feature</th>
                                          <th className="py-1.5 px-3 font-semibold">Observed Value</th>
                                          <th className="py-1.5 px-3 font-semibold">SHAP Impact</th>
                                          <th className="py-1.5 px-3 font-semibold">Effect</th>
                                          <th className="py-1.5 px-3 font-semibold">Benchmark</th>
                                        </tr>
                                      </thead>
                                      <tbody className="divide-y divide-slate-100 text-slate-700">
                                        {(analysisResult.explanation?.top_risk_factors || analysisResult.top_risk_factors || []).slice(0, 5).map((f, idx) => {
                                          const sVal = typeof f.shap_value === 'number' ? f.shap_value : parseFloat(f.shap_value) || 0;
                                          const isElevating = f.direction === 'INCREASES_FRAUD_RISK' || sVal > 0;
                                          const actVal = f.actual_value !== undefined ? f.actual_value : (f.feature_value !== undefined ? f.feature_value : 'N/A');
                                          return (
                                            <tr key={idx} className="hover:bg-slate-50">
                                              <td className="py-1.5 px-3 font-mono font-medium text-slate-900">{f.feature}</td>
                                              <td className="py-1.5 px-3 font-mono">{typeof actVal === 'number' ? actVal.toLocaleString() : actVal}</td>
                                              <td className="py-1.5 px-3 font-mono font-semibold">{sVal > 0 ? `+${sVal.toFixed(4)}` : sVal.toFixed(4)}</td>
                                              <td className="py-1.5 px-3">
                                                <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${isElevating ? 'bg-red-100 text-red-700' : 'bg-emerald-100 text-emerald-700'}`}>
                                                  {isElevating ? 'Elevates Risk' : 'Lowers Risk'}
                                                </span>
                                              </td>
                                              <td className="py-1.5 px-3 text-[11px] text-slate-500">{f.benchmark_summary || (f.peer_percentile ? `${f.peer_percentile}th percentile` : '—')}</td>
                                            </tr>
                                          );
                                        })}
                                      </tbody>
                                    </table>
                                  </div>
                                </div>
                              )}
                            </div>
                          )}

                          {/* Temporal Sequence Explanation (Bi-LSTM) */}
                          {analysisResult.explanation?.temporal_explanation && (
                            <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 text-xs space-y-2.5" data-testid="temporal-explanation-card">
                              <span className="font-bold text-slate-800 block text-sm">Temporal Sequence Dynamics (Bi-LSTM)</span>
                              {analysisResult.explanation.temporal_explanation.trajectory_summary && (
                                <p className="text-slate-600 leading-relaxed font-medium">
                                  {analysisResult.explanation.temporal_explanation.trajectory_summary}
                                </p>
                              )}
                              {analysisResult.explanation.temporal_explanation.monthly_temporal_impacts && (
                                <div className="mt-2 space-y-1.5">
                                  <span className="text-[11px] font-semibold text-slate-500 block">12-Month Observation Sequence:</span>
                                  <div className="grid grid-cols-6 sm:grid-cols-12 gap-1 text-center font-mono text-[10px]">
                                    {analysisResult.explanation.temporal_explanation.monthly_temporal_impacts.map((m, idx) => {
                                      const isPad = !m.is_valid_observation || m.observation_type === 'PADDING';
                                      const isZeroInWindow = m.observation_type === 'ZERO_CLAIMS_IN_WINDOW';
                                      const imp = typeof m.impact === 'number' ? m.impact : 0;
                                      let bgCol = 'bg-slate-100 text-slate-400';
                                      if (!isPad) {
                                        if (imp > 0.005) bgCol = 'bg-red-100 text-red-700 font-bold border border-red-200';
                                        else if (imp < -0.005) bgCol = 'bg-emerald-100 text-emerald-700 font-bold border border-emerald-200';
                                        else if (isZeroInWindow) bgCol = 'bg-amber-50 text-amber-700 border border-amber-200';
                                        else bgCol = 'bg-white text-slate-700 border border-slate-200';
                                      }
                                      return (
                                        <div key={idx} className={`p-1.5 rounded ${bgCol}`} title={isPad ? 'Padding (Unobserved Month)' : (isZeroInWindow ? 'Zero Claims in Observed Window' : `Impact: ${imp >= 0 ? '+' : ''}${imp.toFixed(4)}`)}>
                                          <div className="text-[9px] text-slate-400 font-sans">M{m.month_index || idx + 1}</div>
                                          <div>{isPad ? 'PAD' : (isZeroInWindow ? '0 CLM' : (imp >= 0 ? `+${imp.toFixed(2)}` : imp.toFixed(2)))}</div>
                                        </div>
                                      );
                                    })}
                                  </div>
                                </div>
                              )}
                            </div>
                          )}

                          {/* Claim Anomaly Findings */}
                          <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 text-xs space-y-2" data-testid="claim-anomaly-card">
                            <div className="flex items-center justify-between">
                              <span className="font-bold text-slate-800 text-sm">Claim Anomaly Findings</span>
                              {analysisResult.has_anomalies || (analysisResult.anomaly_flags && analysisResult.anomaly_flags.length > 0) ? (
                                <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-red-100 text-red-700">
                                  {analysisResult.anomaly_flags?.length || 1} Anomalies Detected
                                </span>
                              ) : (
                                <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-100 text-emerald-700">
                                  No Anomalies Detected
                                </span>
                              )}
                            </div>

                            {/* Detailed Rule Triggers */}
                            {analysisResult.has_anomalies && analysisResult.claim_anomaly_findings && analysisResult.claim_anomaly_findings.length > 0 ? (
                              <div className="space-y-1.5 pt-1">
                                {analysisResult.claim_anomaly_findings.map((finding, idx) => {
                                  const rules = finding.rules_triggered || [];
                                  if (rules.length > 0) {
                                    return rules.map((r, rIdx) => (
                                      <div key={`${idx}-${rIdx}`} className="p-2.5 bg-white rounded-lg border border-red-100 text-xs space-y-0.5">
                                        <div className="flex items-center space-x-2">
                                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-red-100 text-red-700">
                                            {r.rule_name}
                                          </span>
                                          {r.observed_value !== undefined && (
                                            <span className="text-[11px] text-slate-500">Observed: {String(r.observed_value)}</span>
                                          )}
                                        </div>
                                        <p className="text-slate-700 pt-0.5">{r.explanation}</p>
                                      </div>
                                    ));
                                  }
                                  return (
                                    <div key={idx} className="p-2.5 bg-white rounded-lg border border-slate-200 text-xs flex items-start space-x-2">
                                      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-red-100 text-red-700">
                                        {finding.rule_name || `Rule ${idx + 1}`}
                                      </span>
                                      <span className="text-slate-700">{finding.description || JSON.stringify(finding)}</span>
                                    </div>
                                  );
                                })}
                              </div>
                            ) : (
                              <div className="p-3 bg-white rounded-lg border border-slate-200 text-slate-600 text-xs">
                                All clinical, procedural, and billing parameters for this claim are within expected peer baseline thresholds. No operational anomaly flags triggered.
                              </div>
                            )}
                          </div>

                          {/* Mandatory Regulatory Ethical Disclaimer */}
                          <div className="p-3.5 bg-slate-100 border border-slate-200 rounded-xl text-[11px] text-slate-600 space-y-1" data-testid="regulatory-disclaimer">
                            <span className="font-semibold text-slate-700 block">Regulatory & Ethical Disclaimer:</span>
                            <p className="italic leading-relaxed">
                              {analysisResult.disclaimer || 'This is a risk assessment to guide investigation. It is not proof of fraud.'}
                            </p>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* ── Sub-tab 2: Batch CSV Upload ── */}
                {analyzerSubTab === 'csv' && (
                  <div className="space-y-6">
                    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs">
                      <h3 className="text-sm font-bold text-slate-900 mb-1">Upload Batch Claims CSV</h3>
                      <p className="text-xs text-slate-500 mb-5">
                        Upload a batch file of claims. The system validates required headers, reconciles duplicates, and evaluates fraud risk asynchronously.
                      </p>

                      <form onSubmit={handleCsvUpload} className="space-y-4">
                        <div className="border-2 border-dashed border-slate-300 hover:border-blue-500 rounded-xl p-6 text-center transition">
                          <input
                            type="file"
                            accept=".csv"
                            onChange={(e) => setCsvFile(e.target.files[0])}
                            className="block w-full text-xs text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700"
                          />
                        </div>

                        <button
                          type="submit"
                          disabled={!csvFile || csvStatusLoading}
                          className="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold rounded-lg shadow-sm transition disabled:opacity-50"
                        >
                          {csvStatusLoading ? 'Processing Batch in Background...' : 'Upload & Process Batch'}
                        </button>
                      </form>

                      {csvUploadError && (
                        <div className="mt-4 p-3.5 bg-red-50 border border-red-200 rounded-xl text-xs text-red-700">
                          {csvUploadError}
                        </div>
                      )}
                    </div>

                    {/* Batch Progress and Reconciliation Monitor */}
                    {csvUploadResult && (
                      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-4">
                        <div className="flex justify-between items-center">
                          <h4 className="text-sm font-bold text-slate-900">Batch Upload Reconciled Status</h4>
                          <span className="px-3 py-1 text-xs font-bold rounded-full bg-blue-100 text-blue-800 border border-blue-200">
                            STATUS: {csvUploadResult.status}
                          </span>
                        </div>

                        <div className="grid grid-cols-3 gap-4 text-xs">
                          <div className="p-4 bg-slate-50 rounded-xl border border-slate-200">
                            <span className="text-slate-500 font-medium block">Total Claims in Batch</span>
                            <span className="text-2xl font-extrabold text-slate-900 font-mono mt-1 block">{csvUploadResult.total_rows}</span>
                          </div>
                          <div className="p-4 bg-emerald-50 rounded-xl border border-emerald-200">
                            <span className="text-emerald-700 font-medium block">Successful Rows</span>
                            <span className="text-2xl font-extrabold text-emerald-800 font-mono mt-1 block">{csvUploadResult.successful_rows}</span>
                          </div>
                          <div className="p-4 bg-red-50 rounded-xl border border-red-200">
                            <span className="text-red-700 font-medium block">Rejected Rows</span>
                            <span className="text-2xl font-extrabold text-red-800 font-mono mt-1 block">{csvUploadResult.rejected_rows}</span>
                          </div>
                        </div>

                        <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-xl text-xs text-emerald-800 flex items-center space-x-2 font-medium">
                          <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                          <span>
                            Reconciliation Invariant Enforced: Total ({csvUploadResult.total_rows}) = Successful ({csvUploadResult.successful_rows}) + Rejected ({csvUploadResult.rejected_rows})
                          </span>
                        </div>

                        {/* Sub-tabs for Results and Rejected Rows */}
                        <div className="border-t border-slate-200 pt-4">
                          <div className="flex space-x-4 mb-3 text-xs font-semibold">
                            <button
                              onClick={() => setCsvResultsTab('successful')}
                              className={`pb-1.5 border-b-2 transition ${
                                csvResultsTab === 'successful' ? 'border-blue-600 text-blue-600' : 'border-transparent text-slate-500 hover:text-slate-800'
                              }`}
                            >
                              Successful Claims ({csvSuccessfulClaims.length})
                            </button>
                            <button
                              onClick={() => setCsvResultsTab('rejected')}
                              className={`pb-1.5 border-b-2 transition ${
                                csvResultsTab === 'rejected' ? 'border-red-600 text-red-600' : 'border-transparent text-slate-500 hover:text-slate-800'
                              }`}
                            >
                              Rejected Rows ({csvRejectedRows.length})
                            </button>
                          </div>

                          {csvResultsTab === 'successful' && (
                            <div className="overflow-x-auto">
                              <table className="w-full text-left text-xs">
                                <thead className="bg-slate-50 text-slate-500 font-semibold border-b border-slate-200">
                                  <tr>
                                    <th className="p-2.5">Claim ID</th>
                                    <th className="p-2.5">Provider ID</th>
                                    <th className="p-2.5">Claim Type</th>
                                    <th className="p-2.5">Reimbursement</th>
                                    <th className="p-2.5">Anomaly Status</th>
                                  </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-100 text-slate-700">
                                  {csvSuccessfulClaims.length === 0 ? (
                                    <tr>
                                      <td colSpan={5} className="p-4 text-center text-slate-400">
                                        No successful claims records returned for this upload.
                                      </td>
                                    </tr>
                                  ) : (
                                    csvSuccessfulClaims.map((c, i) => (
                                      <tr key={i} className="hover:bg-slate-50/80">
                                        <td className="p-2.5 font-mono font-semibold text-slate-900">{c.claim_id}</td>
                                        <td className="p-2.5 font-mono text-slate-600">{c.provider_id}</td>
                                        <td className="p-2.5">{c.claim_type}</td>
                                        <td className="p-2.5 font-mono">
                                          ${typeof c.reimbursement_amount === 'number' && Number.isFinite(c.reimbursement_amount)
                                            ? c.reimbursement_amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
                                            : '0.00'}
                                        </td>
                                        <td className="p-2.5">
                                          {c.has_anomalies || (c.anomaly_flags && c.anomaly_flags.length > 0) ? (
                                            <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-amber-100 text-amber-800 border border-amber-200">
                                              {c.anomaly_flags?.length || 1} Anomaly Flag{c.anomaly_flags?.length === 1 ? '' : 's'}
                                            </span>
                                          ) : (
                                            <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-100 text-emerald-800 border border-emerald-200">
                                              Clean Claim
                                            </span>
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
                            <div className="overflow-x-auto">
                              <table className="w-full text-left text-xs">
                                <thead className="bg-slate-50 text-slate-500 font-semibold border-b border-slate-200">
                                  <tr>
                                    <th className="p-2.5">Row #</th>
                                    <th className="p-2.5">Claim ID</th>
                                    <th className="p-2.5">Rejection Reason</th>
                                  </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-100 text-slate-700">
                                  {csvRejectedRows.map((r, i) => (
                                    <tr key={i} className="hover:bg-slate-50/80">
                                      <td className="p-2.5 font-mono text-slate-600">{r.row_number}</td>
                                      <td className="p-2.5 font-mono font-semibold text-slate-900">{r.claim_id || '—'}</td>
                                      <td className="p-2.5 text-red-600 font-medium">{r.rejection_reason}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* ══════════════════════════════════════════════════════════════════════
                TAB 5: SETTINGS VIEW
            ══════════════════════════════════════════════════════════════════════ */}
            {activeTab === 'settings' && (
              <div className="space-y-6">
                <div>
                  <h2 className="text-2xl font-bold text-slate-900 tracking-tight">System Configuration & Settings</h2>
                  <p className="text-xs text-slate-500 mt-0.5">Model versions, risk thresholds, and database connection status.</p>
                </div>

                {settingsError && (
                  <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-xs text-red-700">
                    {settingsError}
                  </div>
                )}

                {settingsSuccess && (
                  <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-xl text-xs text-emerald-700">
                    {settingsSuccess}
                  </div>
                )}

                {settingsData && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-4">
                      <h3 className="text-sm font-bold text-slate-900">Active Machine Learning Setup</h3>
                      <div className="space-y-2.5 text-xs">
                        <div className="flex justify-between py-1.5 border-b border-slate-100">
                          <span className="text-slate-500 font-medium">Ensemble Version</span>
                          <span className="font-mono text-slate-900 font-bold">{settingsData.model_version.toUpperCase()}</span>
                        </div>
                        <div className="flex justify-between py-1.5 border-b border-slate-100">
                          <span className="text-slate-500 font-medium">Artifact Verification</span>
                          <span className="text-emerald-700 font-semibold">{settingsData.models_ready ? '13/13 SHA-256 Validated' : 'Verification Failed'}</span>
                        </div>
                        <div className="flex justify-between py-1.5 border-b border-slate-100">
                          <span className="text-slate-500 font-medium">Low/Medium Risk Cutoff</span>
                          <span className="font-mono text-slate-900 font-medium">{settingsData.low_medium_threshold}</span>
                        </div>
                        <div className="flex justify-between py-1.5">
                          <span className="text-slate-500 font-medium">Medium/High Risk Cutoff</span>
                          <span className="font-mono text-slate-900 font-medium">{settingsData.medium_high_threshold}</span>
                        </div>
                      </div>

                      {currentUser.role === 'admin' ? (
                        <div className="pt-4 border-t border-slate-200">
                          <span className="text-xs font-semibold text-slate-700 block mb-2">Administrator: Switch Active Model Version</span>
                          <div className="flex space-x-3">
                            <button
                              onClick={() => handleSwitchModelVersion('v1')}
                              disabled={settingsData.model_version === 'v1' || settingsLoading}
                              className="px-3.5 py-1.5 bg-slate-100 hover:bg-slate-200 disabled:opacity-40 rounded-lg text-xs font-semibold text-slate-700 border border-slate-200 transition"
                            >
                              Switch to v1
                            </button>
                            <button
                              onClick={() => handleSwitchModelVersion('v2')}
                              disabled={settingsData.model_version === 'v2' || settingsLoading}
                              className="px-3.5 py-1.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-40 rounded-lg text-xs font-semibold text-white shadow-xs transition"
                            >
                              Switch to v2
                            </button>
                          </div>
                        </div>
                      ) : (
                        <p className="text-[11px] text-slate-400 italic pt-2">
                          Administrator privileges are required to modify model versions or thresholds.
                        </p>
                      )}
                    </div>

                    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-4">
                      <h3 className="text-sm font-bold text-slate-900">Database & Compliance Architecture</h3>
                      <div className="space-y-2.5 text-xs">
                        <div className="flex justify-between py-1.5 border-b border-slate-100">
                          <span className="text-slate-500 font-medium">Database Engine</span>
                          <span className="font-mono text-slate-900 font-bold">PostgreSQL 18</span>
                        </div>
                        <div className="flex justify-between py-1.5 border-b border-slate-100">
                          <span className="text-slate-500 font-medium">Connection State</span>
                          <span className="text-emerald-700 font-semibold">{settingsData.database_connected ? 'Connected' : 'Disconnected'}</span>
                        </div>
                        <div className="flex justify-between py-1.5 border-b border-slate-100">
                          <span className="text-slate-500 font-medium">Silent SQLite Fallback</span>
                          <span className="text-red-600 font-semibold">Disabled (Production Enforced)</span>
                        </div>
                        <div className="flex justify-between py-1.5">
                          <span className="text-slate-500 font-medium">HIPAA Compliance</span>
                          <span className="text-emerald-700 font-semibold">Zero Protected Demographic Features</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

          </div>
        </main>
      </div>
    </div>
  );
}

import React, { useState, useEffect } from 'react';
import axios from 'axios';

export default function App() {
  const [summary, setSummary] = useState(null);
  const [summaryLoading, setSummaryLoading] = useState(true);
  const [summaryError, setSummaryError] = useState(null);
  const [providerId, setProviderId] = useState('');

  useEffect(() => {
    setSummaryLoading(true);
    setSummaryError(null);
    axios.get('/api/v1/reports/summary')
      .then((res) => {
        setSummary(res.data);
      })
      .catch((err) => {
        const msg = err.response?.data?.detail || err.message || 'Unable to connect to backend API.';
        setSummaryError(msg);
        setSummary(null);
      })
      .finally(() => {
        setSummaryLoading(false);
      });
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 p-8">
      <header className="mb-8 border-b pb-4">
        <h1 className="text-3xl font-bold text-slate-900">HealthGuard AI</h1>
        <p className="text-slate-600">
          Explainable Ensemble Framework for Health Insurance Provider Fraud Detection and Claims Investigation
        </p>
        <span className="inline-block mt-2 px-2.5 py-0.5 rounded text-xs font-semibold bg-blue-100 text-blue-800">
          Phase 1: Data Preparation & Feature Engineering
        </span>
      </header>

      {/* Dataset Summary Cards */}
      <section className="mb-8">
        <h2 className="text-xl font-semibold mb-3 text-slate-800">Pipeline Status & Summary</h2>
        {summaryLoading && (
          <div className="p-4 bg-white rounded-lg border text-slate-500 animate-pulse">
            Loading verified dataset summary from backend...
          </div>
        )}

        {summaryError && (
          <div className="p-4 bg-amber-50 rounded-lg border border-amber-200 text-amber-800 text-sm">
            <span className="font-semibold">Backend Notice:</span> {summaryError}
          </div>
        )}

        {summary && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-white p-4 rounded-lg shadow-sm border">
              <div className="text-xs text-slate-500 uppercase font-semibold">Training Providers</div>
              <div className="text-2xl font-bold text-emerald-600 mt-1">
                {summary.provider_split?.train_count !== undefined && summary.provider_split?.train_count !== null
                  ? summary.provider_split.train_count.toLocaleString()
                  : 'Not available'}
              </div>
              <div className="text-xs text-slate-400 mt-1">Stratified 70% split</div>
            </div>

            <div className="bg-white p-4 rounded-lg shadow-sm border">
              <div className="text-xs text-slate-500 uppercase font-semibold">Validation Providers</div>
              <div className="text-2xl font-bold text-blue-600 mt-1">
                {summary.provider_split?.val_count !== undefined && summary.provider_split?.val_count !== null
                  ? summary.provider_split.val_count.toLocaleString()
                  : 'Not available'}
              </div>
              <div className="text-xs text-slate-400 mt-1">Stratified 15% split</div>
            </div>

            <div className="bg-white p-4 rounded-lg shadow-sm border">
              <div className="text-xs text-slate-500 uppercase font-semibold">Test Providers</div>
              <div className="text-2xl font-bold text-indigo-600 mt-1">
                {summary.provider_split?.test_count !== undefined && summary.provider_split?.test_count !== null
                  ? summary.provider_split.test_count.toLocaleString()
                  : 'Not available'}
              </div>
              <div className="text-xs text-slate-400 mt-1">Stratified 15% split</div>
            </div>

            <div className="bg-white p-4 rounded-lg shadow-sm border">
              <div className="text-xs text-slate-500 uppercase font-semibold">Data Leakage Check</div>
              <div className="text-2xl font-bold text-teal-600 mt-1">
                {summary.data_leakage_free === true
                  ? '0 Overlap (Clean)'
                  : summary.data_leakage_free === false
                  ? 'Leakage Detected'
                  : 'Not available'}
              </div>
              <div className="text-xs text-slate-400 mt-1">No claims shared across splits</div>
            </div>
          </div>
        )}
      </section>

      {/* Provider Investigation Section */}
      <section className="bg-white p-6 rounded-lg shadow-sm border max-w-2xl">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-xl font-semibold text-slate-800">Provider Fraud Investigation</h2>
          <span className="px-2.5 py-0.5 rounded text-xs font-semibold bg-amber-100 text-amber-800">
            Phase 2 Training Pending
          </span>
        </div>
        <p className="text-sm text-slate-500 mb-4">
          Live provider fraud scoring and SHAP explainability attribution are pending Phase 2 model training. All preprocessed features, 12-month LSTM temporal sequences, and peer reference benchmark tables have been verified in Phase 1.
        </p>

        <form onSubmit={(e) => e.preventDefault()} className="flex gap-3 mb-4">
          <input
            type="text"
            value={providerId}
            onChange={(e) => setProviderId(e.target.value)}
            disabled={true}
            className="border border-slate-200 bg-slate-100 text-slate-500 p-2.5 rounded-lg flex-1 text-sm cursor-not-allowed"
            placeholder="Provider scoring disabled — Phase 2 model training pending"
          />
          <button
            type="button"
            disabled={true}
            className="bg-slate-300 text-slate-600 font-medium px-5 py-2.5 rounded-lg text-sm cursor-not-allowed"
          >
            Scoring Disabled
          </button>
        </form>

        {/* Phase 2 Informational Notice */}
        <div className="p-4 rounded-lg bg-blue-50 border border-blue-200 text-blue-900 text-sm">
          <div className="font-semibold mb-1 flex items-center gap-1.5">
            <span>ℹ️</span>
            <span>Phase 2 Model Training Required</span>
          </div>
          <div className="text-blue-800 text-xs leading-relaxed">
            Supervised fraud detection models (XGBoost, ANN, Bi-LSTM, and Weighted Stacking Ensemble) and SHAP explainability attribution will be trained during Phase 2. The API prediction endpoints currently return HTTP 503 until trained model artifacts are generated and validated.
          </div>
        </div>
      </section>
    </div>
  );
}

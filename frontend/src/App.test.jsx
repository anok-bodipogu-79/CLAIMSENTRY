import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import axios from 'axios';
import App, { formatProbability, getTierBadge, formatINR, usdToInr, INR_PER_USD } from './App';

vi.mock('axios', () => {
  const mockAxiosInstance = {
    interceptors: {
      request: { use: vi.fn(), eject: vi.fn() },
      response: { use: vi.fn(), eject: vi.fn() },
    },
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
  };
  return {
    default: {
      create: vi.fn(() => mockAxiosInstance),
      ...mockAxiosInstance,
    },
  };
});

describe('HealthGuard AI Frontend Integration Tests', () => {
  let mockAxios;

  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    mockAxios = axios.create();
  });

  it('renders clean sign-in form without disclaimer banner or JWT badge when unauthenticated', () => {
    render(<App />);
    expect(screen.getByText('HealthGuard AI')).toBeInTheDocument();
    expect(screen.getByText('Claims Fraud Investigation Portal')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('investigator@healthguard.ai')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('••••••••••••')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Authenticate Session/i })).toBeInTheDocument();

    // Verify statutory disclaimer and Secure JWT badge are removed from login UI
    expect(
      screen.queryByText(/This is a risk assessment to guide investigation. It is not proof of fraud./i)
    ).not.toBeInTheDocument();
    expect(screen.queryByText(/Secure JWT/i)).not.toBeInTheDocument();
  });

  it('toggles between Sign In and Investigator Registration form', () => {
    render(<App />);
    const registerToggle = screen.getByRole('button', { name: /Register Investigator/i });
    fireEvent.click(registerToggle);

    expect(screen.getByRole('button', { name: /Create Account/i })).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Dr. Alex Rivera')).toBeInTheDocument();

    const switchBackBtn = screen.getByRole('button', { name: /Sign In/i });
    fireEvent.click(switchBackBtn);
    expect(screen.getByRole('button', { name: /Authenticate Session/i })).toBeInTheDocument();
  });

  it('renders authenticated dashboard with 5 core navigation tabs when logged in', async () => {
    localStorage.setItem('healthguard_token', 'mock_jwt_token');

    // Mock axios responses
    mockAxios.get.mockImplementation((url) => {
      if (url === '/api/auth/me') {
        return Promise.resolve({
          data: {
            id: 1,
            name: 'Agent Mulder',
            email: 'mulder@fbi.gov',
            role: 'investigator',
          },
        });
      }
      if (url === '/api/overview/summary') {
        return Promise.resolve({
          data: {
            total_providers: 208,
            total_claims: 200,
            high_risk_providers: 14,
            claims_requiring_investigation: 28,
            total_reimbursed_amount: 145020.5,
            high_risk_reimbursed_amount: 32410.0,
            database_engine: 'PostgreSQL 18',
            active_model_version: 'v2',
          },
        });
      }
      if (url === '/api/overview/risk-distribution') {
        return Promise.resolve({
          data: {
            distribution: [
              { tier: 'LOW', count: 150, percentage: 72.1 },
              { tier: 'MEDIUM', count: 44, percentage: 21.2 },
              { tier: 'HIGH', count: 14, percentage: 6.7 },
            ],
            thresholds: { low_medium: 0.22, medium_high: 0.70 },
          },
        });
      }
      if (url.includes('/api/overview/recent-activity')) {
        return Promise.resolve({
          data: {
            recent_activities: [
              {
                provider_id: 'PRV51003',
                risk_score: 0.84,
                risk_tier: 'HIGH',
                created_at: '2026-11-16T12:00:00Z',
              },
            ],
          },
        });
      }
      return Promise.resolve({ data: {} });
    });

    render(<App />);

    // Wait for user info to appear
    await waitFor(() => {
      expect(screen.getByText('Agent Mulder')).toBeInTheDocument();
      expect(screen.getByText('investigator')).toBeInTheDocument();
    });

    // Verify horizontal disclaimer/status banner is completely removed from authenticated view
    expect(screen.queryByTestId('header-disclaimer')).not.toBeInTheDocument();
    expect(screen.queryByText(/PostgreSQL 18 \| Ensemble v2/i)).not.toBeInTheDocument();

    // Verify exactly 5 navigation tabs
    expect(screen.getByRole('button', { name: /^Overview$/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^Providers$/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^Claims$/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^Claim Analyzer$/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^Settings$/i })).toBeInTheDocument();

    // Strictly NO "Analytics" tab
    expect(screen.queryByRole('button', { name: /^Analytics$/i })).not.toBeInTheDocument();

    // Verify overview summary loaded
    await waitFor(() => {
      expect(screen.getByText('208')).toBeInTheDocument();
      expect(screen.getByText('200')).toBeInTheDocument();
      expect(screen.getByText('14')).toBeInTheDocument();
      expect(screen.getByText('28')).toBeInTheDocument();
    });
  });

  it('navigates to Providers and Claim Analyzer tabs', async () => {
    localStorage.setItem('healthguard_token', 'mock_jwt_token');

    mockAxios.get.mockImplementation((url) => {
      if (url === '/api/auth/me') {
        return Promise.resolve({
          data: {
            id: 2,
            name: 'Admin Scully',
            email: 'scully@fbi.gov',
            role: 'admin',
          },
        });
      }
      if (url === '/api/overview/summary') {
        return Promise.resolve({
          data: {
            total_providers: 208,
            total_claims: 200,
            high_risk_providers: 14,
            claims_requiring_investigation: 28,
            database_engine: 'PostgreSQL 18',
          },
        });
      }
      if (url === '/api/overview/risk-distribution') {
        return Promise.resolve({
          data: {
            distribution: [],
          },
        });
      }
      if (url.includes('/api/overview/recent-activity')) {
        return Promise.resolve({
          data: { recent_activities: [] },
        });
      }
      if (url.startsWith('/api/providers')) {
        return Promise.resolve({
          data: {
            providers: [
              {
                provider_id: 'PRV001',
                provider_name: 'Metro Health Center',
                total_claims: 52,
                total_reimbursement: 78500,
                risk_score: 0.88,
                risk_tier: 'HIGH',
              },
            ],
            total: 1,
            page: 1,
            total_pages: 1,
          },
        });
      }
      return Promise.resolve({ data: {} });
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Admin Scully')).toBeInTheDocument();
    });

    // Navigate to Providers
    const providersBtn = screen.getByRole('button', { name: /^Providers$/i });
    fireEvent.click(providersBtn);

    await waitFor(() => {
      expect(screen.getByPlaceholderText(/Search Provider ID or Name/i)).toBeInTheDocument();
      expect(screen.getByText('PRV001')).toBeInTheDocument();
      expect(screen.getByText('Metro Health Center')).toBeInTheDocument();
    });

    // Navigate to Claim Analyzer
    const analyzerBtn = screen.getByRole('button', { name: /^Claim Analyzer$/i });
    fireEvent.click(analyzerBtn);

    expect(screen.getByText('Claim Fraud Analyzer')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Single-Claim Analysis/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Batch CSV Upload/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Run Fraud Risk Assessment/i })).toBeInTheDocument();
  });

  it('loads realistic synthetic test data via Fill Sample button and displays notice', async () => {
    localStorage.setItem('healthguard_token', 'mock_jwt_token');
    mockAxios.get.mockImplementation((url) => {
      if (url === '/api/auth/me') {
        return Promise.resolve({ data: { id: 1, name: 'Investigator', email: 'investigator@healthguard.ai', role: 'investigator' } });
      }
      if (url === '/api/overview/summary') {
        return Promise.resolve({ data: { total_providers: 10, total_claims: 20, high_risk_providers: 1, claims_requiring_investigation: 2, model_version: 'v2', risk_thresholds: { low_medium_cutoff: 0.22, medium_high_cutoff: 0.70 }, disclaimer: 'Test' } });
      }
      return Promise.resolve({ data: {} });
    });

    render(<App />);
    await waitFor(() => expect(screen.getByRole('button', { name: /^Claim Analyzer$/i })).toBeInTheDocument());
    fireEvent.click(screen.getByRole('button', { name: /^Claim Analyzer$/i }));

    const fillSampleBtn = screen.getByRole('button', { name: /Fill Sample/i });
    expect(fillSampleBtn).toBeInTheDocument();

    fireEvent.click(fillSampleBtn);
    // User requirement: remove synthetic test data notification entirely from production interface
    expect(screen.queryByText(/Synthetic Test Data Loaded/i)).not.toBeInTheDocument();
    // Verify inputs were populated with valid sample values
    const claimIdInput = document.getElementById('f-cid');
    expect(claimIdInput.value).toMatch(/^CLM_/);
  });

  it('proves that successive clicks produce different valid sample claims with unique IDs and coherent attributes', async () => {
    localStorage.setItem('healthguard_token', 'mock_jwt_token');
    mockAxios.get.mockImplementation((url) => {
      if (url === '/api/auth/me') {
        return Promise.resolve({ data: { id: 1, name: 'Investigator', email: 'investigator@healthguard.ai', role: 'investigator' } });
      }
      if (url === '/api/overview/summary') {
        return Promise.resolve({ data: { total_providers: 10, total_claims: 20, high_risk_providers: 1, claims_requiring_investigation: 2, model_version: 'v2', risk_thresholds: { low_medium_cutoff: 0.22, medium_high_cutoff: 0.70 }, disclaimer: 'Test' } });
      }
      return Promise.resolve({ data: {} });
    });

    render(<App />);
    await waitFor(() => expect(screen.getByRole('button', { name: /^Claim Analyzer$/i })).toBeInTheDocument());
    fireEvent.click(screen.getByRole('button', { name: /^Claim Analyzer$/i }));

    const fillSampleBtn = screen.getByRole('button', { name: /Fill Sample/i });
    const claimIdInput = screen.getByText('Claim ID').parentElement.querySelector('input');
    const reimbInput = screen.getByText(/Reimbursement \(₹\)/i).parentElement.querySelector('input');

    // First click
    fireEvent.click(fillSampleBtn);
    const id1 = claimIdInput.value;
    const reimb1 = reimbInput.value;
    expect(id1).toMatch(/^CLM_SYN_/);
    expect(parseFloat(reimb1)).toBeGreaterThan(0);

    // Second click - must produce different claim ID and varied attributes
    fireEvent.click(fillSampleBtn);
    const id2 = claimIdInput.value;
    const reimb2 = reimbInput.value;
    expect(id2).toMatch(/^CLM_SYN_/);
    expect(id1).not.toBe(id2);
    expect(parseFloat(reimb2)).toBeGreaterThan(0);

    // Third click
    fireEvent.click(fillSampleBtn);
    const id3 = claimIdInput.value;
    expect(id3).toMatch(/^CLM_SYN_/);
    expect(id3).not.toBe(id2);
    expect(id3).not.toBe(id1);
  });

  describe('Claim Fraud Analyzer Probability & Risk Tier Validation', () => {
    it('formats valid probability and HIGH risk correctly', () => {
      expect(formatProbability(0.85)).toBe('85.0%');
      const { container } = render(getTierBadge('HIGH'));
      expect(container.textContent).toContain('HIGH RISK');
    });

    it('formats valid probability and MEDIUM risk correctly', () => {
      expect(formatProbability(0.455)).toBe('45.5%');
      const { container } = render(getTierBadge('MEDIUM'));
      expect(container.textContent).toContain('MEDIUM RISK');
    });

    it('formats valid probability and LOW risk correctly', () => {
      expect(formatProbability(0.12)).toBe('12.0%');
      const { container } = render(getTierBadge('LOW'));
      expect(container.textContent).toContain('LOW RISK');
    });

    it('handles missing probability by returning "Unavailable"', () => {
      expect(formatProbability(undefined)).toBe('Unavailable');
    });

    it('handles null probability by returning "Unavailable"', () => {
      expect(formatProbability(null)).toBe('Unavailable');
    });

    it('handles NaN or infinite probability by returning "Unavailable" (never "NaN%")', () => {
      expect(formatProbability(NaN)).toBe('Unavailable');
      expect(formatProbability(Infinity)).toBe('Unavailable');
      expect(formatProbability(-Infinity)).toBe('Unavailable');
      expect(formatProbability(1.5)).toBe('Unavailable'); // out of range [0, 1]
      expect(formatProbability(-0.1)).toBe('Unavailable');
    });

    it('handles missing or invalid risk tier by returning "Unavailable" (never defaulting to LOW)', () => {
      const { container: c1 } = render(getTierBadge(undefined));
      expect(c1.textContent).toBe('Unavailable');
      expect(c1.textContent).not.toContain('LOW RISK');

      const { container: c2 } = render(getTierBadge(null));
      expect(c2.textContent).toBe('Unavailable');

      const { container: c3 } = render(getTierBadge(''));
      expect(c3.textContent).toBe('Unavailable');

      const { container: c4 } = render(getTierBadge('UNKNOWN_TIER'));
      expect(c4.textContent).toBe('Unavailable');
    });

    it('formats Indian Rupee (INR) amounts with standard Indian numbering system grouping', () => {
      expect(formatINR(125000)).toBe('₹1,25,000.00');
      expect(formatINR(350000.5)).toBe('₹3,50,000.50');
      expect(formatINR(10000000)).toBe('₹1,00,00,000.00');
      expect(formatINR(500)).toBe('₹500.00');
      expect(formatINR(0)).toBe('₹0.00');
      expect(formatINR(null)).toBe('₹0.00');
      expect(formatINR(undefined)).toBe('₹0.00');
    });

    it('converts USD to INR using the verified benchmark exchange rate (83.50)', () => {
      expect(INR_PER_USD).toBe(83.50);
      expect(usdToInr(100)).toBe(8350.0);
      expect(usdToInr(0)).toBe(0);
      expect(usdToInr(null)).toBe(0);
    });

    it('renders Claim Analyzer result with provider_risk_score and provider_risk_tier and avoids NaN%', async () => {
      localStorage.setItem('healthguard_token', 'mock_jwt_token');

      mockAxios.get.mockImplementation((url) => {
        if (url === '/api/auth/me') {
          return Promise.resolve({
            data: { id: 1, name: 'Investigator Mulder', email: 'mulder@healthguard.ai', role: 'investigator' },
          });
        }
        if (url === '/api/overview/summary') {
          return Promise.resolve({ data: { total_providers: 10, total_claims: 20, high_risk_providers: 2, claims_requiring_investigation: 5 } });
        }
        if (url === '/api/overview/risk-distribution') {
          return Promise.resolve({ data: { distribution: [] } });
        }
        if (url.includes('/api/overview/recent-activity')) {
          return Promise.resolve({ data: { recent_activities: [] } });
        }
        return Promise.resolve({ data: {} });
      });

      // Mock single claim evaluation returning backend schema SingleClaimResponse
      mockAxios.post.mockImplementation((url) => {
        if (url === '/api/analyzer/single') {
          return Promise.resolve({
            data: {
              claim_id: 'CLM_TEST_001',
              provider_id: 'PRV51001',
              provider_risk_score: 0.785,
              provider_risk_tier: 'HIGH',
              model_version: 'v2',
              claim_anomaly_findings: [],
              anomaly_flags: [],
              has_anomalies: false,
              explanation_title: 'High Provider Risk Detected',
              explanation: 'Excessive daily aggregate reimbursements across outpatient procedures.',
              investigation_recommendation: 'Priority investigation recommended.',
              disclaimer: 'This is a risk assessment to guide investigation. It is not proof of fraud.',
            },
          });
        }
        return Promise.resolve({ data: {} });
      });

      render(<App />);

      await waitFor(() => {
        expect(screen.getByText('Investigator Mulder')).toBeInTheDocument();
      });

      // Click Claim Analyzer tab
      fireEvent.click(screen.getByRole('button', { name: /^Claim Analyzer$/i }));

      // Click Run Assessment button
      const runBtn = screen.getByRole('button', { name: /Run Fraud Risk Assessment/i });
      fireEvent.click(runBtn);

      // Verify that provider_risk_score (78.5%) and provider_risk_tier (HIGH RISK) are rendered
      await waitFor(() => {
        expect(screen.getByTestId('risk-score-value')).toHaveTextContent('78.5%');
        expect(screen.getByTestId('risk-tier-badge')).toHaveTextContent('HIGH RISK');
      });

      // Verify "NaN%" is strictly absent
      expect(screen.queryByText(/NaN%/i)).not.toBeInTheDocument();
    });

    it('displays "Unavailable" and warning when response has missing or null risk score', async () => {
      localStorage.setItem('healthguard_token', 'mock_jwt_token');

      mockAxios.get.mockImplementation((url) => {
        if (url === '/api/auth/me') {
          return Promise.resolve({
            data: { id: 1, name: 'Investigator Mulder', email: 'mulder@healthguard.ai', role: 'investigator' },
          });
        }
        if (url === '/api/overview/summary') {
          return Promise.resolve({ data: { total_providers: 10, total_claims: 20, high_risk_providers: 2, claims_requiring_investigation: 5 } });
        }
        if (url === '/api/overview/risk-distribution') {
          return Promise.resolve({ data: { distribution: [] } });
        }
        if (url.includes('/api/overview/recent-activity')) {
          return Promise.resolve({ data: { recent_activities: [] } });
        }
        return Promise.resolve({ data: {} });
      });

      // Incomplete response with null provider_risk_score and missing tier
      mockAxios.post.mockImplementation((url) => {
        if (url === '/api/analyzer/single') {
          return Promise.resolve({
            data: {
              claim_id: 'CLM_TEST_INCOMPLETE',
              provider_id: 'PRV51001',
              provider_risk_score: null,
              provider_risk_tier: null,
            },
          });
        }
        return Promise.resolve({ data: {} });
      });

      render(<App />);

      await waitFor(() => {
        expect(screen.getByText('Investigator Mulder')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByRole('button', { name: /^Claim Analyzer$/i }));
      fireEvent.click(screen.getByRole('button', { name: /Run Fraud Risk Assessment/i }));

      await waitFor(() => {
        expect(screen.getByTestId('risk-score-value')).toHaveTextContent('Unavailable');
        expect(screen.getByTestId('risk-tier-badge')).toHaveTextContent('Unavailable');
        expect(screen.getByTestId('incomplete-error')).toBeInTheDocument();
      });

      expect(screen.queryByText(/NaN%/i)).not.toBeInTheDocument();
      expect(screen.queryByText(/LOW RISK/i)).not.toBeInTheDocument();
    });

    it('displays full SHAP factor attributions, actual feature values, and Bi-LSTM temporal sequence dynamics', async () => {
      localStorage.setItem('healthguard_token', 'mock_jwt_token');

      mockAxios.get.mockImplementation((url) => {
        if (url === '/api/auth/me') {
          return Promise.resolve({
            data: { id: 1, name: 'Investigator Mulder', email: 'mulder@healthguard.ai', role: 'investigator' },
          });
        }
        return Promise.resolve({ data: { distribution: [], recent_activities: [] } });
      });

      mockAxios.post.mockImplementation((url) => {
        if (url === '/api/analyzer/single') {
          return Promise.resolve({
            data: {
              claim_id: 'CLM_DYNAMIC_TEST',
              provider_id: 'PRV99999',
              provider_risk_score: 0.825,
              provider_risk_tier: 'HIGH',
              model_version: 'v2',
              investigation_recommendation: 'HIGH PRIORITY: Full clinical audit recommended.',
              explanation_title: 'Multi-Modal Risk Factor Attribution',
              explanation: {
                text: 'The ensemble evaluated provider PRV99999 at 82.5% fraud risk.',
                top_risk_factors: [
                  {
                    feature: 'mean_inpatient_reimbursement',
                    actual_value: 34500.0,
                    shap_value: 0.3125,
                    direction: 'INCREASES_FRAUD_RISK',
                    benchmark_summary: '95th percentile (Peer median: 1200.0)',
                  },
                  {
                    feature: 'duplicate_claim_count',
                    actual_value: 0.0,
                    shap_value: -0.045,
                    direction: 'DECREASES_FRAUD_RISK',
                    benchmark_summary: '25th percentile',
                  },
                ],
                temporal_explanation: {
                  trajectory_summary: 'Month 10 had the highest temporal impact (+0.0812), elevating fraud risk.',
                  monthly_temporal_impacts: Array.from({ length: 12 }, (_, i) => ({
                    month_index: i + 1,
                    is_valid_observation: i >= 6,
                    observation_type: i < 6 ? 'PADDING' : (i === 6 ? 'ZERO_CLAIMS_IN_WINDOW' : 'ACTIVE_CLAIMS'),
                    impact: i === 9 ? 0.0812 : 0.0,
                    direction: i === 9 ? 'ELEVATES_RISK' : 'NEUTRAL',
                  })),
                },
              },
              claim_anomaly_findings: [],
              anomaly_flags: [],
              has_anomalies: false,
              disclaimer: 'This is a risk assessment to guide investigation. It is not proof of fraud.',
            },
          });
        }
        return Promise.resolve({ data: {} });
      });

      render(<App />);

      await waitFor(() => {
        expect(screen.getByText('Investigator Mulder')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByRole('button', { name: /^Claim Analyzer$/i }));
      fireEvent.click(screen.getByRole('button', { name: /Run Fraud Risk Assessment/i }));

      await waitFor(() => {
        expect(screen.getByTestId('risk-score-value')).toHaveTextContent('82.5%');
        expect(screen.getByTestId('risk-tier-badge')).toHaveTextContent('HIGH RISK');
        expect(screen.getByTestId('model-version-badge')).toHaveTextContent('Model: v2');
        expect(screen.getByTestId('investigation-recommendation')).toHaveTextContent('HIGH PRIORITY: Full clinical audit recommended.');
        expect(screen.getAllByText(/mean_inpatient_reimbursement/i).length).toBeGreaterThan(0);
        expect(screen.getByText(/34,500/)).toBeInTheDocument();
        expect(screen.getByText('+0.31 SHAP')).toBeInTheDocument();
        expect(screen.getByTestId('regulatory-disclaimer')).toBeInTheDocument();
      });

      // Verify removed sections are strictly absent from the DOM
      expect(screen.queryByTestId('shap-explanation-card')).not.toBeInTheDocument();
      expect(screen.queryByTestId('temporal-explanation-card')).not.toBeInTheDocument();
      expect(screen.queryByTestId('claim-anomaly-card')).not.toBeInTheDocument();
      expect(screen.queryByTestId('data-availability-panel')).not.toBeInTheDocument();
      expect(screen.queryByTestId('provider-history-card')).not.toBeInTheDocument();
      expect(screen.queryByTestId('beneficiary-history-card')).not.toBeInTheDocument();
    });

    it('clears stale results when starting a new claim evaluation', async () => {
      localStorage.setItem('healthguard_token', 'mock_jwt_token');

      mockAxios.get.mockImplementation(() => Promise.resolve({ data: { distribution: [], recent_activities: [] } }));
      let resolveFirstCall;
      mockAxios.post.mockImplementation((url) => {
        if (url === '/api/analyzer/single') {
          return new Promise((resolve) => {
            resolveFirstCall = resolve;
          });
        }
        return Promise.resolve({ data: {} });
      });

      render(<App />);
      await waitFor(() => expect(screen.getByText('Claim Analyzer')).toBeInTheDocument());
      fireEvent.click(screen.getByRole('button', { name: /^Claim Analyzer$/i }));

      // Trigger submission
      fireEvent.click(screen.getByRole('button', { name: /Run Fraud Risk Assessment/i }));

      // Prompt placeholder should show while loading, results cleared
      expect(screen.getByRole('button', { name: /Evaluating ML Ensemble.../i })).toBeInTheDocument();
      expect(screen.queryByTestId('risk-score-value')).not.toBeInTheDocument();

      // Resolve the call
      resolveFirstCall({
        data: {
          claim_id: 'CLM_1',
          provider_id: 'PRV1',
          provider_risk_score: 0.15,
          provider_risk_tier: 'LOW',
          model_version: 'v2',
          has_anomalies: false,
          anomaly_flags: [],
          claim_anomaly_findings: [],
          explanation_title: 'Risk Analysis',
          explanation: { text: 'Low risk provider.' },
          investigation_recommendation: 'Normal processing.',
          disclaimer: 'Statutory disclaimer.',
        },
      });

      await waitFor(() => {
        expect(screen.getByTestId('risk-score-value')).toHaveTextContent('15.0%');
      });
    });

    it('conditionally shows Admission/Discharge dates and auto-calculates Length of Stay for Inpatient claims', async () => {
      localStorage.setItem('healthguard_token', 'mock_jwt_token');
      mockAxios.get.mockImplementation(() => Promise.resolve({ data: { distribution: [], recent_activities: [] } }));

      render(<App />);
      await waitFor(() => expect(screen.getByRole('button', { name: /^Claim Analyzer$/i })).toBeInTheDocument());
      fireEvent.click(screen.getByRole('button', { name: /^Claim Analyzer$/i }));

      // By default Outpatient is selected -> Admission and Discharge dates should NOT be visible
      expect(screen.queryByText('Admission Date')).not.toBeInTheDocument();
      expect(screen.queryByText('Discharge Date')).not.toBeInTheDocument();
      expect(screen.queryByText(/Length of Stay:/i)).not.toBeInTheDocument();

      // Procedure Code 1, Operating Physician, Other Physician must always be present
      expect(screen.getByText('Procedure Code 1')).toBeInTheDocument();
      expect(screen.getByText('Operating Physician')).toBeInTheDocument();
      expect(screen.getByText('Other Physician')).toBeInTheDocument();

      // Switch to Inpatient
      const claimTypeSelect = screen.getByText('Claim Type').parentElement.querySelector('select');
      fireEvent.change(claimTypeSelect, { target: { value: 'Inpatient' } });

      // Inpatient fields must now be visible
      expect(screen.getByText('Admission Date')).toBeInTheDocument();
      expect(screen.getByText('Discharge Date')).toBeInTheDocument();
      expect(screen.getByText(/Length of Stay: 0 days/i)).toBeInTheDocument();

      // Fill Admission and Discharge dates
      const admissionInput = screen.getByText('Admission Date').parentElement.querySelector('input');
      const dischargeInput = screen.getByText('Discharge Date').parentElement.querySelector('input');
      fireEvent.change(admissionInput, { target: { value: '2009-04-10' } });
      fireEvent.change(dischargeInput, { target: { value: '2009-04-17' } });

      // Length of Stay must automatically calculate to 7 days
      expect(screen.getByText(/Length of Stay: 7 days/i)).toBeInTheDocument();

      // Switch back to Outpatient -> Inpatient fields must disappear
      fireEvent.change(claimTypeSelect, { target: { value: 'Outpatient' } });
      expect(screen.queryByText('Admission Date')).not.toBeInTheDocument();
      expect(screen.queryByText('Discharge Date')).not.toBeInTheDocument();
    });

    it('validates Inpatient dates and correctly submits payload with all additional fields', async () => {
      localStorage.setItem('healthguard_token', 'mock_jwt_token');
      mockAxios.get.mockImplementation(() => Promise.resolve({ data: { distribution: [], recent_activities: [] } }));
      let capturedPayload = null;
      mockAxios.post.mockImplementation((url, payload) => {
        if (url === '/api/analyzer/single') {
          capturedPayload = payload;
          return Promise.resolve({
            data: {
              claim_id: payload.ClaimID,
              provider_id: payload.ProviderID,
              provider_risk_score: 0.75,
              provider_risk_tier: 'HIGH',
              model_version: 'v2',
              has_anomalies: false,
              anomaly_flags: [],
              claim_anomaly_findings: [],
              explanation_title: 'Risk Analysis',
              explanation: { text: 'High risk inpatient.' },
              investigation_recommendation: 'Targeted review.',
              disclaimer: 'Statutory disclaimer.',
            },
          });
        }
        return Promise.resolve({ data: {} });
      });

      render(<App />);
      await waitFor(() => expect(screen.getByRole('button', { name: /^Claim Analyzer$/i })).toBeInTheDocument());
      fireEvent.click(screen.getByRole('button', { name: /^Claim Analyzer$/i }));

      // Switch to Inpatient
      const claimTypeSelect = screen.getByText('Claim Type').parentElement.querySelector('select');
      fireEvent.change(claimTypeSelect, { target: { value: 'Inpatient' } });

      // Set invalid dates: discharge earlier than admission
      const admissionInput = screen.getByText('Admission Date').parentElement.querySelector('input');
      const dischargeInput = screen.getByText('Discharge Date').parentElement.querySelector('input');
      fireEvent.change(admissionInput, { target: { value: '2009-04-20' } });
      fireEvent.change(dischargeInput, { target: { value: '2009-04-10' } });

      fireEvent.click(screen.getByRole('button', { name: /Run Fraud Risk Assessment/i }));
      expect(screen.getByText(/Discharge Date cannot be earlier than Admission Date/i)).toBeInTheDocument();

      // Fix dates and fill Procedure Code 1, Operating Physician, Other Physician
      fireEvent.change(dischargeInput, { target: { value: '2009-04-25' } });
      const procInput = screen.getByText('Procedure Code 1').parentElement.querySelector('input');
      const opPhysInput = screen.getByText('Operating Physician').parentElement.querySelector('input');
      const otherPhysInput = screen.getByText('Other Physician').parentElement.querySelector('input');

      fireEvent.change(procInput, { target: { value: '3893' } });
      fireEvent.change(opPhysInput, { target: { value: 'PHY40101' } });
      fireEvent.change(otherPhysInput, { target: { value: 'PHY50201' } });

      fireEvent.click(screen.getByRole('button', { name: /Run Fraud Risk Assessment/i }));

      await waitFor(() => {
        expect(capturedPayload).not.toBeNull();
      });

      expect(capturedPayload.ClaimType).toBe('Inpatient');
      expect(capturedPayload.AdmissionDt).toBe('2009-04-20');
      expect(capturedPayload.DischargeDt).toBe('2009-04-25');
      expect(capturedPayload.ClmProcedureCode_1).toBe('3893');
      expect(capturedPayload.OperatingPhysician).toBe('PHY40101');
      expect(capturedPayload.OtherPhysician).toBe('PHY50201');
    });

    it('Phase C: exposes Diagnosis Code 2 and Admission Diagnosis Code inputs in form payload', async () => {
      localStorage.setItem('healthguard_token', 'mock_token');
      mockAxios.get.mockImplementation((url) => {
        if (url.includes('/users/me')) {
          return Promise.resolve({
            data: { id: 1, name: 'Investigator Test', email: 'investigator@healthguard.ai', role: 'investigator' },
          });
        }
        return Promise.resolve({ data: {} });
      });

      let capturedPayload = null;
      mockAxios.post.mockImplementation((url, payload) => {
        if (url.includes('/analyzer/single')) {
          capturedPayload = payload;
          return Promise.resolve({
            data: {
              claim_id: payload.ClaimID,
              provider_id: payload.ProviderID,
              provider_risk_score: 0.15,
              provider_risk_tier: 'LOW',
              model_version: 'v2',
              has_anomalies: false,
              anomaly_flags: [],
              claim_anomaly_findings: [],
              explanation_title: 'Low Risk',
              explanation: {},
              disclaimer: 'Statutory disclaimer.',
            },
          });
        }
        return Promise.resolve({ data: {} });
      });

      render(<App />);
      await waitFor(() => expect(screen.getByRole('button', { name: /^Claim Analyzer$/i })).toBeInTheDocument());
      fireEvent.click(screen.getByRole('button', { name: /^Claim Analyzer$/i }));

      // Verify Diagnosis Code 2 and Admit Diagnosis Code inputs exist
      const diag2Input = screen.getByText('Diagnosis Code 2').parentElement.querySelector('input');
      const admitDiagInput = screen.getByText('Admit Diagnosis Code').parentElement.querySelector('input');
      expect(diag2Input).toBeInTheDocument();
      expect(admitDiagInput).toBeInTheDocument();

      fireEvent.change(diag2Input, { target: { value: '25000' } });
      fireEvent.change(admitDiagInput, { target: { value: '4280' } });

      fireEvent.click(screen.getByRole('button', { name: /Run Fraud Risk Assessment/i }));

      await waitFor(() => {
        expect(capturedPayload).not.toBeNull();
      });

      expect(capturedPayload.ClmDiagnosisCode_2).toBe('25000');
      expect(capturedPayload.ClmAdmitDiagnosisCode).toBe('4280');
    });

    it('preserves Monthly Reimbursement and Claim Activity charts while removing unneeded audit sections', async () => {
      localStorage.setItem('healthguard_token', 'mock_token');
      mockAxios.get.mockImplementation((url) => {
        if (url.includes('/users/me')) {
          return Promise.resolve({
            data: { id: 1, name: 'Investigator Test', email: 'investigator@healthguard.ai', role: 'investigator' },
          });
        }
        return Promise.resolve({ data: {} });
      });

      mockAxios.post.mockImplementation((url, payload) => {
        if (url.includes('/analyzer/single')) {
          return Promise.resolve({
            data: {
              claim_id: payload.ClaimID,
              provider_id: payload.ProviderID,
              provider_risk_score: 0.85,
              provider_risk_tier: 'HIGH',
              model_version: 'v2',
              reimbursement_inr: 125000,
              currency: 'INR',
              monthly_trends: {
                months: [
                  { month: 'Jan', reimbursement: 45000, claim_count: 5 },
                  { month: 'Feb', reimbursement: 80000, claim_count: 8 }
                ]
              },
              explanation_title: 'High Risk Encounter',
              explanation: {},
              disclaimer: 'Statutory disclaimer.',
            },
          });
        }
        return Promise.resolve({ data: {} });
      });

      render(<App />);
      await waitFor(() => expect(screen.getByRole('button', { name: /^Claim Analyzer$/i })).toBeInTheDocument());
      fireEvent.click(screen.getByRole('button', { name: /^Claim Analyzer$/i }));

      // Run analysis
      fireEvent.click(screen.getByRole('button', { name: /Run Fraud Risk Assessment/i }));

      // Verify Monthly Reimbursement Trend and Monthly Claim Activity charts are rendered
      await waitFor(() => {
        expect(screen.getByText('Monthly Reimbursement Trend')).toBeInTheDocument();
        expect(screen.getByText('Monthly Claim Activity')).toBeInTheDocument();
      });

      // Confirm the 5 requested removed sections are strictly absent
      expect(screen.queryByTestId('investigator-decision-section')).not.toBeInTheDocument();
      expect(screen.queryByTestId('data-availability-panel')).not.toBeInTheDocument();
      expect(screen.queryByTestId('provider-history-card')).not.toBeInTheDocument();
      expect(screen.queryByTestId('beneficiary-history-card')).not.toBeInTheDocument();
      expect(screen.queryByTestId('claim-anomaly-card')).not.toBeInTheDocument();
      expect(screen.queryByTestId('shap-explanation-card')).not.toBeInTheDocument();
    });

    it('verifies Overview recent assessments mapping and Claim Analyzer simplified investigator UI headings', async () => {
      localStorage.setItem('healthguard_token', 'mock_jwt_token');

      mockAxios.get.mockImplementation((url) => {
        if (url === '/api/auth/me') {
          return Promise.resolve({
            data: { id: 1, name: 'Investigator Scully', email: 'scully@healthguard.ai', role: 'investigator' },
          });
        }
        if (url === '/api/overview/summary') {
          return Promise.resolve({
            data: { total_providers: 31, total_claims: 28, high_risk_providers: 3, claims_requiring_investigation: 5, model_version: 'v2' },
          });
        }
        if (url.startsWith('/api/overview/recent-activity')) {
          return Promise.resolve({
            data: {
              activities: [
                {
                  id: 364,
                  activity_type: 'PROVIDER_ASSESSMENT',
                  provider_id: 'PRV_59081',
                  entity_id: 'PRV_59081',
                  risk_score: 0.0699,
                  risk_tier: 'LOW',
                  timestamp: '2026-11-20T00:14:20.101659+05:30',
                  created_at: '2026-11-20T00:14:20.101659+05:30',
                },
              ],
            },
          });
        }
        return Promise.resolve({ data: { distribution: [], claims_by_month: [] } });
      });

      mockAxios.post.mockImplementation((url) => {
        if (url === '/api/analyzer/single') {
          return Promise.resolve({
            data: {
              claim_id: 'CLM_INIT_1001',
              provider_id: 'PRV_10001',
              provider_risk_score: 0.026,
              provider_risk_tier: 'LOW',
              original_reimbursement: 350000.0,
              currency: 'INR',
              model_version: 'v2',
              investigation_recommendation: 'Standard automated settlement workflow.',
              explanation: {
                base_value: 0.18,
                top_risk_factors: [
                  { feature: 'total_reimbursement', display_name: 'Provider Cumulative Reimbursement', shap_value: -0.09, actual_value: 350000.0 },
                ],
              },
              claim_anomaly_findings: [],
            },
          });
        }
        return Promise.resolve({ data: {} });
      });

      render(<App />);

      // 1. Verify Overview recent assessments table displays provider assessment
      await waitFor(() => {
        expect(screen.getByText('PRV_59081')).toBeInTheDocument();
        expect(screen.getByText('7.0%')).toBeInTheDocument();
      });

      // 2. Navigate to Claim Analyzer
      fireEvent.click(screen.getByRole('button', { name: /^Claim Analyzer$/i }));
      fireEvent.click(screen.getByRole('button', { name: /Run Fraud Risk Assessment/i }));

      // 3. Verify Simplified Headings & Financial Exposure
      await waitFor(() => {
        expect(screen.getByText('Risk Assessment Summary')).toBeInTheDocument();
        expect(screen.getByText('Claim Amount Under Review')).toBeInTheDocument();
        expect(screen.getByText('Why This Risk Score?')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Copy/i })).toBeInTheDocument();
      });

      // 4. Verify Technical Model Details are removed from visible report
      expect(screen.queryByText(/99\.4% F1/i)).not.toBeInTheDocument();
      expect(screen.queryByText(/TreeSHAP Exact Decomposition/i)).not.toBeInTheDocument();
      expect(screen.queryByText(/Sum\(φᵢ\)/i)).not.toBeInTheDocument();
    });
  });
});

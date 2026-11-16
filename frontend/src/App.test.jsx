import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import axios from 'axios';
import App, { formatProbability, getTierBadge } from './App';

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

  it('renders sign-in form with statutory disclaimer when unauthenticated', () => {
    render(<App />);
    expect(screen.getByText('HealthGuard AI')).toBeInTheDocument();
    expect(screen.getByText('Claims Fraud Investigation Portal')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('investigator@healthguard.ai')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('••••••••••••')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Authenticate Session/i })).toBeInTheDocument();

    // Check mandatory statutory disclaimer
    expect(
      screen.getByText(/This is a risk assessment to guide investigation. It is not proof of fraud./i)
    ).toBeInTheDocument();
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

    // Verify statutory disclaimer banner in authenticated view
    expect(
      screen.getByText(/This is a risk assessment to guide investigation. It is not proof of fraud./i)
    ).toBeInTheDocument();

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
        expect(screen.getByTestId('shap-explanation-card')).toBeInTheDocument();
        expect(screen.getByText('mean_inpatient_reimbursement')).toBeInTheDocument();
        expect(screen.getByText('34,500')).toBeInTheDocument();
        expect(screen.getByText('+0.3125')).toBeInTheDocument();
        expect(screen.getByTestId('temporal-explanation-card')).toBeInTheDocument();
        expect(screen.getByText(/Month 10 had the highest temporal impact/i)).toBeInTheDocument();
        expect(screen.getByTestId('claim-anomaly-card')).toHaveTextContent('No Anomalies Detected');
        expect(screen.getByTestId('regulatory-disclaimer')).toBeInTheDocument();
      });
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
  });
});

import React, { useState, useCallback } from 'react';
import PredictionPanel from './components/PredictionPanel.jsx';
import SimulationChart from './components/SimulationChart.jsx';
import DriverCard from './components/DriverCard.jsx';
import { getPredictions, getSimulation, getBacktest } from './api/client.js';

const SEASONS = Array.from({ length: 15 }, (_, i) => 2010 + i);
const ROUNDS = Array.from({ length: 24 }, (_, i) => i + 1);

export default function App() {
    const [activeTab, setActiveTab] = useState('predictions');
    const [season, setSeason] = useState(2023);
    const [round, setRound] = useState(5);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Data states
    const [predictions, setPredictions] = useState(null);
    const [simulation, setSimulation] = useState(null);
    const [backtest, setBacktest] = useState(null);

    const handlePredict = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await getPredictions(season, round);
            setPredictions(data);
            setActiveTab('predictions');
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, [season, round]);

    const handleSimulate = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await getSimulation(season, round, 5000);
            setSimulation(data);
            setActiveTab('simulation');
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, [season, round]);

    const handleBacktest = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await getBacktest([season], 1000);
            setBacktest(data);
            setActiveTab('backtest');
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, [season]);

    const currentPredictions = predictions?.predictions || simulation?.drivers || [];

    return (
        <div className="app">
            {/* ── Header ── */}
            <header className="header">
                <div className="header-inner">
                    <div className="header-brand">
                        <div className="header-logo">
                            <span>F1</span> Race Intelligence
                        </div>
                        <span className="header-badge">Engine v1.0</span>
                    </div>
                    <div className="header-status">
                        <span className="status-dot" />
                        <span>System Online</span>
                    </div>
                </div>
            </header>

            {/* ── Main Content ── */}
            <main className="main">
                {/* Controls */}
                <div className="controls">
                    <div className="control-group">
                        <label htmlFor="season-select">Season</label>
                        <select
                            id="season-select"
                            value={season}
                            onChange={e => setSeason(parseInt(e.target.value))}
                        >
                            {SEASONS.map(y => (
                                <option key={y} value={y}>{y}</option>
                            ))}
                        </select>
                    </div>

                    <div className="control-group">
                        <label htmlFor="round-select">Round</label>
                        <select
                            id="round-select"
                            value={round}
                            onChange={e => setRound(parseInt(e.target.value))}
                        >
                            {ROUNDS.map(r => (
                                <option key={r} value={r}>Round {r}</option>
                            ))}
                        </select>
                    </div>

                    <button className="btn btn-primary" onClick={handlePredict} disabled={loading}>
                        {loading && activeTab === 'predictions' ? '⟳ Predicting...' : '🏁 Predict Race'}
                    </button>

                    <button className="btn btn-secondary" onClick={handleSimulate} disabled={loading}>
                        {loading && activeTab === 'simulation' ? '⟳ Simulating...' : '🎲 Simulate'}
                    </button>

                    <button className="btn btn-secondary" onClick={handleBacktest} disabled={loading}>
                        {loading && activeTab === 'backtest' ? '⟳ Testing...' : '📊 Backtest'}
                    </button>
                </div>

                {/* Error State */}
                {error && (
                    <div className="error-banner">
                        ⚠ {error}
                    </div>
                )}

                {/* Loading State */}
                {loading && (
                    <div className="loading-container">
                        <div className="spinner" />
                        <div className="loading-text">
                            {activeTab === 'simulation'
                                ? 'Running Monte Carlo simulation (5,000 iterations)...'
                                : activeTab === 'backtest'
                                    ? 'Running historical backtesting...'
                                    : 'Generating predictions...'}
                        </div>
                    </div>
                )}

                {/* Tabs */}
                {!loading && (predictions || simulation || backtest) && (
                    <>
                        <div className="tabs">
                            <button
                                className={`tab ${activeTab === 'predictions' ? 'active' : ''}`}
                                onClick={() => setActiveTab('predictions')}
                            >
                                Predictions
                            </button>
                            <button
                                className={`tab ${activeTab === 'simulation' ? 'active' : ''}`}
                                onClick={() => setActiveTab('simulation')}
                            >
                                Simulation
                            </button>
                            <button
                                className={`tab ${activeTab === 'backtest' ? 'active' : ''}`}
                                onClick={() => setActiveTab('backtest')}
                            >
                                Backtest
                            </button>
                        </div>

                        {/* ── Predictions Tab ── */}
                        {activeTab === 'predictions' && predictions && (
                            <div className="slide-up">
                                {/* Race Header */}
                                <div style={{ marginBottom: 'var(--space-lg)' }}>
                                    <h2 style={{
                                        fontFamily: 'var(--font-heading)',
                                        fontSize: '1.6rem',
                                        fontWeight: 800,
                                    }}>
                                        {predictions.race_name || `${predictions.season_year} Round ${predictions.round_number}`}
                                    </h2>
                                    {predictions.circuit_name && (
                                        <span style={{ color: 'var(--f1-text-muted)', fontSize: '0.9rem' }}>
                                            {predictions.circuit_name}
                                        </span>
                                    )}
                                </div>

                                {/* Charts */}
                                <PredictionPanel predictions={predictions.predictions} raceName={predictions.race_name} />

                                {/* Driver Cards Grid */}
                                <h3 className="section-title" style={{ marginTop: 'var(--space-xl)' }}>
                                    Driver Breakdown
                                </h3>
                                <div className="grid-3">
                                    {predictions.predictions
                                        .sort((a, b) => a.expected_position - b.expected_position)
                                        .map((driver, idx) => (
                                            <DriverCard key={driver.driver_id} driver={driver} rank={idx + 1} />
                                        ))}
                                </div>
                            </div>
                        )}

                        {/* ── Simulation Tab ── */}
                        {activeTab === 'simulation' && simulation && (
                            <div className="slide-up">
                                <div style={{ marginBottom: 'var(--space-lg)' }}>
                                    <h2 style={{
                                        fontFamily: 'var(--font-heading)',
                                        fontSize: '1.6rem',
                                        fontWeight: 800,
                                    }}>
                                        Monte Carlo Simulation
                                    </h2>
                                    <span style={{ color: 'var(--f1-text-muted)', fontSize: '0.9rem' }}>
                                        {simulation.race_name || `${simulation.season_year} Round ${simulation.round_number}`}
                                    </span>
                                </div>

                                <SimulationChart simulationData={simulation} />

                                {/* Driver Cards for Simulation */}
                                <h3 className="section-title" style={{ marginTop: 'var(--space-xl)' }}>
                                    Driver Results
                                </h3>
                                <div className="grid-3">
                                    {simulation.drivers
                                        .sort((a, b) => a.expected_position - b.expected_position)
                                        .slice(0, 12)
                                        .map((driver, idx) => (
                                            <DriverCard
                                                key={driver.driver_id}
                                                driver={{
                                                    ...driver,
                                                    driver_name: driver.driver_name || driver.name,
                                                    dnf_probability: driver.dnf_probability || driver.dnf_rate || 0,
                                                }}
                                                rank={idx + 1}
                                            />
                                        ))}
                                </div>
                            </div>
                        )}

                        {/* ── Backtest Tab ── */}
                        {activeTab === 'backtest' && backtest && (
                            <div className="slide-up">
                                <h2 style={{
                                    fontFamily: 'var(--font-heading)',
                                    fontSize: '1.6rem',
                                    fontWeight: 800,
                                    marginBottom: 'var(--space-lg)',
                                }}>
                                    Backtesting Results
                                </h2>

                                {/* Overall Metrics */}
                                <div className="grid-3" style={{ marginBottom: 'var(--space-xl)' }}>
                                    <div className="card stat-card">
                                        <div className="stat-value red">
                                            {backtest.overall_metrics?.log_loss?.toFixed(4) || '—'}
                                        </div>
                                        <div className="stat-label">Log Loss</div>
                                    </div>
                                    <div className="card stat-card">
                                        <div className="stat-value amber">
                                            {backtest.overall_metrics?.brier_score?.toFixed(4) || '—'}
                                        </div>
                                        <div className="stat-label">Brier Score</div>
                                    </div>
                                    <div className="card stat-card">
                                        <div className="stat-value green">
                                            {backtest.overall_metrics?.top3_accuracy
                                                ? `${(backtest.overall_metrics.top3_accuracy * 100).toFixed(1)}%`
                                                : '—'}
                                        </div>
                                        <div className="stat-label">Top-3 Accuracy</div>
                                    </div>
                                </div>
                                <div className="grid-2">
                                    <div className="card stat-card">
                                        <div className="stat-value blue">
                                            {backtest.overall_metrics?.rank_correlation?.toFixed(4) || '—'}
                                        </div>
                                        <div className="stat-label">Rank Correlation (Spearman)</div>
                                    </div>
                                    <div className="card stat-card">
                                        <div className="stat-value" style={{ color: 'var(--f1-text)' }}>
                                            {backtest.per_season?.reduce((sum, s) => sum + s.n_races, 0) || '—'}
                                        </div>
                                        <div className="stat-label">Races Evaluated</div>
                                    </div>
                                </div>

                                {/* Per-Season Table */}
                                {backtest.per_season?.length > 0 && (
                                    <div className="card" style={{ marginTop: 'var(--space-lg)' }}>
                                        <h3 className="section-title">Per-Season Breakdown</h3>
                                        <table className="predictions-table">
                                            <thead>
                                                <tr>
                                                    <th>Season</th>
                                                    <th>Races</th>
                                                    <th>Log Loss</th>
                                                    <th>Brier</th>
                                                    <th>Top-3 Acc.</th>
                                                    <th>Rank Corr.</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {backtest.per_season.map(s => (
                                                    <tr key={s.season}>
                                                        <td style={{ fontWeight: 600 }}>{s.season}</td>
                                                        <td>{s.n_races}</td>
                                                        <td>{s.log_loss.toFixed(4)}</td>
                                                        <td>{s.brier_score.toFixed(4)}</td>
                                                        <td>{(s.top3_accuracy * 100).toFixed(1)}%</td>
                                                        <td>{s.rank_correlation.toFixed(4)}</td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                )}
                            </div>
                        )}
                    </>
                )}

                {/* ── Empty State ── */}
                {!loading && !predictions && !simulation && !backtest && (
                    <div className="empty-state slide-up" style={{ marginTop: '80px' }}>
                        <div style={{ fontSize: '4rem', marginBottom: '16px' }}>🏎️</div>
                        <h2 style={{
                            fontFamily: 'var(--font-heading)',
                            fontSize: '1.5rem',
                            fontWeight: 700,
                            marginBottom: '8px',
                            color: 'var(--f1-text)',
                        }}>
                            F1 Race Intelligence Engine
                        </h2>
                        <p style={{ maxWidth: '500px', margin: '0 auto', lineHeight: 1.6 }}>
                            Select a season and round, then click <strong>Predict Race</strong> to generate
                            AI-powered race outcome predictions, or <strong>Simulate</strong> to run
                            Monte Carlo race simulations.
                        </p>
                    </div>
                )}
            </main>

            {/* ── Footer ── */}
            <footer style={{
                textAlign: 'center',
                padding: 'var(--space-lg)',
                color: 'var(--f1-text-muted)',
                fontSize: '0.75rem',
                borderTop: '1px solid var(--f1-border)',
            }}>
                F1 Race Intelligence Engine — Built with FastAPI, XGBoost, Monte Carlo Simulation & React
            </footer>
        </div>
    );
}

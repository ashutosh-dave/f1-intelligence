import React from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, Cell, ErrorBar,
    ComposedChart, Scatter, Line,
} from 'recharts';

const COLORS = [
    '#E10600', '#FF6B35', '#FFD700', '#00E676',
    '#40C4FF', '#AB47BC', '#FF7043', '#26C6DA',
    '#EF5350', '#66BB6A', '#FFA726', '#42A5F5',
    '#EC407A', '#26A69A', '#FFCA28', '#7E57C2',
];

/**
 * SimulationChart — visualizes finishing position distributions
 * from Monte Carlo simulation results.
 */
export default function SimulationChart({ simulationData }) {
    if (!simulationData || !simulationData.drivers?.length) {
        return (
            <div className="empty-state">
                <div className="icon">🎲</div>
                <p>No simulation data. Run a simulation first.</p>
            </div>
        );
    }

    const drivers = [...simulationData.drivers]
        .sort((a, b) => a.expected_position - b.expected_position)
        .slice(0, 15);

    // Expected position with uncertainty bars
    const positionData = drivers.map(d => ({
        name: d.driver_code || d.driver_name?.split(' ').pop() || `D${d.driver_id}`,
        expected: +d.expected_position.toFixed(1),
        std: d.position_std ? +d.position_std.toFixed(1) : 0,
        fullName: d.driver_name || d.name,
    }));

    // Position distribution heatmap data
    const distributionData = [];
    drivers.slice(0, 10).forEach(driver => {
        const dist = driver.position_distribution || {};
        Object.entries(dist).forEach(([pos, prob]) => {
            distributionData.push({
                driver: driver.driver_code || driver.driver_name?.split(' ').pop() || `D${driver.driver_id}`,
                position: parseInt(pos),
                probability: +(prob * 100).toFixed(1),
            });
        });
    });

    const CustomTooltip = ({ active, payload }) => {
        if (!active || !payload?.length) return null;
        const data = payload[0]?.payload;
        return (
            <div style={{
                background: 'var(--f1-card)',
                border: '1px solid var(--f1-border)',
                borderRadius: 'var(--radius-sm)',
                padding: '10px 14px',
                fontSize: '0.85rem',
            }}>
                <div style={{ fontWeight: 600, marginBottom: '4px' }}>
                    {data?.fullName || data?.name}
                </div>
                <div>Expected: P{data?.expected}</div>
                <div style={{ color: 'var(--f1-text-muted)' }}>
                    Uncertainty: ±{data?.std}
                </div>
            </div>
        );
    };

    return (
        <div className="fade-in">
            {/* Summary stats */}
            <div className="grid-3" style={{ marginBottom: 'var(--space-lg)' }}>
                <div className="card stat-card">
                    <div className="stat-value blue">{simulationData.n_simulations?.toLocaleString()}</div>
                    <div className="stat-label">Simulations</div>
                </div>
                <div className="card stat-card">
                    <div className="stat-value amber">{simulationData.avg_safety_cars_per_race?.toFixed(1)}</div>
                    <div className="stat-label">Avg Safety Cars</div>
                </div>
                <div className="card stat-card">
                    <div className="stat-value green">
                        {drivers[0]?.driver_code || drivers[0]?.driver_name?.split(' ').pop() || '—'}
                    </div>
                    <div className="stat-label">Most Likely Winner</div>
                </div>
            </div>

            {/* Expected Position Chart */}
            <div className="card" style={{ marginBottom: 'var(--space-lg)' }}>
                <h3 className="section-title">Expected Finishing Position</h3>
                <div className="chart-container">
                    <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={positionData} margin={{ top: 10, right: 30, bottom: 40, left: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#2A2A2A" />
                            <XAxis
                                dataKey="name"
                                stroke="#6B7280"
                                tick={{ fill: '#9CA3AF', fontSize: 11 }}
                                angle={-45}
                                textAnchor="end"
                            />
                            <YAxis
                                stroke="#6B7280"
                                tick={{ fill: '#9CA3AF', fontSize: 11 }}
                                reversed
                                domain={[0.5, 20.5]}
                                label={{
                                    value: 'Position',
                                    angle: -90,
                                    position: 'insideLeft',
                                    fill: '#6B7280',
                                    fontSize: 12,
                                }}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <Bar dataKey="expected" radius={[4, 4, 0, 0]} maxBarSize={35}>
                                {positionData.map((_, idx) => (
                                    <Cell key={idx} fill={COLORS[idx % COLORS.length]} fillOpacity={0.85} />
                                ))}
                                <ErrorBar
                                    dataKey="std"
                                    width={6}
                                    strokeWidth={2}
                                    stroke="#ccc"
                                />
                            </Bar>
                        </ComposedChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* DNF Risk Comparison */}
            <div className="card">
                <h3 className="section-title">DNF Risk</h3>
                <div className="chart-container" style={{ height: '250px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                            data={drivers.map(d => ({
                                name: d.driver_code || d.driver_name?.split(' ').pop() || `D${d.driver_id}`,
                                dnf: +((d.dnf_probability || d.dnf_rate || 0) * 100).toFixed(1),
                            }))}
                            margin={{ top: 5, right: 20, bottom: 40, left: 0 }}
                        >
                            <CartesianGrid strokeDasharray="3 3" stroke="#2A2A2A" />
                            <XAxis
                                dataKey="name"
                                stroke="#6B7280"
                                tick={{ fill: '#9CA3AF', fontSize: 11 }}
                                angle={-45}
                                textAnchor="end"
                            />
                            <YAxis
                                stroke="#6B7280"
                                tick={{ fill: '#9CA3AF', fontSize: 11 }}
                                tickFormatter={v => `${v}%`}
                            />
                            <Tooltip
                                contentStyle={{
                                    background: 'var(--f1-card)',
                                    border: '1px solid var(--f1-border)',
                                    borderRadius: 'var(--radius-sm)',
                                }}
                                formatter={v => [`${v}%`, 'DNF Risk']}
                            />
                            <Bar dataKey="dnf" fill="#6B7280" radius={[4, 4, 0, 0]} maxBarSize={35} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
}

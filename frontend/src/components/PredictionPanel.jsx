import React from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, Cell, Legend,
} from 'recharts';

const COLORS = [
    '#E10600', '#FF6B35', '#FFD700', '#00E676',
    '#40C4FF', '#AB47BC', '#FF7043', '#26C6DA',
    '#EF5350', '#66BB6A', '#FFA726', '#42A5F5',
    '#EC407A', '#26A69A', '#FFCA28', '#7E57C2',
    '#8D6E63', '#78909C', '#9CCC65', '#29B6F6',
];

/**
 * Prediction panel showing win and podium probability bar charts.
 */
export default function PredictionPanel({ predictions, raceName }) {
    if (!predictions || predictions.length === 0) {
        return (
            <div className="empty-state">
                <div className="icon">🏁</div>
                <p>No predictions available. Select a race and click Predict.</p>
            </div>
        );
    }

    // Sort by expected position and take top 15 for readability
    const sorted = [...predictions]
        .sort((a, b) => a.expected_position - b.expected_position)
        .slice(0, 15);

    const winData = sorted.map(d => ({
        name: d.driver_code || d.driver_name.split(' ').pop(),
        value: +(d.win_probability * 100).toFixed(1),
        fullName: d.driver_name,
    }));

    const podiumData = sorted.map(d => ({
        name: d.driver_code || d.driver_name.split(' ').pop(),
        value: +(d.podium_probability * 100).toFixed(1),
        fullName: d.driver_name,
    }));

    const CustomTooltip = ({ active, payload, label }) => {
        if (!active || !payload?.length) return null;
        return (
            <div style={{
                background: 'var(--f1-card)',
                border: '1px solid var(--f1-border)',
                borderRadius: 'var(--radius-sm)',
                padding: '10px 14px',
                fontSize: '0.85rem',
            }}>
                <div style={{ fontWeight: 600, marginBottom: '4px' }}>
                    {payload[0]?.payload?.fullName || label}
                </div>
                <div style={{ color: payload[0]?.color }}>
                    {payload[0]?.value}%
                </div>
            </div>
        );
    };

    return (
        <div className="grid-2 fade-in">
            {/* Win Probability Chart */}
            <div className="card">
                <h3 className="section-title">Win Probability</h3>
                <div className="chart-container">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={winData} margin={{ top: 5, right: 20, bottom: 40, left: 0 }}>
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
                            <Tooltip content={<CustomTooltip />} />
                            <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={40}>
                                {winData.map((_, idx) => (
                                    <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Podium Probability Chart */}
            <div className="card">
                <h3 className="section-title">Podium Probability</h3>
                <div className="chart-container">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={podiumData} margin={{ top: 5, right: 20, bottom: 40, left: 0 }}>
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
                            <Tooltip content={<CustomTooltip />} />
                            <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={40}>
                                {podiumData.map((_, idx) => (
                                    <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
}

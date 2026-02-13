import React from 'react';

/**
 * Individual driver prediction card showing key stats.
 */
export default function DriverCard({ driver, rank }) {
    const posClass = rank <= 3 ? `pos-${rank}` : '';

    return (
        <div className="card fade-in" style={{ animationDelay: `${rank * 40}ms` }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                        <span className={`pos ${posClass}`} style={{ fontSize: '1.6rem' }}>
                            P{rank}
                        </span>
                        <div>
                            <div className="driver-name" style={{ fontSize: '1.05rem' }}>
                                {driver.driver_name}
                            </div>
                            {driver.constructor && (
                                <div className="driver-constructor">{driver.constructor}</div>
                            )}
                        </div>
                    </div>
                </div>
                {driver.driver_code && (
                    <span style={{
                        fontFamily: 'var(--font-heading)',
                        fontSize: '0.85rem',
                        fontWeight: 700,
                        color: 'var(--f1-text-muted)',
                        background: 'var(--f1-surface)',
                        padding: '4px 10px',
                        borderRadius: 'var(--radius-sm)',
                    }}>
                        {driver.driver_code}
                    </span>
                )}
            </div>

            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: '16px',
                marginTop: '16px',
            }}>
                <StatMini
                    label="Win"
                    value={`${(driver.win_probability * 100).toFixed(1)}%`}
                    color="var(--f1-red)"
                    barWidth={driver.win_probability}
                />
                <StatMini
                    label="Podium"
                    value={`${(driver.podium_probability * 100).toFixed(1)}%`}
                    color="var(--accent-amber)"
                    barWidth={driver.podium_probability}
                />
                <StatMini
                    label="DNF Risk"
                    value={`${(driver.dnf_probability * 100).toFixed(1)}%`}
                    color="var(--f1-text-muted)"
                    barWidth={driver.dnf_probability}
                />
            </div>

            {driver.position_std != null && (
                <div style={{
                    marginTop: '12px',
                    fontSize: '0.75rem',
                    color: 'var(--f1-text-muted)',
                    display: 'flex',
                    gap: '16px',
                }}>
                    <span>Expected: P{driver.expected_position.toFixed(1)}</span>
                    <span>Uncertainty: ±{driver.position_std.toFixed(1)}</span>
                </div>
            )}
        </div>
    );
}

function StatMini({ label, value, color, barWidth }) {
    return (
        <div>
            <div style={{ fontSize: '0.65rem', color: 'var(--f1-text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '4px' }}>
                {label}
            </div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700, fontFamily: 'var(--font-heading)', color }}>
                {value}
            </div>
            <div className="prob-bar-container" style={{ marginTop: '4px' }}>
                <div
                    className="prob-bar"
                    style={{
                        width: `${Math.min(barWidth * 100, 100)}%`,
                        background: color,
                    }}
                />
            </div>
        </div>
    );
}

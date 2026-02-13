/**
 * API client for the F1 Race Intelligence Engine backend.
 */

const API_BASE = import.meta.env.VITE_API_URL || '';

async function fetchJSON(url, options = {}) {
    const response = await fetch(`${API_BASE}${url}`, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
    });
    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || `HTTP ${response.status}`);
    }
    return response.json();
}

export async function getPredictions(seasonYear, roundNumber) {
    return fetchJSON('/predict', {
        method: 'POST',
        body: JSON.stringify({
            season_year: seasonYear,
            round_number: roundNumber,
        }),
    });
}

export async function getSimulation(seasonYear, roundNumber, nSimulations = 5000) {
    const params = new URLSearchParams({
        season_year: seasonYear,
        round_number: roundNumber,
        n_simulations: nSimulations,
    });
    return fetchJSON(`/simulate?${params}`);
}

export async function getBacktest(seasons = [2023], nSimulations = 1000) {
    const params = new URLSearchParams({
        seasons: seasons.join(','),
        n_simulations: nSimulations,
    });
    return fetchJSON(`/backtest?${params}`);
}

export async function getHealth() {
    return fetchJSON('/health');
}

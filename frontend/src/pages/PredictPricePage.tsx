import { useState } from "react";

import { ApiClientError } from "../api/client";
import { predictPrices } from "../api/predictions";
import { PropertyFeatureForm } from "../components/PropertyFeatureForm";
import { PredictionResult } from "../components/PredictionResult";
import type {
  PredictionResponse,
  HouseFeatures,
} from "../features/predictions/predictionTypes";

export function PredictPricePage() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handlePredict(features: HouseFeatures) {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await predictPrices({
        houses: [features],
      });

      setResult(response);
    } catch (err: unknown) {
      if (err instanceof ApiClientError) {
        if (err.status === 503) {
          setError(
            "The prediction model is currently unavailable. Please try again shortly.",
          );
        } else {
          setError(err.message);
        }
      } else {
        setError("Something went wrong while predicting the property price.");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page-container">
      <header className="hero">
        <span className="hero-badge">
          <span className="status-dot" />
          MACHINE LEARNING ESTIMATOR
        </span>

        <h1>
          Discover a home's
          <br />
          <span>estimated value.</span>
        </h1>

        <p>
          Enter the property's features to receive an estimate from your trained
          house-price prediction model.
        </p>
      </header>

      <div className="workspace">
        <section className="form-panel">
          <PropertyFeatureForm loading={loading} onSubmit={handlePredict} />
        </section>

        <aside className="result-panel">
          {loading ? (
            <div className="empty-state" role="status">
              <div className="spinner" />
              <h2>Analyzing property</h2>
              <p>Your features are being processed by the prediction model.</p>
            </div>
          ) : error ? (
            <div className="error-card" role="alert">
              <span className="error-icon">!</span>
              <h2>Prediction unsuccessful</h2>
              <p>{error}</p>
              <button
                type="button"
                className="secondary-button"
                onClick={() => setError(null)}
              >
                Dismiss
              </button>
            </div>
          ) : result ? (
            <PredictionResult result={result} />
          ) : (
            <div className="empty-state">
              <div className="house-icon" aria-hidden="true">
                ⌂
              </div>
              <h2>Your estimate will appear here</h2>
              <p>
                Complete the property form and select “Estimate property price”
                to see the result.
              </p>
              <div className="feature-note">
                <strong>What happens next?</strong>
                <span>Validated inputs</span>
                <span>Trained ML model</span>
                <span>Estimated property value</span>
              </div>
            </div>
          )}
        </aside>
      </div>

      <footer className="page-footer">
        <span>House Price Estimator</span>
        <span>
          Machine learning estimates may differ from actual sale prices.
        </span>
      </footer>
    </main>
  );
}

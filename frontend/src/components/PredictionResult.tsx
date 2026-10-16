import type { PredictionResponse } from "../features/predictions/predictionTypes";

interface PredictionResultProps {
  result: PredictionResponse;
  currency?: string;
  locale?: string;
}

export function PredictionResult({
  result,
  currency = "USD",
  locale = "en-US",
}: PredictionResultProps) {
  const firstPrediction = result.predictions[0];

  if (!firstPrediction) {
    return (
      <section className="result-card" aria-live="polite">
        <h2>No prediction available</h2>
        <p>The API returned no prediction records.</p>
      </section>
    );
  }

  const formattedPrice = new Intl.NumberFormat(locale, {
    style: "currency",
    currency,
    maximumFractionDigits: 0,
  }).format(firstPrediction.predicted_price);

  return (
    <section className="result-card" aria-live="polite">
      <p className="eyebrow">ESTIMATED PROPERTY VALUE</p>
      <h2 className="price-value">{formattedPrice}</h2>

      <p className="result-disclaimer">
        This is a model estimate, not a professional property valuation.
      </p>

      <div className="result-metadata">
        <div>
          <span>Model</span>
          <strong>{result.model_name}</strong>
        </div>
        <div>
          <span>Model version</span>
          <strong>{result.model_version}</strong>
        </div>
        <div>
          <span>Predicted target</span>
          <strong>{result.target}</strong>
        </div>
      </div>

      {result.predictions.length > 1 && (
        <p>{result.predictions.length} property predictions were returned.</p>
      )}
    </section>
  );
}

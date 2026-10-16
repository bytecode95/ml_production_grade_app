import { useState, type FormEvent } from "react";
import type { HouseFeatures } from "../features/predictions/predictionTypes";

interface PropertyFeatureFormProps {
  loading: boolean;
  onSubmit: (features: HouseFeatures) => Promise<void>;
}

const initialValues: HouseFeatures = {
  "Gr Liv Area": 1800,
  "Total Bsmt SF": 900,
  "Garage Cars": 2,
  "Overall Qual": 7,
  "Year Built": 2005,
  "Full Bath": 2,
  "Bedroom AbvGr": 3,
  "Lot Area": 8000,
  "TotRms AbvGrd": 7,
  Fireplaces: 1,
  Neighborhood: "CollgCr",
  "House Style": "2Story",
  "Sale Condition": "Normal",
};

const neighborhoods = [
  "Blmngtn",
  "Blueste",
  "BrDale",
  "BrkSide",
  "ClearCr",
  "CollgCr",
  "Crawfor",
  "Edwards",
  "Gilbert",
  "IDOTRR",
  "MeadowV",
  "Mitchel",
  "Names",
  "NoRidge",
  "NPkVill",
  "NridgHt",
  "NWAmes",
  "OldTown",
  "SWISU",
  "Sawyer",
  "SawyerW",
  "Somerst",
  "StoneBr",
  "Timber",
  "Veenker",
];

const houseStyles = [
  "1Story",
  "1.5Fin",
  "1.5Unf",
  "2Story",
  "2.5Fin",
  "2.5Unf",
  "SFoyer",
  "SLvl",
];

const saleConditions = [
  "Normal",
  "Abnorml",
  "AdjLand",
  "Alloca",
  "Family",
  "Partial",
];

type NumericField =
  | "Gr Liv Area"
  | "Total Bsmt SF"
  | "Garage Cars"
  | "Overall Qual"
  | "Year Built"
  | "Full Bath"
  | "Bedroom AbvGr"
  | "Lot Area"
  | "TotRms AbvGrd"
  | "Fireplaces";

const numericFields: {
  key: NumericField;
  label: string;
  min: number;
  max: number;
  step?: number;
  nullable?: boolean;
}[] = [
  {
    key: "Gr Liv Area",
    label: "Above-ground living area (sq ft)",
    min: 1,
    max: 100000,
  },
  {
    key: "Total Bsmt SF",
    label: "Basement area (sq ft)",
    min: 0,
    max: 100000,
    nullable: true,
  },
  {
    key: "Garage Cars",
    label: "Garage capacity (cars)",
    min: 0,
    max: 20,
    nullable: true,
  },
  {
    key: "Overall Qual",
    label: "Overall quality (1–10)",
    min: 1,
    max: 10,
    step: 1,
  },
  { key: "Year Built", label: "Year built", min: 1800, max: 2100, step: 1 },
  { key: "Full Bath", label: "Full bathrooms", min: 0, max: 20, step: 1 },
  {
    key: "Bedroom AbvGr",
    label: "Bedrooms above ground",
    min: 0,
    max: 50,
    step: 1,
  },
  { key: "Lot Area", label: "Lot area (sq ft)", min: 1, max: 10000000 },
  {
    key: "TotRms AbvGrd",
    label: "Rooms above ground",
    min: 0,
    max: 100,
    step: 1,
  },
  { key: "Fireplaces", label: "Fireplaces", min: 0, max: 20, step: 1 },
];

export function PropertyFeatureForm({
  loading,
  onSubmit,
}: PropertyFeatureFormProps) {
  const [values, setValues] = useState<HouseFeatures>(initialValues);
  const [error, setError] = useState<string | null>(null);

  function updateNumeric(
    key: NumericField,
    rawValue: string,
    nullable = false,
  ) {
    setValues((current) => ({
      ...current,
      [key]:
        nullable && rawValue === ""
          ? null
          : rawValue === ""
            ? Number.NaN
            : Number(rawValue),
    }));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);

    for (const field of numericFields) {
      const value = values[field.key];

      if (value === null && field.nullable) continue;

      if (
        typeof value !== "number" ||
        !Number.isFinite(value) ||
        value < field.min ||
        value > field.max
      ) {
        setError(
          `${field.label} must be between ${field.min} and ${field.max}.`,
        );
        return;
      }

      if (field.step === 1 && !Number.isInteger(value)) {
        setError(`${field.label} must be a whole number.`);
        return;
      }
    }

    if (
      !values.Neighborhood.trim() ||
      !values["House Style"].trim() ||
      !values["Sale Condition"].trim()
    ) {
      setError("Please select all categorical property features.");
      return;
    }

    try {
      await onSubmit(values);
    } catch {
      // The page displays API errors; prevent an unhandled rejection here.
    }
  }

  return (
    <form className="property-form" onSubmit={handleSubmit}>
      <div className="form-section-heading">
        <div>
          <p className="eyebrow">PROPERTY DETAILS</p>
          <h2>Tell us about the home</h2>
        </div>
        <span className="field-count">13 features</span>
      </div>

      <div className="form-grid">
        {numericFields.map((field) => (
          <label className="field" key={field.key}>
            <span>{field.label}</span>
            <input
              type="number"
              name={field.key}
              aria-label={field.label}
              min={field.min}
              max={field.max}
              step={field.step ?? "any"}
              required={!field.nullable}
              value={values[field.key] ?? ""}
              onChange={(event) =>
                updateNumeric(
                  field.key,
                  event.currentTarget.value,
                  field.nullable,
                )
              }
            />
          </label>
        ))}

        <label className="field">
          <span>Neighborhood</span>
          <select
            name="Neighborhood"
            aria-label="Neighborhood"
            value={values.Neighborhood}
            onChange={(event) =>
              setValues((current) => ({
                ...current,
                Neighborhood: event.currentTarget.value,
              }))
            }
            required
          >
            {neighborhoods.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span>House style</span>
          <select
            name="House Style"
            aria-label="House style"
            value={values["House Style"]}
            onChange={(event) =>
              setValues((current) => ({
                ...current,
                "House Style": event.currentTarget.value,
              }))
            }
            required
          >
            {houseStyles.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span>Sale condition</span>
          <select
            name="Sale Condition"
            aria-label="Sale condition"
            value={values["Sale Condition"]}
            onChange={(event) =>
              setValues((current) => ({
                ...current,
                "Sale Condition": event.currentTarget.value,
              }))
            }
            required
          >
            {saleConditions.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
      </div>

      {error && (
        <p className="form-error" role="alert">
          {error}
        </p>
      )}

      <button className="primary-button" type="submit" disabled={loading}>
        {loading ? "Estimating price…" : "Estimate property price"}
      </button>

      <p className="form-footnote">
        Estimates depend on the trained model and the features provided.
      </p>
    </form>
  );
}

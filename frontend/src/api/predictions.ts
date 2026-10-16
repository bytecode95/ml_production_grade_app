import { apiRequest } from "./client";
import type {
    PredictionRequest,
    PredictionResponse,
} from "../features/predictions/predictionTypes";

export function predictPrices(
    request: PredictionRequest,
    signal?: AbortSignal,
): Promise<PredictionResponse> {
    return apiRequest<PredictionResponse>("/predictions", {
        method: "POST",
        body: JSON.stringify(request),
        signal,
    });
}
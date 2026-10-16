export class ApiClientError extends Error {
    constructor(
        message: string,
        public readonly status: number | null,
        public readonly code: string,
    ) {
        super(message);
        this.name = "ApiClientError";
    }
}

const API_BASE_URL = (
    import.meta.env.VITE_API_BASE_URL ??
    "http://127.0.0.1:8000/api/v1"
).replace(/\/+$/, "");

export async function apiRequest<T>(
    path: string,
    options: RequestInit = {},
): Promise<T> {
    let response: Response;

    try {
        response = await fetch(`${API_BASE_URL}${path}`, {
            ...options,
            headers: {
                Accept: "application/json",
                ...(options.body ? { "Content-Type": "application/json" } : {}),
                ...options.headers,
            },
        });
    } catch {
        throw new ApiClientError(
            "Unable to reach the prediction server. Check your connection and try again.",
            null,
            "NETWORK_ERROR",
        );
    }

    if (!response.ok) {
        let message = `The request failed with status ${response.status}.`;
        let code = "HTTP_ERROR";

        try {
            const body: unknown = await response.json();

            if (
                typeof body === "object" &&
                body !== null &&
                "error" in body
            ) {
                const error = (body as {
                    error?: { code?: string; message?: string };
                }).error;

                if (error?.message) message = error.message;
                if (error?.code) code = error.code;
            } else if (
                typeof body === "object" &&
                body !== null &&
                "detail" in body
            ) {
                // FastAPI's default HTTPException response format.
                const detail = (body as { detail?: unknown }).detail;

                if (typeof detail === "string") {
                    message = detail;
                } else if (
                    typeof detail === "object" &&
                    detail !== null &&
                    "message" in detail &&
                    typeof detail.message === "string"
                ) {
                    message = detail.message;
                }
            }
        } catch {
            // Keep the safe fallback message when the error body isn't JSON.
        }

        throw new ApiClientError(message, response.status, code);
    }

    try {
        return (await response.json()) as T;
    } catch {
        throw new ApiClientError(
            "The server returned an invalid response.",
            response.status,
            "INVALID_RESPONSE",
        );
    }
}
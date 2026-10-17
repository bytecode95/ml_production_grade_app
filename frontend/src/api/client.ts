
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
    const normalizedPath = path.startsWith("/")
        ? path
        : `/ ${path} `;

    const url = `${API_BASE_URL}${normalizedPath} `;

    console.groupCollapsed(`[API] ${options.method ?? "GET"} ${url} `);
    console.log("Base URL:", API_BASE_URL);
    console.log("Path:", path);
    console.log("Request body:", options.body ?? "(none)");

    let response: Response;

    try {
        response = await fetch(url, {
            ...options,
            headers: {
                Accept: "application/json",
                ...(options.body
                    ? { "Content-Type": "application/json" }
                    : {}),
                ...options.headers,
            },
        });
    } catch (error) {
        console.error("Fetch failed:", error);
        console.error(
            "Check backend availability, CORS, URL, and browser Network tab.",
        );
        console.groupEnd();

        throw new ApiClientError(
            error instanceof Error
                ? `Network request failed: ${error.message} `
                : "Network request failed. Inspect the browser console.",
            null,
            "NETWORK_ERROR",
        );
    }

    console.log("HTTP status:", response.status, response.statusText);
    console.log("Response URL:", response.url);

    const responseText = await response.text();
    console.log("Raw response:", responseText);

    if (!response.ok) {
        console.error("HTTP request failed.");
        console.groupEnd();

        let message = `Request failed with status ${response.status}.`;

        try {
            const body = JSON.parse(responseText) as {
                detail?: unknown;
                error?: { message?: string; code?: string };
            };

            if (typeof body.detail === "string") {
                message = body.detail;
            } else if (body.error?.message) {
                message = body.error.message;
            }

            throw new ApiClientError(
                message,
                response.status,
                body.error?.code ?? "HTTP_ERROR",
            );
        } catch (error) {
            if (error instanceof ApiClientError) {
                throw error;
            }
        }

        throw new ApiClientError(
            message,
            response.status,
            "HTTP_ERROR",
        );
    }

    try {
        const result = JSON.parse(responseText) as T;
        console.log("Parsed response:", result);
        console.groupEnd();
        return result;
    } catch {
        console.error("Response is not valid JSON.");
        console.groupEnd();

        throw new ApiClientError(
            "The server returned an invalid JSON response.",
            response.status,
            "INVALID_RESPONSE",
        );
    }
}
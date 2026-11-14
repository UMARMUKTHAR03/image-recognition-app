import {
  BatchResponse,
  GradCamResult,
  HealthResponse,
  ImageNetClass,
  ModelInfo,
  PredictionResult,
} from '../types';

export const DEFAULT_API_URL =
  import.meta.env.VITE_API_URL ||
  (window.location.hostname.includes('github.io')
    ? 'https://image-recognition-api-jq01.onrender.com'
    : 'http://localhost:8000');

export function getApiBaseUrl(): string {
  const custom = localStorage.getItem('image_recognition_api_url');
  if (custom && custom.trim().length > 0) {
    const cleaned = custom.trim().replace(/\/$/, '');
    // Automatically migrate users from legacy/temporary render hostnames
    if (cleaned.includes('image-recognition-api-bgzb.onrender.com')) {
      const migrated = 'https://image-recognition-api-jq01.onrender.com';
      localStorage.setItem('image_recognition_api_url', migrated);
      return migrated;
    }
    return cleaned;
  }
  return DEFAULT_API_URL.replace(/\/$/, '');
}

export function setApiBaseUrl(url: string): void {
  localStorage.setItem('image_recognition_api_url', url.trim());
}

export async function checkHealth(): Promise<HealthResponse> {
  const base = getApiBaseUrl();
  const resp = await fetch(`${base}/health`);
  if (!resp.ok) {
    throw new Error(`Health check failed with HTTP ${resp.status}`);
  }
  return resp.json();
}

export async function getModelInfo(): Promise<ModelInfo> {
  const base = getApiBaseUrl();
  const resp = await fetch(`${base}/api/v1/model-info`);
  if (!resp.ok) {
    throw new Error(`Failed to fetch model info (HTTP ${resp.status})`);
  }
  const json = await resp.json();
  return json.data;
}

export async function getClasses(): Promise<ImageNetClass[]> {
  const base = getApiBaseUrl();
  const resp = await fetch(`${base}/api/v1/classes`);
  if (!resp.ok) {
    throw new Error(`Failed to load classes (HTTP ${resp.status})`);
  }
  const json = await resp.json();
  return json.classes;
}

export async function predictImage(
  file: File,
  topK: number = 5
): Promise<PredictionResult> {
  const base = getApiBaseUrl();
  const formData = new FormData();
  formData.append('file', file);
  formData.append('top_k', topK.toString());

  const resp = await fetch(`${base}/api/v1/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!resp.ok) {
    const errorData = await resp.json().catch(() => ({}));
    throw new Error(errorData.detail || `Prediction failed with HTTP ${resp.status}`);
  }

  const json = await resp.json();
  return json.data;
}

export async function predictBatch(
  files: File[],
  topK: number = 5
): Promise<BatchResponse> {
  const base = getApiBaseUrl();
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file);
  });
  formData.append('top_k', topK.toString());

  const resp = await fetch(`${base}/api/v1/predict/batch`, {
    method: 'POST',
    body: formData,
  });

  if (!resp.ok) {
    const errorData = await resp.json().catch(() => ({}));
    throw new Error(errorData.detail || `Batch prediction failed with HTTP ${resp.status}`);
  }

  return resp.json();
}

export async function explainImage(
  file: File,
  targetClassIndex?: number
): Promise<GradCamResult> {
  const base = getApiBaseUrl();
  const formData = new FormData();
  formData.append('file', file);
  if (targetClassIndex !== undefined) {
    formData.append('target_class_index', targetClassIndex.toString());
  }

  const resp = await fetch(`${base}/api/v1/explain`, {
    method: 'POST',
    body: formData,
  });

  if (!resp.ok) {
    const errorData = await resp.json().catch(() => ({}));
    throw new Error(errorData.detail || `Grad-CAM explainability failed with HTTP ${resp.status}`);
  }

  const json = await resp.json();
  return json.data;
}

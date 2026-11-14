export interface TopPrediction {
  class_index: number;
  label: string;
  confidence_percent: number;
  probability: number;
}

export interface PredictionResult {
  predicted_class: string;
  class_index: number;
  confidence_percent: number;
  top_predictions: TopPrediction[];
  is_low_confidence: boolean;
  warning: string | null;
  latency_ms: number;
  image_dimensions: {
    width: number;
    height: number;
  };
  model: string;
  filename: string;
  timestamp?: string;
  previewUrl?: string;
}

export interface BatchItemResult extends Partial<PredictionResult> {
  filename: string;
  status: 'success' | 'error';
  error?: string;
}

export interface BatchResponse {
  status: string;
  total_files: number;
  successful_predictions: number;
  results: BatchItemResult[];
}

export interface GradCamResult {
  filename: string;
  target_class_index: number;
  target_class_label: string;
  heatmap_base64: string;
  overlay_base64: string;
  latency_ms: number;
  feature_layer: string;
  cam_resolution: string;
}

export interface ModelInfo {
  model_name: string;
  architecture: string;
  weights: string;
  num_classes: number;
  device: string;
  input_resolution: string;
  explainability_layer: string;
  has_custom_artifacts: boolean;
  custom_evaluation?: any;
  custom_training_history?: any;
}

export interface HealthResponse {
  status: string;
  app_title: string;
  tagline: string;
  version: string;
  model: string;
  classes_count: number;
}

export interface ImageNetClass {
  index: number;
  label: string;
}

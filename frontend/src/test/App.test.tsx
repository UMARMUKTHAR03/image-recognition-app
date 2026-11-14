import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';
import App from '../App';

// Mock API service calls
vi.mock('../services/api', () => ({
  DEFAULT_API_URL: 'http://localhost:8000',
  getApiBaseUrl: vi.fn(() => 'http://localhost:8000'),
  setApiBaseUrl: vi.fn(),
  checkHealth: vi.fn().mockResolvedValue({
    status: 'online',
    app_title: 'Image Recognition App – Deep Learning Computer Vision Platform',
    tagline: 'See. Recognize. Understand.',
    version: '2.0.0',
    model: 'MobileNetV3-Small',
    classes_count: 1000,
  }),
  getModelInfo: vi.fn().mockResolvedValue({
    model_name: 'MobileNetV3-Small',
    architecture: 'MobileNetV3-Small',
    weights: 'MobileNet_V3_Small_Weights.DEFAULT',
    num_classes: 1000,
    device: 'cpu',
    input_resolution: '224x224',
    explainability_layer: 'features.12',
    has_custom_artifacts: false,
  }),
  getClasses: vi.fn().mockResolvedValue([
    { index: 0, label: 'tench' },
    { index: 1, label: 'goldfish' },
    { index: 207, label: 'golden retriever' },
  ]),
  predictImage: vi.fn().mockResolvedValue({
    predicted_class: 'golden retriever',
    class_index: 207,
    confidence_percent: 89.45,
    top_predictions: [
      { class_index: 207, label: 'golden retriever', confidence_percent: 89.45, probability: 0.8945 },
      { class_index: 208, label: 'Labrador retriever', confidence_percent: 6.20, probability: 0.0620 },
      { class_index: 218, label: 'Welsh springer spaniel', confidence_percent: 1.15, probability: 0.0115 },
      { class_index: 219, label: 'cocker spaniel', confidence_percent: 0.85, probability: 0.0085 },
      { class_index: 220, label: 'Sussex spaniel', confidence_percent: 0.40, probability: 0.0040 },
    ],
    is_low_confidence: false,
    warning: null,
    latency_ms: 18.5,
    image_dimensions: { width: 400, height: 300 },
    model: 'MobileNetV3-Small',
    filename: 'test_dog.jpg',
  }),
  predictBatch: vi.fn().mockResolvedValue({
    status: 'success',
    total_files: 1,
    successful_predictions: 1,
    results: [
      {
        filename: 'batch_dog.jpg',
        status: 'success',
        predicted_class: 'golden retriever',
        class_index: 207,
        confidence_percent: 91.2,
        latency_ms: 20.1,
      },
    ],
  }),
  explainImage: vi.fn().mockResolvedValue({
    filename: 'test_dog.jpg',
    target_class_index: 207,
    target_class_label: 'golden retriever',
    heatmap_base64: 'data:image/jpeg;base64,mockHeatmap',
    overlay_base64: 'data:image/jpeg;base64,mockOverlay',
    latency_ms: 32.4,
    feature_layer: 'features.12',
    cam_resolution: '7x7',
  }),
}));

describe('Image Recognition App Component Suite', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
  });

  it('renders application header and navigation tabs correctly', async () => {
    render(<App />);
    expect(screen.getByText('Image Recognition App')).toBeInTheDocument();
    expect(screen.getByText('Vision Studio')).toBeInTheDocument();
    expect(screen.getByText('Batch Recognition')).toBeInTheDocument();
    expect(screen.getByText('Grad-CAM XAI')).toBeInTheDocument();
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
  });

  it('navigates between application views', async () => {
    render(<App />);

    const dashboardTab = screen.getByText('Dashboard');
    fireEvent.click(dashboardTab);
    expect(screen.getByText('Computer Vision Operations & System Telemetry')).toBeInTheDocument();

    const batchTab = screen.getByText('Batch Recognition');
    fireEvent.click(batchTab);
    expect(screen.getByText('High-Throughput Batch Recognition Studio')).toBeInTheDocument();

    const gradCamTab = screen.getByText('Grad-CAM XAI');
    fireEvent.click(gradCamTab);
    expect(screen.getByText('Explainable AI — Grad-CAM Visual Heatmap')).toBeInTheDocument();
  });

  it('displays ImageNet taxonomy class lookup view', async () => {
    render(<App />);
    const taxonomyTab = screen.getByText('ImageNet Classes');
    fireEvent.click(taxonomyTab);

    await waitFor(() => {
      expect(screen.getByText('ImageNet-1K Taxonomy (1,000 Classes)')).toBeInTheDocument();
    });
  });

  it('navigates to settings and renders backend configuration', async () => {
    render(<App />);
    const settingsTab = screen.getByText('Settings');
    fireEvent.click(settingsTab);

    expect(screen.getByText('Platform Configuration & Gateway Settings')).toBeInTheDocument();
    expect(screen.getByText('Save & Test')).toBeInTheDocument();
  });
});

import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import { toast } from 'sonner';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// API Types
export interface PredictSalaryRequest {
  role?: string;
  skills?: string;
  work_type: string;
  qualification: string;
  experience_mid: number;
  latitude: number;
  longitude: number;
  embedding_1024?: number[];
}

export interface PredictSalaryResponse {
  predicted_salary: number;
}

export interface RecommendJobsRequest {
  role: string;
  skills: string;
  education: string;
  work_type: string;
  experience: number;
  latitude: number;
  longitude: number;
  gender?: string;
  max_recommendations?: number;
}

export interface JobRecommendation {
  Job_Id: string | number;
  Role: string;
  Company: string;
  location: string;
  work_type: string;
  qualification?: string;
  experience: string;
  salary_range?: string;
  latitude: number;
  longitude: number;
  // Optional legacy field (not provided by backend); keep for backward compatibility
  id?: string | number;
  [key: string]: any;
}

export interface RecommendJobsResponse {
  recommendations: JobRecommendation[];
}

export interface PredictAndRecommendRequest {
  role: string;
  skills: string;
  qualification: string;
  work_type: string;
  experience: number;
  latitude: number;
  longitude: number;
  gender?: string;
  max_recommendations?: number;
}

export interface PredictAndRecommendResponse {
  predicted_salary: number;
  recommendations: JobRecommendation[];
}

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000,
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response) {
          const message = error.response.data?.detail || 'An error occurred';
          toast.error(message);
        } else if (error.request) {
          toast.error('No response from server. Please check your connection.');
        } else {
          toast.error('Request failed. Please try again.');
        }
        return Promise.reject(error);
      }
    );
  }

  async predictSalary(data: PredictSalaryRequest): Promise<PredictSalaryResponse> {
    const response = await this.client.post<PredictSalaryResponse>('/api/predict_salary', data);
    return response.data;
  }

  async recommendJobs(data: RecommendJobsRequest): Promise<RecommendJobsResponse> {
    const response = await this.client.post<RecommendJobsResponse>('/api/recommendations', data);
    return response.data;
  }

  async predictAndRecommend(data: PredictAndRecommendRequest): Promise<PredictAndRecommendResponse> {
    const response = await this.client.post<PredictAndRecommendResponse>('/api/predict_and_recommend', data);
    return response.data;
  }

  async getJob(jobId: string | number): Promise<JobRecommendation> {
    const response = await this.client.get<JobRecommendation>(`/api/jobs/${jobId}`);
    return response.data;
  }

  async healthCheck(): Promise<any> {
    const response = await this.client.get('/health');
    return response.data;
  }
}

export const apiClient = new ApiClient();

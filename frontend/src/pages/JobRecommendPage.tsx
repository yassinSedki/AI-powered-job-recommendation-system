import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Card } from '@/components/ui/card';
import MapPicker from '@/components/MapPicker';
import JobCard from '@/components/JobCard';
import { apiClient, PredictAndRecommendRequest, JobRecommendation } from '@/lib/api';
import { toast } from 'sonner';
import { Loader2, DollarSign } from 'lucide-react';

export default function JobRecommendPage() {
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    role: '',
    skills: '',
    qualification: '',
    work_type: '',
    experience: 0,
    gender: '',
    max_recommendations: 10,
    latitude: 36.8065,
    longitude: 10.1815,
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [results, setResults] = useState<{
    salary: number | null;
    jobs: JobRecommendation[];
  }>({ salary: null, jobs: [] });

  const STORAGE_KEY = 'jh:recommendations:v1';

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const saved = JSON.parse(raw);
        if (saved?.formData && saved?.results) {
          setFormData(saved.formData);
          setResults(saved.results);
        }
      }
    } catch (e) {
      // ignore restore errors
    }
  }, []);

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};
    
    if (!formData.role.trim()) newErrors.role = 'Role is required';
    if (!formData.skills.trim()) newErrors.skills = 'Skills are required';
    if (!formData.qualification) newErrors.qualification = 'Education is required';
    if (!formData.work_type) newErrors.work_type = 'Work type is required';
    // removed preference validation
    if (formData.experience < 0) newErrors.experience = 'Experience must be positive';
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      toast.error('Please fill in all required fields');
      return;
    }

    setLoading(true);
    try {
      const request: PredictAndRecommendRequest = {
        role: formData.role,
        skills: formData.skills,
        qualification: formData.qualification,
        work_type: formData.work_type,
        experience: formData.experience,
        latitude: formData.latitude,
        longitude: formData.longitude,
        gender: formData.gender || undefined,
        max_recommendations: formData.max_recommendations,
      };

      const response = await apiClient.predictAndRecommend(request);
      
      setResults({
        salary: response.predicted_salary,
        jobs: response.recommendations,
      });

      // Persist latest successful recommendations and form inputs
      try {
        localStorage.setItem(
          STORAGE_KEY,
          JSON.stringify({
            formData,
            results: { salary: response.predicted_salary, jobs: response.recommendations },
            savedAt: Date.now(),
          })
        );
      } catch (e) {
        // ignore save errors
      }
      
      toast.success('Recommendations found!');
      
      // Scroll to results
      setTimeout(() => {
        document.getElementById('results')?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFormData({
      role: '',
      skills: '',
      qualification: '',
      work_type: '',
      experience: 0,
      gender: '',
      max_recommendations: 10,
      latitude: 36.8065,
      longitude: 10.1815,
    });
    setResults({ salary: null, jobs: [] });
    setErrors({});
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch (e) {
      // ignore clear errors
    }
  };

  const handleLocationSelect = (lat: number, lng: number) => {
    setFormData({ ...formData, latitude: lat, longitude: lng });
    toast.success('Location updated');
  };

  return (
    <div className="min-h-screen py-8 px-4">
      <div className="container max-w-7xl">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Find Your Perfect Job</h1>
          <p className="text-muted-foreground text-lg">
            Tell us about your preferences and we'll find the best matches near you
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Form Section */}
          <Card className="p-6">
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="role">Role *</Label>
                <Input
                  id="role"
                  value={formData.role}
                  onChange={(e) => setFormData({ ...formData, role: e.target.value })}
                  placeholder="e.g., Software Engineer"
                  className={errors.role ? 'border-destructive' : ''}
                />
                {errors.role && <p className="text-sm text-destructive">{errors.role}</p>}
              </div>

              <div className="space-y-2">
                <Label htmlFor="skills">Skills *</Label>
                <Textarea
                  id="skills"
                  value={formData.skills}
                  onChange={(e) => setFormData({ ...formData, skills: e.target.value })}
                  placeholder="e.g., React, TypeScript, Node.js"
                  className={errors.skills ? 'border-destructive' : ''}
                  rows={3}
                />
                {errors.skills && <p className="text-sm text-destructive">{errors.skills}</p>}
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="qualification">Education *</Label>
                  <Select
                    value={formData.qualification}
                    onValueChange={(value) => setFormData({ ...formData, qualification: value })}
                  >
                    <SelectTrigger className={errors.qualification ? 'border-destructive' : ''}>
                      <SelectValue placeholder="Select education" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Bachelor">Bachelor</SelectItem>
                      <SelectItem value="Master">Master</SelectItem>
                      <SelectItem value="PhD">PhD</SelectItem>
                    </SelectContent>
                  </Select>
                  {errors.qualification && <p className="text-sm text-destructive">{errors.qualification}</p>}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="work_type">Work Type *</Label>
                  <Select
                    value={formData.work_type}
                    onValueChange={(value) => setFormData({ ...formData, work_type: value })}
                  >
                    <SelectTrigger className={errors.work_type ? 'border-destructive' : ''}>
                      <SelectValue placeholder="Select type" />
                    </SelectTrigger>
                    <SelectContent>
                       <SelectItem value="Contract">Contract</SelectItem>
                       <SelectItem value="Part-Time">Part-Time</SelectItem>
                       <SelectItem value="Intern">Intern</SelectItem>
                       <SelectItem value="Full-Time">Full-Time</SelectItem>
                     </SelectContent>
                  </Select>
                  {errors.work_type && <p className="text-sm text-destructive">{errors.work_type}</p>}
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="experience">Experience (years) *</Label>
                  <Input
                    id="experience"
                    type="number"
                    min="0"
                    value={formData.experience}
                    onChange={(e) => setFormData({ ...formData, experience: parseInt(e.target.value) || 0 })}
                    className={errors.experience ? 'border-destructive' : ''}
                  />
                  {errors.experience && <p className="text-sm text-destructive">{errors.experience}</p>}
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="gender">Gender (Optional)</Label>
                  <Select
                    value={formData.gender}
                    onValueChange={(value) => setFormData({ ...formData, gender: value })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select gender" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="male">Male</SelectItem>
                      <SelectItem value="female">Female</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="max_recommendations">Max Results</Label>
                  <Input
                    id="max_recommendations"
                    type="number"
                    min="1"
                    max="50"
                    value={formData.max_recommendations}
                    onChange={(e) => setFormData({ ...formData, max_recommendations: parseInt(e.target.value) || 10 })}
                  />
                </div>
              </div>


              <div className="flex gap-4">
                <Button
                  type="submit"
                  disabled={loading}
                  className="flex-1 bg-gradient-to-r from-primary to-accent"
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Finding matches...
                    </>
                  ) : (
                    'Get Recommendations'
                  )}
                </Button>
                <Button type="button" variant="outline" onClick={handleReset}>
                  Reset
                </Button>
              </div>
            </form>
          </Card>

          {/* Map Section */}
          <div className="space-y-4">
            <div>
              <h2 className="text-xl font-semibold mb-2">Select Your Location</h2>
              <p className="text-sm text-muted-foreground mb-4">
                Click on the map to set your coordinates
              </p>
            </div>
            <MapPicker
              latitude={formData.latitude}
              longitude={formData.longitude}
              onLocationSelect={handleLocationSelect}
              markers={results.jobs.map((job) => ({
                id: job.Job_Id ?? job.id,
                lat: job.latitude,
                lng: job.longitude,
                title: job.Role,
                company: job.Company,
              }))}
            />
          </div>
        </div>

        {/* Results Section */}
        {(results.salary !== null || results.jobs.length > 0) && (
          <div id="results" className="mt-12 space-y-8">
            {results.salary !== null && (
              <Card className="p-6 bg-gradient-to-r from-secondary to-secondary/80">
                <div className="flex items-center gap-4">
                  <div className="bg-white p-3 rounded-lg">
                    <DollarSign className="w-8 h-8 text-secondary" />
                  </div>
                  <div>
                    <p className="text-sm text-secondary-foreground/80 font-medium">
                      Estimated Salary
                    </p>
                    <p className="text-3xl font-bold text-secondary-foreground">
                      ${results.salary.toLocaleString()} / year
                    </p>
                  </div>
                </div>
              </Card>
            )}

            {results.jobs.length > 0 && (
              <div>
                <h2 className="text-2xl font-bold mb-6">
                  Recommended Jobs ({results.jobs.length})
                </h2>
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {results.jobs.map((job) => (
                    <JobCard key={job.Job_Id ?? job.id} job={job} />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
